# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from voice100.vocoder import WORLDVocoder
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from .models.tts import CharToAudioModel
from .datasets import AudioTextDataModule
from .models.tts import get_padding_mask

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_data_specific_args(parser)
    parser = CharToAudioModel.add_model_specific_args(parser)
    parser.add_argument('--calc_stat', action='store_true', help='Calculate WORLD statistics')
    parser.add_argument('--align', action='store_true', help='')
    parser.set_defaults(task='tts')
    args = parser.parse_args()

    if args.calc_stat:
        calc_stat(args)
    elif args.test:
        if args.align:
            test_align(args)
        else:
            test_force_align(args)
    else:
        data = AudioTextDataModule.from_argparse_args(args)
        #model = CharToAudioModel.from_argparse_args(args)
        model = CharToAudioModel.load_from_checkpoint('a.pt', strict=False)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True, period=10)
        from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
        monitor_callback = LearningRateMonitor()
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=[monitor_callback, checkpoint_callback])
        trainer.fit(model, data)

def calc_stat(args):
    from tqdm import tqdm
    data = AudioTextDataModule.from_argparse_args(args)
    data.setup()
    vocoder: WORLDVocoder = data.transform.vocoder

    logspc_size = 257
    codeap_size = 1

    f0_sum = torch.zeros(1, dtype=torch.double)
    logspc_sum = torch.zeros(logspc_size, dtype=torch.double)
    codeap_sum = torch.zeros(codeap_size, dtype=torch.double)
    f0_sqrsum = torch.zeros(1, dtype=torch.double)
    logspc_sqrsum = torch.zeros(logspc_size, dtype=torch.double)
    codeap_sqrsum = torch.zeros(codeap_size, dtype=torch.double)
    f0_count = 0
    logspc_count = 0
    for batch_idx, batch in enumerate(tqdm(data.train_dataloader())):
        (f0, f0_len, mcep, codeap), (text, text_len), (aligntext, aligntext_len) = batch
        with torch.no_grad():
            logspc = mcep @ vocoder.mc2sp_matrix
            mask = get_padding_mask(f0, f0_len)
            f0mask = (f0 > 30.0).float() * mask

            f0_sum += torch.sum(f0 * f0mask)
            f0_sqrsum += torch.sum(f0 ** 2 * f0mask)
            f0_count += torch.sum(f0mask)

            logspc_sum += torch.sum(torch.sum(logspc * mask[:, :, None], axis=1), axis=0)
            logspc_sqrsum += torch.sum(torch.sum(logspc ** 2 * mask[:, :, None], axis=1), axis=0)
            logspc_count += torch.sum(mask)

            codeap_sum += torch.sum(torch.sum(codeap * mask[:, :, None], axis=1), axis=0)
            codeap_sqrsum += torch.sum(torch.sum(codeap ** 2 * mask[:, :, None], axis=1), axis=0)

    codeap_count = logspc_count
    state_dict = {
        'f0_mean': f0_sum / f0_count,
        'f0_std': torch.sqrt((f0_sqrsum / f0_count) - (f0_sum / f0_count) ** 2),
        'logspc_mean': logspc_sum / logspc_count,
        'logspc_std': torch.sqrt((logspc_sqrsum / logspc_count) - (logspc_sum / logspc_count) ** 2),
        'codeap_mean': codeap_sum / codeap_count,
        'codeap_std': torch.sqrt((codeap_sqrsum / codeap_count) - (codeap_sum / codeap_count) ** 2),
    }
    print('saving...')
    torch.save(state_dict, f'data/stat_{args.dataset}.pt')

def test_force_align(args):

    assert args.resume_from_checkpoint
    data = AudioTextDataModule.from_argparse_args(args)
    model = CharToAudioModel.load_from_checkpoint(args.resume_from_checkpoint)
    from .text import CharTokenizer
    tokenizer = CharTokenizer()
    use_cuda = args.gpus and args.gpus > 0
    if use_cuda:
        print('cuda')
        model.cuda()
    model.eval()
    data.setup()
    from tqdm import tqdm
    for batch in data.test_dataloader():
        (f0, f0_len, spec, codeap), (text, text_len), (aligntext, aligntext_len) = batch
        print('===')
        tgt_in = aligntext
        if use_cuda:
            text = text.cuda()
            text_len = text_len.cuda()
            tgt_in = tgt_in.cuda()
        if True:
            logits, hasf0_hat, f0_hat, spec_hat, codeap_hat = model.forward(text, text_len, tgt_in)
            f0_hat
            f0_hat, spec_hat, codeap_hat = model.world_norm.unnormalize(f0_hat, spec_hat, codeap_hat)
        else:
            logits, hasf0_hat, f0_hat, spec_hat, codeap_hat = model.forward(text, text_len, tgt_in)
        tgt_out = logits.argmax(axis=-1)
        tgt_in = torch.cat([tgt_in, tgt_out[:, -1:]], axis=1)
        for j in range(text.shape[0]):
            print('---')
            print('S:', tokenizer.decode(text[j, :]))
            print('T:', tokenizer.decode(aligntext[j, :]))
            print('H:', tokenizer.decode(tgt_out[j, :]))

def test_align(args):

    assert args.resume_from_checkpoint
    data = AudioTextDataModule.from_argparse_args(args)
    model = CharToAudioModel.load_from_checkpoint(args.resume_from_checkpoint)

    from .text import CharTokenizer
    tokenizer = CharTokenizer()
    use_cuda = args.gpus and args.gpus > 0
    if use_cuda:
        print('cuda')
        model.cuda()
    model.eval()
    data.setup()

    for batch in data.test_dataloader():
        (f0, f0_len, spec, codeap), (text, text_len), (aligntext, aligntext_len) = batch
        print('===')
        with torch.no_grad():
            aligntext_hat = model.predict(text, text_len, max_steps=100)

        if True:
            for j in range(text.shape[0]):
                print('---')
                print('S:', tokenizer.decode(text[j, :]))
                print('T:', tokenizer.decode(aligntext[j, :]))
                print('H:', tokenizer.decode(aligntext_hat[j, :]))
        break
        if True:
            for i in range(f0.shape[0]):
                print('---')
                x = aligntext[i, :f0_len[i]]
                x = tokenizer.decode(x)
                x = tokenizer.merge_repeated(x)
                print(x)
                x = text[i, :text_len[i]]
                x = tokenizer.decode(x)
                print(x)

if __name__ == '__main__':
    cli_main()