# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import torch
import os
from tqdm import tqdm

from .models import AudioToAlignText
from .data_modules import AudioTextDataModule


def cli_main():
    parser = ArgumentParser()
    parser = AudioTextDataModule.add_argparse_args(parser)
    parser.add_argument("--checkpoint", required=True, type=str, help="Load from checkpoint")
    parser.add_argument("--split", default="train", type=str, help="Split to use")
    parser.set_defaults(vocoder="mel")
    args = parser.parse_args()
    args.write_cache = False
    args.timing = True
    if args.use_phone:
        args.output = f'data/{args.dataset}-phone-align-{args.split}.txt'
    else:
        args.output = f'data/{args.dataset}-align-{args.split}.txt'

    data: AudioTextDataModule = AudioTextDataModule.from_argparse_args(args)
    model = AudioToAlignText.load_from_checkpoint(args.checkpoint)
    if args.split == "train":
        data.setup("predict")
    else:
        data.setup()
    tokenizer = data.text_transform
    model.eval()
    with open(args.output, 'w') as f:
        if args.split == "train":
            loader = data.predict_dataloader()
        elif args.split == "valid":
            loader = data.val_dataloader()
        for idx, batch in enumerate(tqdm(loader)):
            (audio, audio_len), (text, text_len) = batch
            score, hist, path, path_len = model.ctc_best_path(audio, audio_len, text, text_len)
            if args.write_cache:
                file = os.path.join(args.cache, 'align-%d.pt' % idx)
                torch.save({
                    'score': score, 'path': path, 'path_len': path_len
                }, file)

            for i in range(path.shape[0]):
                align = [0] * (2 * text_len[i] + 1)
                for j in hist[i, :path_len[i]]:
                    align[j] += 1
                align = ' '.join([str(x) for x in align])
                raw_text = tokenizer.decode(text[i, :text_len[i]])

                raw_align_text = tokenizer.decode(path[i, :path_len[i]])
                f.write(raw_text + '|' + raw_align_text + '|' + align + '\n')


if __name__ == '__main__':
    cli_main()
