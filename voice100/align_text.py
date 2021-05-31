# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import torch
import os
from tqdm import tqdm

from .models.align import AudioAlignCTC
from .datasets import AlignInferDataModule

def cli_main():
    parser = ArgumentParser()
    parser = AlignInferDataModule.add_data_specific_args(parser)
    args = parser.parse_args()
    args.output = f'data/{args.dataset}-align.txt'

    data = AlignInferDataModule.from_argparse_args(args)
    model = AudioAlignCTC.load_from_checkpoint('model/align_en_lstm_base_ctc.ckpt')
    data.setup()
    encoder = data.transform.encoder
    model.eval()
    with open(args.output, 'w') as f:
        for idx, batch in enumerate(tqdm(data.infer_dataloader())):
            (audio, audio_len), (text, text_len) = batch
            score, path, path_len = model.ctc_best_path(audio, audio_len, text, text_len)
            file = os.path.join(args.cache, 'align-%d.pt' % idx)
            torch.save({
                'score': score, 'path': path, 'path_len': path_len
            }, file)
        
            for i in range(path.shape[0]):
                raw_text = encoder.decode(path[i, :path_len[i]])
                f.write(raw_text + '\n')

if __name__ == '__main__':
    cli_main()
