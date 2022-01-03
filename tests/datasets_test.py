import argparse
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from voice100.datasets import AudioTextDataModule
parser = argparse.ArgumentParser()
AudioTextDataModule.add_argparse_args(parser)
# args = parser.parse_args("--help".split())
args = parser.parse_args("--dataset kokoro_small --language ja".split())
# data = ASRDataModule('librispeech', 0.1, 'en', 10, 'cache', 128)
data = AudioTextDataModule.from_argparse_args(args, task="asr")
data.setup()

from voice100.audio import BatchSpectrogramAugumentation
augument = BatchSpectrogramAugumentation()

import random
for i in range(1000000):
    for batch in tqdm(data.train_dataloader()):
        #print(batch)
        with torch.no_grad():
            (audio, audio_len), (text, text_len) = batch
            audio1, audio_len1 = augument(audio, audio_len)
            packed_audio = pack_padded_sequence(audio1, audio_len1, batch_first=True, enforce_sorted=False)
            x = pad_packed_sequence(packed_audio, batch_first=False)

            if False:
                rate = 0.5 + random.random()
                i = (torch.arange(int(audio.shape[1] * rate), device=audio.device) / rate).int()
                audio1 = torch.index_select(audio, 1, i)
                audio_len1 = (audio_len * rate).int()
            assert not (torch.any(audio1.isnan()))
            assert (torch.min(audio_len1) > 0)
            assert (torch.max(audio_len1) == audio1.shape[1]), f'{audio_len1} {audio1.shape}'