import unittest
import argparse
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from voice100.datasets import AudioTextDataModule
from voice100.text import CMUTokenizer, CharTokenizer

ARGS = "--dataset kokoro_small --language ja"
ARGS = "--use_phone --batch_size 2"


class DatasetTest(unittest.TestCase):
    def test_help(self):
        parser = argparse.ArgumentParser()
        AudioTextDataModule.add_argparse_args(parser)
        _ = parser.parse_args("--help".split())

    def test_dataset(self):
        from voice100.datasets import get_dataset
        ds = get_dataset("ljspeech", use_phone=True)
        for x, y, z in ds:
            print(x, y, z)

    def test_data_module(self):
        parser = argparse.ArgumentParser()
        AudioTextDataModule.add_argparse_args(parser)
        args = parser.parse_args(ARGS.split())
        data = AudioTextDataModule.from_argparse_args(args, task="asr")
        data.setup()
        # tokenizer = CharTokenizer()
        tokenizer = CMUTokenizer()

        for batch in tqdm(data.train_dataloader()):
            # print(batch)
            with torch.no_grad():
                (audio, audio_len), (text, text_len) = batch
                packed_audio = pack_padded_sequence(audio, audio_len, batch_first=True, enforce_sorted=False)
                _ = pad_packed_sequence(packed_audio, batch_first=False)
                assert not (torch.any(audio.isnan()))
                assert (torch.min(audio_len) > 0)
                assert (torch.max(audio_len) == audio.shape[1]), f'{audio_len} {audio.shape}'

                for i in range(text.shape[0]):
                    t = text[i, :text_len[i]]
                    print(tokenizer.decode(t))
            break

if __name__ == "__main__":
    # DatasetTest().test_dataset()
    DatasetTest().test_data_module()
    # DatasetTest().test()
