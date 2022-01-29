import argparse
import unittest
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from voice100.datasets import AudioTextDataModule

ARGS = "--dataset kokoro_small --language ja"
ARGS = "--use_phone --batch_size 2"
ARGS = "--batch_size 2 --dataset librispeech --language en"


class DatasetTest(unittest.TestCase):
    def test_help(self):
        parser = argparse.ArgumentParser()
        AudioTextDataModule.add_argparse_args(parser)
        # self.assertRaises(SystemExit, parser.parse_args("--help".split()))
        try:
            parser.parse_args("--help".split())
        except SystemExit:
            pass

    def test_dataset(self):
        from voice100.datasets import get_dataset
        ds = get_dataset("ljspeech", split="train", use_phone=True)
        for x, y, z in ds:
            pass  # print(x, y, z)

    def test_data_module(self):
        parser = argparse.ArgumentParser()
        AudioTextDataModule.add_argparse_args(parser)
        args = parser.parse_args(ARGS.split())
        data: AudioTextDataModule = AudioTextDataModule.from_argparse_args(args, vocoder="mel")
        data.setup()

        print(f"audio_size={data.audio_size}")
        print(f"vocab_size={data.vocab_size}")
        use_phone = '--use_phone' in ARGS
        expected_vocab_size = 77 if use_phone else 29
        expected_audio_size = 64
        self.assertEqual(data.vocab_size, expected_vocab_size)
        self.assertEqual(data.audio_size, expected_audio_size)

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
                    print(data.text_transform.tokenizer.decode(t))
            break


if __name__ == "__main__":
    # DatasetTest().test_dataset()
    DatasetTest().test_data_module()
    # DatasetTest().test()
