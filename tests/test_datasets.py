from typing import Text
import os
import random
import numpy as np
import soundfile as sf
import argparse
import unittest
from tqdm import tqdm
from tempfile import TemporaryDirectory
import torch
import pytest
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from voice100.datasets import (
    get_dataset,
    MergeDataset,
    MetafileDataset,
    TextDataset,
    AudioTextDataModule
)

ARGS = "--dataset kokoro_small --language ja"
ARGS = "--use_phone --batch_size 2"
ARGS = "--batch_size 2 --dataset librispeech --language en"

FAKE_VOCAB = "hello world good bye morning".split()
FAKE_PHONE = "HH/AH0/L/OW1 W/ER1/L/D G/UH1/D B/AY1 M/AO1/R/N/IH0/NG".split()

# <tmpdir>/fake/metadata.csv
# <tmpdir>/fake/wavs/xxxx.wav
# <tmpdir>/fake-phone-train.txt
# <tmpdir>/fake-align-train.txt
# <tmpdir>/fake-phone-align-train.txt


def make_random_text():
    k = random.randint(a=1, b=10)
    return " ".join(random.choices(FAKE_VOCAB, k=k)).title()


def make_random_phone():
    k = random.randint(a=1, b=10)
    return "/ /".join(random.choices(FAKE_PHONE, k=k)).title()


def make_random_audio(
    file: Text, duration: float = 5.0, rate: int = 16000
) -> None:
    t = np.linspace(0, duration, int(duration * rate))
    freq = random.randint(100, 1000)
    waveform = []
    k = random.randint(a=2, b=10)
    for i in range(1, k):
        amp = random.random() * np.exp(-0.2 * i)
        phrase = random.random() * 2 * np.pi
        x = amp * np.sin(2 * np.pi * i * freq * t + phrase)
        waveform.append(x)
    waveform = np.sum(waveform, axis=0)
    sf.write(file, waveform, rate)


def make_fake_metadataset(dirpath: Text, n: int = 10) -> None:
    metafile_file = os.path.join(dirpath, "metadata.csv")
    wavs_path = os.path.join(dirpath, "wavs")
    os.makedirs(wavs_path)
    with open(metafile_file, "wt") as metafile_fp:
        for i in range(n):
            clipid = f"FAKE001-{i:04}"
            audiopath = os.path.join(wavs_path, clipid + ".wav")
            text = make_random_text()
            transcript = text.lower()
            metafile_fp.write(f"{clipid}|{text}|{transcript}\n")
            make_random_audio(audiopath)


def make_fake_aligntext_dataset(dirpath: Text, is_phone: bool, n: int = 10) -> None:
    if is_phone:
        text_file = os.path.join(dirpath, "fake-phone-align-train.txt")
    else:
        text_file = os.path.join(dirpath, "fake-align-train.txt")

    with open(text_file, "wt") as phone_fp:
        for i in range(n):
            text = make_random_text()
            if is_phone:
                text = make_random_phone()
                text = text.replace("/", "/_/")
            else:
                text = "_".join(make_random_text())
            phone_fp.write(f"{text}|{text}|0,1,2,3\n")


def make_fake_phonetext_dataset(dirpath: Text, n: int = 10) -> None:
    text_file = os.path.join(dirpath, "fake-phone-train.txt")

    with open(text_file, "wt") as phone_fp:
        for i in range(n):
            clipid = f"FAKE001-{i:04}"
            text = make_random_phone()
            phone_fp.write(f"{clipid}|{text}\n")


class DatasetTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tempdir = TemporaryDirectory()
        self.data_path = (self.tempdir.name)
        self.fake_path = os.path.join(self.data_path, "fake")
        make_fake_metadataset(self.fake_path)
        make_fake_aligntext_dataset(self.data_path, is_phone=False)
        make_fake_aligntext_dataset(self.data_path, is_phone=True)
        make_fake_phonetext_dataset(self.data_path)

    def doCleanups(self) -> None:
        super().doCleanups()
        self.tempdir.cleanup()

    def test_metafile_dataset(self):
        dataset = MetafileDataset(
            self.fake_path, metafile="metadata.csv", header=False,
            idcol=0, textcol=2)
        self.assertEqual(10, len(dataset))
        for clipid, audiopath, text in dataset:
            self.assertTrue(clipid.startswith("FAKE001-"))
            waveform, rate = sf.read(audiopath)
            self.assertIsInstance(waveform, np.ndarray)
            self.assertEqual(16000, rate)
            self.assertIsInstance(text, Text)

    def test_text_dataset(self):
        phone_file = os.path.join(self.data_path, "fake-phone-train.txt")
        phone_ds = TextDataset(phone_file)
        self.assertEqual(10, len(phone_ds))
        for clipid, phone_text in phone_ds:
            self.assertTrue(clipid.startswith("FAKE001-"))
            self.assertIsInstance(phone_text, Text)

    def test_merge_dataset_align(self):
        dataset = MetafileDataset(
            self.fake_path, metafile="metadata.csv", header=False,
            idcol=0, textcol=2)
        self.assertEqual(10, len(dataset))
        alignfile = os.path.join(self.data_path, "fake-align-train.txt")
        align_ds = TextDataset(alignfile, idcol=-1, textcol=1)
        self.assertEqual(10, len(align_ds))
        ds = MergeDataset(dataset, align_ds=align_ds)
        self.assertEqual(10, len(ds))
        for clipid, audiopath, phonetext in ds:
            self.assertTrue(clipid.startswith("FAKE001-"))
            waveform, rate = sf.read(audiopath)
            self.assertIsInstance(waveform, np.ndarray)
            self.assertEqual(16000, rate)
            self.assertIsInstance(phonetext, Text)

    def test_merge_dataset_phone(self):
        dataset = MetafileDataset(
            self.fake_path, metafile="metadata.csv", header=False,
            idcol=0, textcol=2)
        self.assertEqual(10, len(dataset))
        phone_file = os.path.join(self.data_path, "fake-phone-train.txt")
        phone_ds = TextDataset(phone_file)
        self.assertEqual(10, len(phone_ds))
        ds = MergeDataset(dataset, phone_ds=phone_ds)
        self.assertEqual(10, len(ds))
        for clipid, audiopath, phonetext in ds:
            self.assertTrue(clipid.startswith("FAKE001-"))
            waveform, rate = sf.read(audiopath)
            self.assertIsInstance(waveform, np.ndarray)
            self.assertEqual(16000, rate)
            self.assertIsInstance(phonetext, Text)

    def test_help(self):
        parser = argparse.ArgumentParser()
        AudioTextDataModule.add_argparse_args(parser)
        # self.assertRaises(SystemExit, parser.parse_args("--help".split()))
        try:
            parser.parse_args("--help".split())
        except SystemExit:
            pass

    @unittest.skipUnless(os.path.exists("./data/LJSpeech-1.1"), "Need LJSpeech-1.1 dataset")
    def test_dataset(self):
        from voice100.datasets import get_dataset
        ds = get_dataset("ljspeech", split="train", use_phone=True)
        for x, y, z in ds:
            pass  # print(x, y, z)

    @unittest.skipUnless(os.path.exists("./data/LJSpeech-1.1"), "Need LJSpeech-1.1 dataset")
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


@pytest.mark.skip("dataset are needed")
def test_data_module():
    ARGS = "--dataset cv_ja --language ja --use_phone --vocoder mel"
    parser = argparse.ArgumentParser()
    AudioTextDataModule.add_argparse_args(parser)
    args = parser.parse_args(ARGS.split())
    data: AudioTextDataModule = AudioTextDataModule.from_argparse_args(args)
    data.setup()

    print(f"audio_size={data.audio_size}")
    print(f"vocab_size={data.vocab_size}")
    use_phone = '--use_phone' in ARGS
    expected_vocab_size = 44 if use_phone else 29
    expected_audio_size = 64
    assert data.vocab_size == expected_vocab_size
    assert data.audio_size == expected_audio_size

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
                assert text_len[i] > 0
                if text_len[i] < 5:
                    t = text[i, :text_len[i]]
                    print(data.text_transform.tokenizer.decode(t))
        #break


@pytest.mark.skip("dataset are needed")
def test_kokoro_dataset():
    dataset = get_dataset(
        dataset="kokoro_small", split="train",
        use_align=False, use_phone=False, use_target=False)
    s = set()
    for i in range(len(dataset)):
        clip_id, file, text = dataset[i]
        # print(clip_id, file, text)
        s.update(text.split(' '))
    print(sorted(s))


if __name__ == "__main__":
    # DatasetTest().test_dataset()
    # DatasetTest().test_data_module()
    test_data_module()