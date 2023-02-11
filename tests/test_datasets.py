from typing import Text
import os
import random
import math
import argparse
import unittest
import torchaudio
from tqdm import tqdm
from tempfile import TemporaryDirectory
import torch
import pytest
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from voice100.data_modules import (
    get_dataset,
    MergeDataset,
    MetafileDataset,
    TextDataset,
    AudioTextDataModule
)

ARGS = "--dataset kokoro_small --language ja"
ARGS = "--use_phone --batch_size 2"
ARGS = "--batch_size 2 --dataset librispeech --language en"

DUMMY_VOCAB = "hello world good bye morning".split()
DUMMY_PHONE = "HH/AH0/L/OW1 W/ER1/L/D G/UH1/D B/AY1 M/AO1/R/N/IH0/NG".split()
DUMMY_JA_CHARS = [chr(code) for code in range(ord('ぁ'), ord('ん') + 1)] + list('。、！？')
DUMMY_JA_PHONE = "a i u e o a: i: u: e: o: k g s sh N q . !".split()

# <tmpdir>/dummy/metadata.csv
# <tmpdir>/dummy/wavs/xxxx.wav
# <tmpdir>/dummy-phone-train.txt
# <tmpdir>/dummy-align-train.txt
# <tmpdir>/dummy-phone-align-train.txt


def make_random_text(language: Text) -> Text:
    if language == "en":
        text_len = random.randint(a=1, b=10)
        text = " ".join(random.choices(DUMMY_VOCAB, k=text_len)).title()
    elif language == "ja":
        text_len = random.randint(a=1, b=30)
        text = ''.join(random.choices(DUMMY_JA_CHARS, k=text_len))
    return text


def make_random_phone(language: Text):
    if language == "en":
        text_len = random.randint(a=1, b=10)
        return "/ /".join(random.choices(DUMMY_PHONE, k=text_len))
    elif language == "ja":
        text_len = random.randint(a=0, b=30)
        text = ' '.join(random.choices(DUMMY_JA_PHONE, k=text_len))
    return text


def make_random_align_phone(language: Text):
    if language == "en":
        text_len = random.randint(a=1, b=10)
        text = "/ /".join(random.choices(DUMMY_PHONE, k=text_len))
        text = text.replace("/", "/_/")
    elif language == "ja":
        text_len = random.randint(a=1, b=30)
        text = ' '.join(random.choices(DUMMY_JA_PHONE, k=text_len))
        text = text.replace(" ", " - ")
    return text


def make_random_audio(
    file: Text, duration: float = 5.0, rate: int = 16000
) -> None:
    t = torch.linspace(0, duration, int(duration * rate))
    freq = random.randint(100, 1000)
    waveform = []
    k = random.randint(a=2, b=10)
    for i in range(1, k):
        amp = random.random() * math.exp(-0.2 * i)
        phrase = random.random() * 2 * math.pi
        x = amp * torch.sin(2 * math.pi * i * freq * t + phrase)
        waveform.append(x)
    waveform = torch.sum(torch.stack(waveform), dim=0, keepdims=True)
    torchaudio.save(file, waveform, rate)


def make_dummy_metadataset(dirpath: Text, language: Text, n: int = 10) -> None:
    metafile_file = os.path.join(dirpath, "metadata.csv")
    wavs_path = os.path.join(dirpath, "wavs")
    os.makedirs(wavs_path)
    with open(metafile_file, "wt") as metafile_fp:
        for i in range(n):
            clipid = f"dummy001-{i:04}"
            audiopath = os.path.join(wavs_path, clipid + ".wav")
            text = make_random_text(language=language)
            transcript = text.lower()
            metafile_fp.write(f"{clipid}|{text}|{transcript}\n")
            make_random_audio(audiopath)


def make_dummy_aligntext_dataset(dirpath: Text, language: Text, is_phone: bool, n: int = 10) -> None:
    if is_phone:
        text_file = os.path.join(dirpath, f"dummy_{language}-phone-align-train.txt")
    else:
        text_file = os.path.join(dirpath, f"dummy_{language}-align-train.txt")

    with open(text_file, "wt") as phone_fp:
        for i in range(n):
            text = make_random_text(language=language)
            if is_phone:
                aligntext = make_random_align_phone(language=language)
            else:
                aligntext = "_".join(make_random_text(language=language))
            phone_fp.write(f"{text}|{aligntext}|0 1 2 3\n")


def make_dummy_phonetext_dataset(dirpath: Text, language: Text, n: int = 10) -> None:
    text_file = os.path.join(dirpath, f"dummy_{language}-phone-train.txt")

    with open(text_file, "wt") as phone_fp:
        for i in range(n):
            clipid = f"dummy001-{i:04}"
            text = make_random_phone(language=language)
            phone_fp.write(f"{clipid}|{text}\n")


class DatasetTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.language = "en"
        self.tempdir = TemporaryDirectory()
        self.data_path = (self.tempdir.name)
        self.dummy_path = os.path.join(self.data_path, f"dummy-speech-{self.language}")
        make_dummy_metadataset(self.dummy_path, language=self.language)
        make_dummy_aligntext_dataset(self.data_path, language=self.language, is_phone=False)
        make_dummy_aligntext_dataset(self.data_path, language=self.language, is_phone=True)
        make_dummy_phonetext_dataset(self.data_path, language=self.language)

    def doCleanups(self) -> None:
        super().doCleanups()
        self.tempdir.cleanup()

    def test_metafile_dataset(self):
        dataset = MetafileDataset(
            self.dummy_path, metafile="metadata.csv", header=False,
            idcol=0, textcol=2)
        self.assertEqual(10, len(dataset))
        for clipid, audiopath, text in dataset:
            self.assertTrue(clipid.startswith("dummy001-"))
            waveform, rate = torchaudio.load(audiopath)
            self.assertIsInstance(waveform, torch.Tensor)
            self.assertEqual(16000, rate)
            self.assertIsInstance(text, Text)

    def test_text_dataset(self):
        phone_file = os.path.join(self.data_path, f"dummy_{self.language}-phone-train.txt")
        phone_ds = TextDataset(phone_file)
        self.assertEqual(10, len(phone_ds))
        for clipid, phone_text in phone_ds:
            self.assertTrue(clipid.startswith("dummy001-"))
            self.assertIsInstance(phone_text, Text)

    def test_merge_dataset_align(self):
        dataset = MetafileDataset(
            self.dummy_path, metafile="metadata.csv", header=False,
            idcol=0, textcol=2)
        self.assertEqual(10, len(dataset))
        alignfile = os.path.join(self.data_path, f"dummy_{self.language}-align-train.txt")
        align_ds = TextDataset(alignfile, idcol=-1, textcol=1)
        self.assertEqual(10, len(align_ds))
        ds = MergeDataset(dataset, align_ds=align_ds)
        self.assertEqual(10, len(ds))
        for clipid, audiopath, phonetext in ds:
            self.assertTrue(clipid.startswith("dummy001-"))
            waveform, rate = torchaudio.load(audiopath)
            self.assertIsInstance(waveform, torch.Tensor)
            self.assertEqual(16000, rate)
            self.assertIsInstance(phonetext, Text)

    def test_merge_dataset_phone(self):
        dataset = MetafileDataset(
            self.dummy_path, metafile="metadata.csv", header=False,
            idcol=0, textcol=2)
        self.assertEqual(10, len(dataset))
        phone_file = os.path.join(self.data_path, f"dummy_{self.language}-phone-train.txt")
        phone_ds = TextDataset(phone_file)
        self.assertEqual(10, len(phone_ds))
        ds = MergeDataset(dataset, phone_ds=phone_ds)
        self.assertEqual(10, len(ds))
        for clipid, audiopath, phonetext in ds:
            self.assertTrue(clipid.startswith("dummy001-"))
            waveform, rate = torchaudio.load(audiopath)
            self.assertIsInstance(waveform, torch.Tensor)
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
        from voice100.data_modules import get_dataset
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


def test_data_module(language="ja"):

    with TemporaryDirectory() as data_dir:
        dummy_path = os.path.join(data_dir, f"dummy-speech-{language}")
        make_dummy_metadataset(dummy_path, language=language)
        make_dummy_aligntext_dataset(data_dir, language=language, is_phone=False)
        make_dummy_aligntext_dataset(data_dir, language=language, is_phone=True)
        make_dummy_phonetext_dataset(data_dir, language=language)

        from voice100.audio import BatchSpectrogramAugumentation
        ARGS = "--dataset dummy_ja --language ja --use_phone --vocoder mel"
        parser = argparse.ArgumentParser()
        AudioTextDataModule.add_argparse_args(parser)
        args = parser.parse_args(ARGS.split())
        data: AudioTextDataModule = AudioTextDataModule.from_argparse_args(args, data_dir=data_dir)
        data.setup("predict")
        spec_augment = BatchSpectrogramAugumentation()

        print(f"audio_size={data.audio_size}")
        print(f"vocab_size={data.vocab_size}")
        use_phone = '--use_phone' in ARGS
        expected_vocab_size = 44 if use_phone else 29
        expected_audio_size = 64
        assert data.vocab_size == expected_vocab_size
        assert data.audio_size == expected_audio_size

        counter = [0] * data.vocab_size
        short_count = 0
        for batch in data.predict_dataloader():
            with torch.no_grad():
                (audio, audio_len), (text, text_len) = batch
                c = torch.sum((torch.divide(audio_len + 1, 2, rounding_mode='trunc') < text_len).to(torch.int)).item()
                if c:
                    short_count += c
                print(short_count)
                audio, audio_len = spec_augment(audio, audio_len)
                packed_audio = pack_padded_sequence(audio, audio_len, batch_first=True, enforce_sorted=False)
                _ = pad_packed_sequence(packed_audio, batch_first=False)
                assert not (torch.any(audio.isnan()))
                assert not (torch.any(audio.isinf()))
                assert (torch.min(audio_len) > 0)
                assert (torch.max(audio_len) == audio.shape[1]), f'{audio_len} {audio.shape}'
                for i in range(text.shape[0]):
                    assert text_len[i] > 0
                    for j in range(text_len[i]):
                        counter[text[i, j]] += 1
                assert counter[0] == 0, text


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
