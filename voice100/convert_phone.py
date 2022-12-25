# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from typing import Text
from argparse import ArgumentParser
from .datasets import get_base_dataset
from .text import CMUPhonemizer
from tqdm import tqdm
from glob import glob
import os


def convert_phone_ljspeech(dataset, split: Text, output_file: Text) -> None:
    if split != "train":
        raise ValueError("Unknown split")
    phonemizer = CMUPhonemizer()
    with open("./data/LJSpeech-1.1/metadata.csv", "rt") as f:
        with open(output_file, "wt") as outf:
            for line in tqdm(f):
                clipid, text, _ = line.rstrip("\r\n").split("|")
                phone_text = phonemizer(text)
                outf.write("%s|%s\n" % (clipid, phone_text))


def convert_phone_librispeech(dataset, split: Text, output_file: Text) -> None:
    if split == "train":
        root = './data/LibriSpeech/train-clean-100'
    elif split == "val":
        root = './data/LibriSpeech/dev-clean'
    else:
        raise ValueError("Unknown split")
    phonemizer = CMUPhonemizer()
    files = glob(os.path.join(root, '**', '*.txt'), recursive=True)
    with open(output_file, "wt") as outf:
        for file in tqdm(sorted(files)):
            with open(file, "rt") as f:
                for line in f:
                    clipid, _, text = line.rstrip('\r\n').partition(' ')
                    phone_text = phonemizer(text)
                    outf.write("%s|%s\n" % (clipid, phone_text))


def convert_phone_kokoro(dataset: Text, split: Text, language: Text, output_file: Text) -> None:
    if split != "train":
        raise ValueError("Unknown split")
    assert language == 'ja'
    dataset = get_base_dataset(dataset, split)
    with open(output_file, "wt") as outf:
        for clipid, _, phone_text in tqdm(dataset):
            outf.write("%s|%s\n" % (clipid, phone_text))


def convert_phone(dataset: Text, split: Text, language: Text, output_file: Text) -> None:
    if split != "train":
        raise ValueError("Unknown split")
    assert language == 'ja'
    from .japanese import JapanesePhonemizer
    phonemizer = JapanesePhonemizer(use_phone=True)
    dataset = get_base_dataset(dataset, split)
    with open(output_file, "wt") as outf:
        for clipid, _, text in tqdm(dataset):
            phone_text = phonemizer(text)
            outf.write("%s|%s\n" % (clipid, phone_text))


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ljspeech")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    for dataset in args.dataset.split(','):
        for split in args.split.split(','):
            output_file = f"./data/{dataset}-phone-{split}.txt"
            if dataset == "ljspeech":
                convert_phone_ljspeech(dataset, split, output_file)
            elif dataset == "librispeech":
                convert_phone_librispeech(dataset, split, output_file)
            elif dataset.startswith("kokoro_"):
                convert_phone_kokoro(dataset, split=split, language='ja', output_file=output_file)
            elif dataset == 'cv_ja':
                convert_phone(dataset, split=split, language='ja', output_file=output_file)
            else:
                raise ValueError("Unknown dataset")


if __name__ == "__main__":
    cli_main()
