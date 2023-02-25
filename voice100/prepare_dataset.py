# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from typing import Text
from argparse import ArgumentParser
from .data_modules import get_base_dataset
from tqdm import tqdm
import os


def get_phonemizer(language: Text, use_phone: bool):
    if language == "en":
        if use_phone:
            from .text import CMUPhonemizer
            return CMUPhonemizer()
        else:
            from .text import BasicPhonemizer
            return BasicPhonemizer()
    elif language == "ja":
        from .japanese import JapanesePhonemizer
        return JapanesePhonemizer(use_phone=use_phone)
    else:
        raise ValueError(f"Unknown language `{language}'")


def convert_phone_kokoro(data_dir: Text, dataset: Text, split: Text, language: Text, use_phone: bool, output_file: Text) -> None:
    if split != "train":
        raise ValueError("Unknown split")
    if not use_phone:
        raise ValueError("Dataset doesn't support `use_phone=True'")
    if language != "ja":
        raise ValueError(f"Dataset doesn't support `language={language}'")
    dataset = get_base_dataset(data_dir, dataset, split)
    with open(output_file, "wt") as outf:
        for clipid, _, phone_text in tqdm(dataset):
            outf.write("%s|%s\n" % (clipid, phone_text))


def convert_phone(data_dir: Text, dataset: Text, split: Text, language: Text, use_phone: bool, output_file: Text) -> None:
    phonemizer = get_phonemizer(language=language, use_phone=use_phone)
    dataset = get_base_dataset(data_dir, dataset, split)
    with open(output_file, "wt") as outf:
        for clipid, _, text in tqdm(dataset):
            phone_text = phonemizer(text)
            outf.write("%s|%s\n" % (clipid, phone_text))


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--use_phone", action='store_true')
    args = parser.parse_args()
    for dataset in args.dataset.split(','):
        for split in args.split.split(','):
            if args.use_phone:
                output_file = os.path.join(args.data_dir, f"{dataset}-phone-{split}.txt")
            else:
                output_file = os.path.join(args.data_dir, f"{dataset}-{split}.txt")
            if dataset.startswith("kokoro_"):
                convert_phone_kokoro(args.data_dir, dataset, split=split, language=args.language, use_phone=args.use_phone, output_file=output_file)
            else:
                convert_phone(args.data_dir, dataset, split=split, language=args.language, use_phone=args.use_phone, output_file=output_file)


if __name__ == "__main__":
    cli_main()
