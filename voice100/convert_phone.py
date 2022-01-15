# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from typing import Text
from argparse import ArgumentParser
from g2p_en import G2p
from tqdm import tqdm
from glob import glob
import os


def convert_phone_ljspeech(g2p: G2p, split: Text, output_file: Text) -> None:
    if split != "train":
        raise ValueError("Unknown split")
    with open("./data/LJSpeech-1.1/metadata.csv", "rt") as f:
        with open(output_file, "wt") as outf:
            for line in tqdm(f):
                clipid, text, _ = line.rstrip("\r\n").split("|")
                phone = g2p(text)
                phone_text = '/'.join(phone)
                outf.write("%s|%s\n" % (clipid, phone_text))


def convert_phone_librispeech(g2p: G2p, split: Text, output_file: Text) -> None:
    if split == "train":
        root = './data/LibriSpeech/train-clean-100'
    elif split == "val":
        root = './data/LibriSpeech/dev-clean'
    else:
        raise ValueError("Unknown split")
    files = glob(os.path.join(root, '**', '*.txt'), recursive=True)
    with open(output_file, "wt") as outf:
        for file in tqdm(sorted(files)):
            with open(file, "rt") as f:
                for line in f:
                    clipid, _, text = line.rstrip('\r\n').partition(' ')
                    phone = g2p(text)
                    phone_text = '/'.join(phone)
                    outf.write("%s|%s\n" % (clipid, phone_text))


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ljspeech")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    g2p = G2p()
    for dataset in args.dataset.split(','):
        for split in args.split.split(','):
            output_file = f"./data/{dataset}-phone-{split}.txt"
            if dataset == "ljspeech":
                convert_phone_ljspeech(g2p, split, output_file)
            elif dataset == "librispeech":
                convert_phone_librispeech(g2p, split, output_file)
            else:
                raise ValueError("Unknown dataset")


if __name__ == "__main__":
    cli_main()
