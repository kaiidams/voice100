# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from g2p_en import G2p
from tqdm import tqdm
from glob import glob
import os


def convert_phone_ljspeech() -> None:
    g2p = G2p()
    with open("./data/LJSpeech-1.1/metadata.csv", "rt") as f:
        with open("./data/phone-ljspeech.txt", "wt") as outf:
            for line in tqdm(f):
                clipid, text, _ = line.rstrip("\r\n").split("|")
                phone = g2p(text)
                phone_text = '/'.join(phone)
                outf.write("%s|%s\n" % (clipid, phone_text))


def convert_phone_librispeech() -> None:
    g2p = G2p()
    root = './data/LibriSpeech/dev-clean-100'
    files = glob(os.path.join(root, '**', '*.txt'), recursive=True)
    with open("./data/librispeech-phone-valid.txt", "wt") as outf:
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
    args = parser.parse_args()
    if args.dataset == "ljspeech":
        convert_phone_ljspeech()
    else:
        convert_phone_librispeech()


if __name__ == "__main__":
    cli_main()
