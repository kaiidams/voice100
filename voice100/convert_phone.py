from g2p_en import G2p
from tqdm import tqdm


def cli_main():
    g2p = G2p()
    with open("data/LJSpeech-1.1/metadata.csv", "rt") as f:
        with open("data/phone-ljspeech.txt", "wt") as outf:
            for line in tqdm(f):
                id_, text, _ = line.rstrip("\r\n").split("|")
                phone = g2p(text)
                phone_text = '/'.join(phone)
                outf.write("%s|%s\n" % (id_, phone_text))


if __name__ == "__main__":
    cli_main()
