from glob import glob
import torchaudio
import torch
from tqdm import tqdm
from .vocoder import WORLDVocoder


def main():
    vocoder = WORLDVocoder()
    for file in tqdm(glob('data/LJSpeech-1.1/wavs/*.flac')):
        waveform, sr = torchaudio.sox_effects.apply_effects_file(file, effects=[
            ['rate', '16000']
        ])
        f0, mc, codeap = vocoder.encode(waveform[0])
        print(codeap.shape)
        torch.save(dict(f0=f0, mc=mc, codeap=codeap), file.replace('.flac', '.pt'))


if __name__ == "__main__":
    main()
