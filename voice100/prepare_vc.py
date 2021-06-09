# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import os
import librosa
import numpy as np
import pyworld
import torch
from tqdm import tqdm

def prepare(use_gpu=False, use_sptk=True, source_sample_rate=16000, target_sample_rate=22050):

    if target_sample_rate == 16000:
        n_fft = 512
    elif target_sample_rate == 22050:
        n_fft = 1024
    eps = 1e-15

    device = torch.device('cuda' if use_gpu else 'cpu')

    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    # load pretrained model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec.eval()
    wav2vec.to(device)

    os.makedirs('data', exist_ok=True)
    stat = []

    with open('./kokoro-speech-v1_1-small/metadata.csv') as f:
        for i, line in enumerate(tqdm(f, total=8812)):
            parts = line.rstrip().split('|')
            wavid, _, _ = parts
            wavfile = f'./kokoro-speech-v1_1-small/wavs/{wavid}.wav'
            audio_input, sample_rate = librosa.load(wavfile, sr=source_sample_rate)
            input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
            input_values = input_values.to(device)
            with torch.no_grad():
                wavvec = wav2vec(input_values).last_hidden_state
            wavvec = wavvec.cpu().numpy()[0, :, :]
            # wavvec: wavvec_len, wavvec_dim

            if sample_rate != target_sample_rate:
                audio_input, sample_rate = librosa.load(wavfile, sr=target_sample_rate)
            waveform = audio_input.astype(np.double)
            waveform = waveform * 0.8 / np.max(waveform)
            f0, time_axis = pyworld.dio(
                waveform, sample_rate,
                f0_floor=40, f0_ceil=400,
                frame_period=5.0)
            spc = pyworld.cheaptrick(waveform, f0, time_axis, sample_rate, fft_size=n_fft)
            ap = pyworld.d4c(waveform, f0, time_axis, sample_rate, fft_size=n_fft)
            codeap = pyworld.code_aperiodicity(ap, sample_rate)

            f0 = f0.astype(np.float32)
            spc = np.log(spc + eps).astype(np.float32)
            codeap = codeap.astype(np.float32)

            stat.append(np.array([
                np.mean(f0),
                np.mean(spc),
                np.mean(codeap),
                np.std(f0),
                np.std(spc),
                np.std(codeap)
            ]))

            outfile = f'data/{i}.npz'
            np.savez(outfile, 
                wavvec=wavvec, f0=f0, spc=spc, codeap=codeap)

    stat = np.mean(np.stack(stat), axis=0)
    print(stat)

prepare(use_gpu=True)