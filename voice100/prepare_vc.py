# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import os
import librosa
import numpy as np
import pyworld
import torch
from torch import nn
from tqdm import tqdm

class AudioProcessor(nn.Module):

    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.n_mels = 64
        self.log_offset = 1e-6
        from torchaudio.transforms import MelSpectrogram
        self._transform = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels)

    def forward(self, waveform):
        melspec = self._transform(waveform)
        melspec = torch.transpose(melspec[0, :, :], 0, 1)
        melspec = torch.log(melspec + self.log_offset)
        return melspec

def prepare(use_gpu=False, source_sample_rate=16000, target_sample_rate=22050, use_w2v2=False):

    if target_sample_rate == 16000:
        n_fft = 512
    elif target_sample_rate == 22050:
        n_fft = 1024
    eps = 1e-15

    device = torch.device('cuda' if use_gpu else 'cpu')

    if use_w2v2:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        # load pretrained model
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        wav2vec.eval()
        wav2vec.to(device)
    else:
        from .models import AudioToCharCTC
        processor = AudioProcessor()
        audio2char = AudioToCharCTC.load_from_checkpoint('model/stt_ja_conv_base_ctc-20210608.ckpt')
        audio2char.eval()
        audio2char.to(device)

    os.makedirs('data/vc', exist_ok=True)
    stat = []

    with open('./data/kokoro-speech-v1_1-small/metadata.csv') as f:
        for i, line in enumerate(tqdm(f, total=8812)):
            parts = line.rstrip().split('|')
            wavid, _, _ = parts
            wavfile = f'./data/kokoro-speech-v1_1-small/wavs/{wavid}.flac'
            audio_input, sample_rate = librosa.load(wavfile, sr=source_sample_rate)
            if use_w2v2:
                input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
                input_values = input_values.to(device)
                with torch.no_grad():
                    wavvec = wav2vec(input_values).last_hidden_state
                wavvec = wavvec.cpu().numpy()[0, :, :]
                # wavvec: wavvec_len, wavvec_dim
            else:
                with torch.no_grad():
                    melspec = processor(torch.from_numpy(audio_input).float()[None, :])
                    melspec = melspec[None, :, :].to(device)
                    melspec_len = torch.tensor([melspec.shape[1]]).to(device)
                    wavvec, wavvec_len = audio2char.encode(melspec, melspec_len)
                    wavvec = wavvec.cpu().numpy()[0, :, :]

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

            outfile = f'data/vc/{i}.npz'
            np.savez(outfile, 
                wavvec=wavvec, f0=f0, spc=spc, codeap=codeap)

    stat = np.mean(np.stack(stat), axis=0)
    print(stat)

prepare(use_gpu=torch.cuda.is_available())