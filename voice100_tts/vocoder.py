import torch
import torchaudio
import numpy as np
import pyworld
import pysptk
import librosa
import math

from .wsola import WSOLA

SAMPLE_RATE = 16000

class WORLDVocoder:
    def __init__(self, sample_rate=16000, frame_period=10.0, n_fft=512, mcep_dim=24, mcep_alpha=0.410):
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.n_fft = n_fft
        self.mcep_dim = mcep_dim
        self.mcep_alpha = mcep_alpha

    def encode(self, waveform, f0_floor=40, f0_ceil=400):
        waveform = waveform.astype(np.double)
        f0, time_axis = pyworld.harvest(waveform, self.sample_rate, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=self.frame_period)
        spc = pyworld.cheaptrick(waveform, f0, time_axis, self.sample_rate, fft_size=self.n_fft)
        ap = pyworld.d4c(waveform, f0, time_axis, self.sample_rate, fft_size=self.n_fft)

        mcep = pysptk.sp2mc(spc, self.mcep_dim, self.mcep_alpha)
        codeap = pyworld.code_aperiodicity(ap, self.sample_rate)

        return np.concatenate([
            f0[:, None], mcep, codeap
        ], axis=1).astype(np.float32)

    def decode(self, encoded):
        f0 = encoded[:, 0].copy().astype(np.double)
        mcep = encoded[:, 1:2 + self.mcep_dim].copy().astype(np.double)
        codeap = encoded[:, 2 + self.mcep_dim:].copy().astype(np.double)

        ap = pyworld.decode_aperiodicity(codeap, self.sample_rate, self.n_fft)
        spc = pysptk.mc2sp(mcep, self.mcep_alpha, self.n_fft)
        waveform = pyworld.synthesize(f0, spc, ap, self.sample_rate, frame_period=self.frame_period)
        return waveform

class MelSpectrogramVocoder:
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=64, log_offset=1e-6):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.log_offset = log_offset

    def encode(self, waveform):
        melspec = librosa.feature.melspectrogram(
            waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_mels=self.n_mels,
            norm='slaney',
            htk=True,
        )
        return np.log(melspec.T + self.log_offset)

    def decode(self, x):
        raise NotImplementedError("Decoding from Mel-spectrogram is not supported")

class AudioAugmentation:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.n_fft = 1024

    def augment(self, waveform, f0_floor=40, f0_ceil=400):
        pitch_shift = np.random.uniform(low=0.95, high=1.2)
        f0_shift = np.random.uniform(low=0.95, high=1.2)
        sn_rate = np.random.uniform(low=5.0, high=100.0)
        tone_freq = np.random.uniform(low=10.0, high=2000.0)
        tone_rate = np.random.uniform(low=5.0, high=100.0)

        # Shift pitch
        speech_rate = 1 / pitch_shift
        wsola = WSOLA(self.sample_rate, speech_rate)
        waveform = wsola.duration_modification(waveform)
        waveform = librosa.resample(waveform, pitch_shift * self.sample_rate, self.sample_rate)

        # Shift F0
        f0, time_axis = pyworld.harvest(
            waveform, self.sample_rate,
            f0_floor=pitch_shift * f0_floor, f0_ceil=pitch_shift * f0_ceil)
        spc = pyworld.cheaptrick(waveform, f0, time_axis, self.sample_rate, fft_size=self.n_fft)
        ap = pyworld.d4c(waveform, f0, time_axis, self.sample_rate, fft_size=self.n_fft)
        f0 = f0 * f0_shift / pitch_shift
        waveform = pyworld.synthesize(f0, spc, ap, self.sample_rate)

        # Noise
        noise = np.random.random(waveform.shape)
        waveform = (waveform * sn_rate + noise) / (1 + sn_rate)

        # Tone
        tone = np.sin(2 * math.pi * tone_freq * np.arange(len(waveform)) / self.sample_rate)
        waveform = (waveform * tone_rate + tone) / (1 + tone_rate)

        return waveform
