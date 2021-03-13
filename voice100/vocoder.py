# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import soundfile as sf
import librosa
import pyworld
import pysptk
import numpy as np

SAMPLE_RATE = 16000
FFT_SIZE = 1024

def readwav(file, fs=SAMPLE_RATE):
    x, origfs = sf.read(file)
    if fs is not None:
        x = librosa.resample(x, origfs, fs)
    x = x / x.max()
    return x

def writewav(file, x, fs=SAMPLE_RATE):
    sf.write(file, x, fs, 'PCM_16')

def estimatef0(x, fs=SAMPLE_RATE, frame_period=20):
    f0, _ = pyworld.harvest(x, fs, f0_floor=40, f0_ceil=700, frame_period=frame_period)
    return f0

def analyze_world(x, fs, f0_floor, f0_ceil, frame_period):
    f0, time_axis = pyworld.harvest(x, fs, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period)
    spc = pyworld.cheaptrick(x, f0, time_axis, fs, fft_size=FFT_SIZE)
    ap = pyworld.d4c(x, f0, time_axis, fs, fft_size=FFT_SIZE)
    return f0, spc, ap

def analyze(x, fs, f0_floor, f0_ceil, frame_period=20.0, pitchshift=None):
    if pitchshift is not None:
        f0, spc, ap = analyze_world(x, fs * pitchshift, f0_floor, f0_ceil, frame_period / pitchshift)
    else:
        f0, spc, ap = analyze_world(x, fs, f0_floor, f0_ceil, frame_period)
    mcep = pysptk.sp2mc(spc, 24, 0.410)
    codeap = pyworld.code_aperiodicity(ap, fs)
    
    #return x, fs, f0, time_axis, spc, ap, mcep, codeap
    return f0, mcep, codeap

def synthesize(f0, mcep, codeap, fs, frame_period=20):
    ap = pyworld.decode_aperiodicity(codeap, fs, FFT_SIZE)
    spc = pysptk.mc2sp(mcep, 0.410, FFT_SIZE)
    y = pyworld.synthesize(f0, spc, ap, fs, frame_period=frame_period)
    return y

def encode_audio(x, f0_floor, f0_ceil, fs=SAMPLE_RATE, pitchshift=None):
    f0, mcep, codeap = analyze(x, fs, f0_floor, f0_ceil, pitchshift=pitchshift)
    encoded = np.hstack((f0.reshape((-1, 1)), mcep, codeap))
    return encoded.astype(np.float32)

def decode_audio(encoded, fs=SAMPLE_RATE):
    encoded = encoded.astype(np.float)
    f0 = encoded[:, 0].copy()
    mcep = encoded[:, 1:26].copy()
    codeap = encoded[:, 26:].copy()
    y = synthesize(f0, mcep, codeap, fs)
    return y
