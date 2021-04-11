# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torchaudio

from .encoder import encode_text
from .encoder import encode_text2
from .data import open_index_data_for_write

import logging
logging.basicConfig(level=logging.INFO)

CORPUSDATA_CSS10JA_PATH = 'data/japanese-single-speaker-speech-dataset'
CORPUSDATA_COMMONVOICE_PATH = 'data/cv-corpus-6.1-2020-12-11/ja'

TEXT_PATH = 'data/%s-text'
AUDIO_PATH = 'data/%s-audio'

def readcorpus_css10ja(file):
    from ._css10ja2voca import css10ja2voca
    corpus = []
    with open(file) as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            id_, _, yomi, _ = parts
            monophone = css10ja2voca(yomi)
            corpus.append((id_, monophone))
    return corpus

def get_silent_ranges(voiced):
    silent_to_voiced = np.where((~voiced[:-1]) & voiced[1:])[0] + 1 # The position where the voice starts
    voiced_to_silent = np.where((voiced[:-1]) & ~voiced[1:])[0] + 1 # The position where the silence starts
    if not voiced[0]:
        # Eliminate the preceding silence
        silent_to_voiced = silent_to_voiced[1:]
    if not voiced[-1]:
        # Eliminate the succeeding silence
        voiced_to_silent = voiced_to_silent[:-1]
    return np.stack([voiced_to_silent, silent_to_voiced]).T

def get_split_points(x, minimum_silent_frames, minimum_split_distance, maximum_split_distance, window_size, eps=1e-12):

    num_frames = len(x) // window_size
    mX = np.mean(x[:window_size * num_frames].reshape((-1, window_size)) ** 2, axis=1)
    mX = 10 * np.log(mX + eps)

    silent_threshold = (np.max(mX) + np.min(mX)) / 2
    
    while True:
        # Fill short silent
        voiced = mX > silent_threshold
        silent_ranges = get_silent_ranges(voiced)
        for s, e in silent_ranges:
            if e - s < minimum_silent_frames:
                voiced[s:e] = True

        silent_ranges = get_silent_ranges(voiced)

        # Split in the center of silence.
        silent_points = (silent_ranges[:, 0] + silent_ranges[:, 1]) // 2

        split_distance = (
            np.append(silent_points, num_frames) - np.insert(silent_points, 0, 0))
        if np.max(split_distance) < maximum_split_distance:
            break

        minimum_silent_frames *= 0.5
        if minimum_silent_frames < 0.05:
            raise ValueError("Audio cannot be split into")


    # Merge short splits
    while len(silent_points):
        split_distance = (
            np.append(silent_points, num_frames) - np.insert(silent_points, 0, 0))
        i = np.argmin(split_distance)
        if split_distance[i] > minimum_split_distance:
            break
        if i == 0:
            silent_points = np.delete(silent_points, i)
        elif i == len(silent_points):
            silent_points = np.delete(silent_points, len(silent_points) - 1)
        else:
            if split_distance[i - 1] < split_distance[i + 1]:
                silent_points = np.delete(silent_points, i - 1)
            else:
                silent_points = np.delete(silent_points, i)

    return silent_points

def split_audio(
    audio_file, segment_file, audio_data_file,
    expected_sample_rate=22050, n_mfcc=40, n_mels=40, n_fft=512
    ):
    window_size = n_fft // 2 # 46ms
    minimum_silent_duration = 0.25 # 500ms
    padding_duration = 0.05 # 50ms
    minimum_silent_frames = minimum_silent_duration * expected_sample_rate / window_size
    minimum_split_distance = 3.0 * expected_sample_rate / window_size
    maximum_split_distance = 15.0 * expected_sample_rate / window_size
    padding_frames = min(1, int(padding_duration * expected_sample_rate // window_size))

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=expected_sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'hop_length': n_fft // 2})

    with open(segment_file, 'wt') as segf:
        with open_index_data_for_write(audio_data_file) as data:
            y, sr = torchaudio.load(audio_file)
            assert len(y.shape) == 2 and y.shape[0] == 1
            assert sr == expected_sample_rate
            y = torch.mean(y, axis=0) # to mono
            split_points = get_split_points(y.numpy(), minimum_silent_frames, 
                minimum_split_distance, maximum_split_distance, window_size) * window_size
            for i in range(len(split_points) + 1):
                start = split_points[i - 1] if i > 0 else 0
                end = split_points[i] if i < len(split_points) else len(y)
                mfcc = mfcc_transform(y[start:end]).T
                data.write(mfcc.numpy().astype(np.float32))
                segf.write(f'{end}\n')

def preprocess_css10ja(args, expected_sample_rate=22050, n_mfcc=40, n_mels=40, n_fft=512):

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=expected_sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'hop_length': n_fft // 2})

    corpus = readcorpus_css10ja(os.path.join(CORPUSDATA_CSS10JA_PATH, 'transcript.txt'))
    with open_index_data_for_write(TEXT_PATH % (args.dataset,)) as textf:
        with open_index_data_for_write(AUDIO_PATH % (args.dataset,)) as audiof:
            for id_, monophone in tqdm(corpus):

                if not monophone:
                    print('Skipping: <empty>')
                    continue
                try:
                    encoded = encode_text(monophone)
                    assert encoded.dtype == np.int8
                except:
                    print(f'Skipping: {monophone}')
                    continue
                encoded = encode_text2(monophone)
            
                file = os.path.join(CORPUSDATA_CSS10JA_PATH, id_)
                assert '..' not in file # Just make sure it is under the current directory.
                y, sr = torchaudio.load(file)
                assert len(y.shape) == 2 and y.shape[0] == 1
                assert sr == expected_sample_rate
                y = torch.mean(y, axis=0) # to mono
                mfcc = mfcc_transform(y).T
                textf.write(encoded)
                audiof.write(mfcc.numpy().astype(np.float32))

def readcorpus_commonvoice(file):
    from ._text2voca import text2voca
    res = []
    with open(file, 'rt') as f:
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            _, path, sentence, _, _, _, _, _, _, _ = parts
            voca = ' '.join(x for _, x in text2voca(sentence))
            res.append((path, voca))
    res = res[1:]
    return res

def preprocess_commonvoice(args, expected_sample_rate=22050, n_mfcc=40, n_mels=40, n_fft=512):

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=expected_sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'hop_length': n_fft // 2})

    corpus = readcorpus_commonvoice(os.path.join(CORPUSDATA_COMMONVOICE_PATH, 'validated.tsv'))
    with open_index_data_for_write(TEXT_PATH % (args.dataset,)) as textf:
        with open_index_data_for_write(AUDIO_PATH % (args.dataset,)) as audiof:
            for id_, monophone in tqdm(corpus):

                if not monophone:
                    print('Skipping: <empty>')
                    continue
                try:
                    encoded = encode_text(monophone)
                    assert encoded.dtype == np.int8
                except:
                    print(f'Skipping: {monophone}')
                    continue
                encoded = encode_text2(monophone)
            
                file = os.path.join(CORPUSDATA_COMMONVOICE_PATH, 'clips', id_)
                assert '..' not in file # Just make sure it is under the current directory.
                effects = [["rate", "22050"]]
                y, sr = torchaudio.sox_effects.apply_effects_file(file, effects=effects)
                assert len(y.shape) == 2 and y.shape[0] == 1
                assert sr == expected_sample_rate
                y = torch.mean(y, axis=0) # to mono
                y = y / torch.max(torch.abs(y))
                mfcc = mfcc_transform(y).T
                textf.write(encoded)
                audiof.write(mfcc.numpy().astype(np.float32))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='css10ja', help='Dataset name')
    args = parser.parse_args()

    if args.dataset == 'cv_ja':
        preprocess_commonvoice(args)
    elif args.dataset != 'css10ja':
        split_audio(args)
    else:
        preprocess_css10ja(args)
