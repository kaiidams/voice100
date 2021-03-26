# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import os
import numpy as np
import math
from tqdm import tqdm
import argparse

from .vocoder import readwav, estimatef0, encode_audio
from .encoder import encode_text

CORPUSDATA_PATH = 'data/balance_sentences.txt'
CORPUSDATA_CSS10JA_PATH = 'data/japanese-single-speaker-speech-dataset/transcript.txt'
CORPUSDATA_KOKORO_PATH = 'data/kokoro-speech-v1_0/metadata.csv'

WAVDATA_PATH = {
    'css10ja': 'data/japanese-single-speaker-speech-dataset/%s',
    'tsuchiya_normal': 'data/tsuchiya_normal/tsuchiya_normal_%s.wav',
    'hiroshiba_normal': 'data/hiroshiba_normal/hiroshiba_normal_%s.wav',
    'tsukuyomi_normal': 'data/つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠）'
        '/01 WAV（収録時の音量のまま）/VOICEACTRESS100_%s.wav',
}

# F0 mean +/- 2.5 std
F0_RANGE = {
    'css10ja': (57.46701428196299, 196.7528135117272),
    'kokoro': (57.46701428196299, 196.7528135117272),
    'tsukuyomi_normal': (138.7640311667663, 521.2003965068923)
}

OUTPUT_PATH = 'data/%s_%s.npz'

def readcorpus(file):
    corpus = []
    with open(file) as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            id_, _, monophone, _ = parts
            monophone = monophone.replace('/', '').replace(',', '')
            corpus.append((id_, monophone))

    return corpus

def readcorpus_ljcorpus(file):
    corpus = []
    with open(file) as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            id_, _, monophone = parts
            corpus.append((id_, monophone))
    return corpus

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

def analyze_files(name, files, eps=1e-20):
    f0_list = []
    power_list = []
    for file in tqdm(files):
        x = readwav(file)

        # Estimate F0
        f0 = estimatef0(x)
        f0 = f0[f0 > 0].copy()
        assert f0.ndim == 1
        f0_list.append(f0)

        # Calculate loudness
        window_size = 1024
        power = x[:(x.shape[0] // window_size) * window_size].reshape((-1, window_size))
        power = np.log(np.mean(power ** 2, axis=1) + eps)
        assert power.ndim == 1
        power_list.append(power)

    f0 = np.concatenate(f0_list, axis=0)
    f0_mean = np.mean(f0)
    f0_std = np.std(f0)
    f0_hist, f0_bin_edges = np.histogram(f0, bins=np.linspace(0, 1000.0, 101), density=True)

    power = np.concatenate(power_list, axis=0)
    power_mean = np.mean(power)
    power_std = np.std(power)
    power_hist, power_bin_edges = np.histogram(power, bins=np.linspace(-30, 0.0, 100), density=True)

    np.savez(os.path.join('data', '%s_stat.npz' % name),
        f0=f0,
        f0_mean=f0_mean, f0_std=f0_std,
        f0_hist=f0_hist, f0_bin_edges=f0_bin_edges,
        power_mean=power_mean, power_std=power_std,
        power_hist=power_hist, power_bin_edges=power_bin_edges)

def analyze_css10ja(name):
    corpus = readcorpus_css10ja(CORPUSDATA_CSS10JA_PATH)
    files = []
    for id_, monophone in corpus[17::6841 // 100]: # Use 100 samples from corpus
        file = os.path.join('data', 'japanese-single-speaker-speech-dataset', id_)
        files.append(file)
    analyze_files(name, files)

def analyze_jvs(name):
    corpus = readcorpus(CORPUSDATA_PATH)
    files = []
    for id_, monophone in corpus:
        assert len(id_) == 3
        file = WAVDATA_PATH[name] % id_
        files.append(file)
    analyze_files(name, files)

def make_empty_data():
    return {
        'id': [],
        'text_index': [],
        'text_data': [],
        'audio_index': [],
        'audio_data': []
    }

def append_data(data, id_, text_index, text, audio_index, audio):
    data['id'].append(id_)
    data['text_index'].append(text_index)
    data['text_data'].append(text)
    data['audio_index'].append(audio_index)
    data['audio_data'].append(audio)

def finish_data(data, file):
    data['id'] = np.array(data['id'])
    data['text_index'] = np.array(data['text_index'], dtype=np.int32)
    data['text_data'] = np.concatenate(data['text_data'], axis=0)
    data['audio_index'] = np.array(data['audio_index'], dtype=np.int32)
    data['audio_data'] = np.concatenate(data['audio_data'], axis=0)
    np.savez(file, **data)

def preprocess_css10ja(name):

    if name.endswith('_highpitch'):
        tsukuyomi_average_logf0 = 5.783612067835965
        css10ja_average_logf0 = 4.830453997458316
        pitchshift = math.exp(tsukuyomi_average_logf0 - css10ja_average_logf0)
    else:
        pitchshift = None
    f0_floor, f0_ceil = F0_RANGE[name]

    text_index = [0, 0]
    audio_index = [0, 0]
    data = [
        make_empty_data(),
        make_empty_data(),
    ]

    corpus = readcorpus_css10ja(CORPUSDATA_CSS10JA_PATH)
    split_index = 13
    split_total = 57 # We take 1 in 57 samples aside for validation
    for id_, monophone in tqdm(corpus):

        if not monophone:
            print('Skipping: <empty>')
            continue
        try:
            text = encode_text(monophone)
        except:
            print(f'Skipping: {monophone}')
            continue
    
        file = os.path.join('data', 'japanese-single-speaker-speech-dataset', id_)
        assert '..' not in file # Just make sure it is under the current directory.
        cache_file = os.path.join('data', 'cache', name, id_.replace('.wav', '.npz'))
        if os.path.exists(cache_file):
            audio = np.load(cache_file, allow_pickle=True)['arr_0']
            assert audio.shape[0] > 0
        else:
            x = readwav(file)
            audio = encode_audio(x, f0_floor, f0_ceil, pitchshift=pitchshift)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.savez(cache_file, audio)

        split = 0 if split_index else 1
        text_index[split] += text.shape[0]
        audio_index[split] += audio.shape[0]
        append_data(data[split], id_, text_index[split], text, audio_index[split], audio)

        split_index += 1
        if split_index == split_total: split_index = 0

    finish_data(data[0], OUTPUT_PATH % (name, "train"))
    finish_data(data[1], OUTPUT_PATH % (name, "val"))

def preprocess_ljcorpus(name):

    pitchshift = None
    f0_floor, f0_ceil = F0_RANGE[name]

    text_index = [0, 0]
    audio_index = [0, 0]
    data = [
        make_empty_data(),
        make_empty_data(),
    ]

    corpus = readcorpus_ljcorpus(CORPUSDATA_KOKORO_PATH)
    wavs_dir = os.path.join(os.path.dirname(CORPUSDATA_KOKORO_PATH), 'wavs')
    split_index = 13
    split_total = 57 # We take 1 in 57 samples aside for validation
    for id_, monophone in tqdm(corpus):

        if not monophone:
            print('Skipping: <empty>')
            continue
        try:
            text = encode_text(monophone)
        except:
            print(f'Skipping: {monophone}')
            continue
    
        file = os.path.join(wavs_dir, id_ + '.wav')
        assert '..' not in file # Just make sure it is under the current directory.
        cache_file = os.path.join('data', 'cache', name, id_.replace('.wav', '.npz'))
        if os.path.exists(cache_file):
            audio = np.load(cache_file, allow_pickle=True)['arr_0']
            assert audio.shape[0] > 0
        else:
            x = readwav(file)
            audio = encode_audio(x, f0_floor, f0_ceil, pitchshift=pitchshift)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.savez(cache_file, audio)

        split = 0 if split_index else 1
        text_index[split] += text.shape[0]
        audio_index[split] += audio.shape[0]
        append_data(data[split], id_, text_index[split], text, audio_index[split], audio)

        split_index += 1
        if split_index == split_total: split_index = 0

    finish_data(data[0], OUTPUT_PATH % (name, "train"))
    finish_data(data[1], OUTPUT_PATH % (name, "val"))

def preprocess_jvs(name):
    corpus = readcorpus(CORPUSDATA_PATH)
    f0_floor, f0_ceil = F0_RANGE[name]
    data = make_empty_data()
    text_index = 0
    audio_index = 0
    for id_, monophone in tqdm(corpus):
        assert len(id_) == 3
        file = WAVDATA_PATH[name] % id_
        x = readwav(file)
        text = encode_text(monophone)
        audio = encode_audio(x, f0_floor, f0_ceil)
        text_index += text.shape[0]
        audio_index += audio.shape[0]
        append_data(data, id_, text_index, text, audio_index, audio)

    finish_data(data, OUTPUT_PATH % (name, "train"))

def normalize_css10ja(name):

    if False:
        from .data_pipeline import normparams
        for split in 'train', 'val':
            with np.load(OUTPUT_PATH % (name, split) + '.bak') as f:
                data = {k:v for k, v in f.items()}
            print(data.keys())
            print(data['audio_data'].shape)
            data['audio_data'] = data['audio_data'] * normparams[:, 0] + normparams[:, 1]
            np.savez(OUTPUT_PATH % (name, split), **data)

    with np.load(OUTPUT_PATH % (name, "train")) as f:
        audio = f['audio_data']
    mean = np.mean(audio, axis=0)
    std = np.std(audio, axis=0)
    x = np.stack([mean, std], axis=1)
    for i in range(x.shape[0]):
        print('    [%s, %s],' % (str(x[i, 0]), str(x[i, 1])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true', help='Analyze F0 of sampled data.')
    parser.add_argument('--normalize', action='store_true', help='Compute normalization parameters.')
    parser.add_argument('--dataset', required=True, help='Dataset to process, css10ja, tsukuyomi_normal')
    args = parser.parse_args()

    if args.analyze:
        if args.dataset == 'css10ja':
            analyze_css10ja(args.dataset)
        else:
            analyze_jvs(args.dataset)
    if args.normalize:
        if args.dataset == 'css10ja':
            normalize_css10ja(args.dataset)
        else:
            assert False
    else:
        if args.dataset == 'css10ja' or args.dataset == 'css10ja_highpitch':
            preprocess_css10ja(args.dataset)
        if args.dataset == 'kokoro':
            preprocess_ljcorpus(args.dataset)
        else:
            preprocess_jvs(args.dataset)
