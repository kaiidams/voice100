# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import os
import numpy as np
import math
from tqdm import tqdm
import argparse

from .vocoder import *
from .encoder import encode_text
from .data import open_index_data_for_write

CORPUSDATA_PATH = {
    'jvs': 'data/balance_sentences.txt',
    'css10ja': 'data/japanese-single-speaker-speech-dataset/transcript.txt',
    'kokoro_large': 'data/kokoro-speech-v1_1-large/metadata.csv',
    'kokoro_small': 'data/kokoro-speech-v1_1-small/metadata.csv',
    'kokoro_tiny': 'data/kokoro-speech-v1_1-tiny/metadata.csv',
    'cv_ja': 'data/cv-corpus-6.1-2020-12-11/ja/validated.tsv',
}

WAVDATA_PATH = {
    'css10ja': 'data/japanese-single-speaker-speech-dataset/%s',
    'tsuchiya_normal': 'data/tsuchiya_normal/tsuchiya_normal_%s.wav',
    'hiroshiba_normal': 'data/hiroshiba_normal/hiroshiba_normal_%s.wav',
    'tsukuyomi_normal': 'data/つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠）'
        '/01 WAV（収録時の音量のまま）/VOICEACTRESS100_%s.wav',
    'cv_ja': 'data/cv-corpus-6.1-2020-12-11/ja/clips/%s',
}

# F0 mean +/- 2.5 std
F0_RANGE = {
    'css10ja': (57.46701428196299, 196.7528135117272),
    'kokoro_large': (57.46701428196299, 196.7528135117272),
    'kokoro_small': (57.46701428196299, 196.7528135117272),
    'kokoro_tiny': (57.46701428196299, 196.7528135117272),
    'tsuchiya_normal': (108.7640311667663, 421.2003965068923), # ?
    'tsukuyomi_normal': (138.7640311667663, 521.2003965068923),
    'cv_ja': (60, 520),
}

OUTPUT_PATH = 'data/%s_%s.npz'

def readcorpus_jvs(file):
    from ._text2voca import text2voca
    corpus = []
    with open(file) as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            id_, _, _, monophone = parts
            monophone = monophone.replace('/', ' ').replace(',', ' ')
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
    corpus = readcorpus_jvs(CORPUSDATA_PATH)
    files = []
    for id_, monophone in corpus:
        assert len(id_) == 3
        file = WAVDATA_PATH[name] % id_
        files.append(file)
    analyze_files(name, files)

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
#append_data(data[split], id_, text_index[split], text, audio_index[split], audio)

        split_index += 1
        if split_index == split_total: split_index = 0

    finish_data(data[0], OUTPUT_PATH % (name, "train"))
    finish_data(data[1], OUTPUT_PATH % (name, "val"))

def preprocess_ctc_ljcorpus(args):

    from .encoder import PhoneEncoder

    dataset = args.dataset
    sr = args.sample_rate
    f0_floor, f0_ceil = F0_RANGE[dataset]

    corpus = readcorpus_ljcorpus(CORPUSDATA_PATH[dataset])
    wavs_dir = os.path.join(os.path.dirname(CORPUSDATA_PATH[dataset]), 'wavs')
    cache_dir = os.path.join(os.path.dirname(CORPUSDATA_PATH[dataset]), f'cache-{sr}')

    phone_file = os.path.join('data', f'{dataset}-ctc-phone-{sr}')
    melspec_file = os.path.join('data', f'{dataset}-ctc-melspec-{sr}')
    os.makedirs(cache_dir, exist_ok=True)

    encoder = PhoneEncoder()
    vocoder = MelSpectrogramVocoder

    with open_index_data_for_write(phone_file) as phone_f:
        with open_index_data_for_write(melspec_file) as audio_f:
            for id_, monophone in tqdm(corpus):
                phone = encoder.encode(monophone)
                assert '..' not in id_ # Just make sure the file name is under the directory.
                wav_file = os.path.join(wavs_dir, f'{id_}.wav')
                cache_file = os.path.join(cache_dir, f'{id_}.npz')
                if os.path.exists(cache_file):
                    audio = np.load(cache_file, allow_pickle=False)['arr_0']
                    assert audio.shape[0] > 0
                else:
                    x = readwav(wav_file)
                    audio = vocoder.encode(x)
                    np.savez(cache_file, audio=audio)

                phone_f.write(bytes(memoryview(phone)))
                audio_f.write(bytes(memoryview(audio)))

def preprocess_vc_ljcorpus(name):

    melspec_per_audio = 3
    sr = SAMPLE_RATE
    f0_floor, f0_ceil = F0_RANGE[name]

    corpus = readcorpus_ljcorpus(CORPUSDATA_PATH[name])
    wavs_dir = os.path.join(os.path.dirname(CORPUSDATA_PATH[name]), 'wavs')
    cache_dir = os.path.join(os.path.dirname(CORPUSDATA_PATH[name]), f'cache-vc-{sr}')

    melspec_file = os.path.join('data', f'{name}-vc-melspec-{sr}')
    audio_file = os.path.join('data', f'{name}-vc-audio-{sr}')
    os.makedirs(cache_dir, exist_ok=True)

    melspec_vocoder = MelSpectrogramVocoder()
    world_vocoder = WORLDVocoder()
    augmentation = AudioAugmentation()

    with open_index_data_for_write(melspec_file) as melspec_f:
        with open_index_data_for_write(audio_file) as audio_f:
            for id_, _ in tqdm(corpus):                
                assert '..' not in id_ # Just make sure the file name is under the directory.
                wav_file = os.path.join(wavs_dir, f'{id_}.wav')
                waveform, _ = librosa.load(wav_file, SAMPLE_RATE)
                waveform = 0.8 * waveform / np.max(waveform)

                cache_file = os.path.join(cache_dir, f'{id_}.npz')
                if os.path.exists(cache_file):
                    audio = np.load(cache_file, allow_pickle=False)['audio']
                    assert audio.shape[0] > 0
                else:
                    audio = world_vocoder.encode(waveform)
                    np.savez(cache_file, audio=audio)

                audio_len = audio.shape[0]
                assert audio.dtype == np.float32
                audio_f.write(bytes(memoryview(audio)))

                for i in range(melspec_per_audio):
                    cache_file = os.path.join(cache_dir, f'{id_}_{i}.npz')
                    if os.path.exists(cache_file):
                        melspec = np.load(cache_file, allow_pickle=False)['melspec']
                        assert melspec.shape[0] > 0
                    else:
                        if i == 0:
                            waveform2 = waveform
                        else:
                            waveform2 = augmentation.augment(waveform)
                        melspec = melspec_vocoder.encode(waveform2)
                        np.savez(cache_file, melspec=melspec)

                    if melspec.shape[0] > audio_len:
                        melspec = melspec[:audio_len, :]
                    elif melspec.shape[0] < audio_len:
                        diff = audio_len - melspec.shape[0]
                        melspec = np.concatenate([melspec, melspec[-diff:, :]], axis=0)
                    melspec = melspec.astype(np.float32)
                    #assert melspec.shape[0] == audio_len, f'{audio.shape} {melspec.shape}'
                    assert melspec.dtype == np.float32
                    melspec_f.write(bytes(memoryview(melspec)))

def preprocess_jvs(args):
    corpus = readcorpus_jvs(CORPUSDATA_PATH['jvs'])
    f0_floor, f0_ceil = F0_RANGE[args.dataset]

    sr = SAMPLE_RATE

    wavs_dir = WAVDATA_PATH[args.dataset]
    cache_dir = os.path.join(WAVDATA_PATH[args.dataset], f'cache-{sr}')

    text_file = os.path.join('data', f'{args.dataset}-text-{sr}')
    audio_file = os.path.join('data', f'{args.dataset}-audio-{sr}')
    os.makedirs(cache_dir, exist_ok=True)

    with open_index_data_for_write(text_file) as text_f:
        with open_index_data_for_write(audio_file) as audio_f:
            for id_, monophone in tqdm(corpus):
                print(monophone)
                text = encode_text(monophone)
                assert '..' not in id_ # Just make sure the file name is under the directory.
                mp3_file = os.path.join(wavs_dir % id_)
                cache_file = os.path.join(cache_dir, f'{id_}.npz')
                if os.path.exists(cache_file):
                    audio = np.load(cache_file, allow_pickle=False)['audio']
                    assert audio.shape[0] > 0
                else:
                    x = readmp3(mp3_file)
                    audio = encode_audio(x, f0_floor, f0_ceil)
                    np.savez(cache_file, audio=audio)

                text_f.write(bytes(memoryview(text)))
                audio_f.write(bytes(memoryview(audio)))

def preprocess_ctc_commonvoice(args):

    dataset = args.dataset
    sr = args.sample_rate

    corpus = readcorpus_commonvoice(CORPUSDATA_PATH[dataset])
    wavs_dir = os.path.join(os.path.dirname(CORPUSDATA_PATH[dataset]), 'clips')
    cache_dir = os.path.join(os.path.dirname(CORPUSDATA_PATH[dataset]), f'cache-ctc-{sr}')

    phone_file = os.path.join('data', f'{dataset}-ctc-phone-{sr}')
    melspec_file = os.path.join('data', f'{dataset}-ctc-melspec-{sr}')
    os.makedirs(cache_dir, exist_ok=True)

    encoder = PhoneEncoder()
    vocoder = MelSpectrogramVocoder()
    augmentation = AudioAugmentation()

    with open_index_data_for_write(phone_file) as phone_f:
        with open_index_data_for_write(melspec_file) as melspec_f:
            for id_, monophone in tqdm(corpus):
                phone = encoder.encode(monophone)
                assert '..' not in id_ # Just make sure the file name is under the directory.
                mp3_file = os.path.join(wavs_dir, id_)
                waveform = readmp3(mp3_file)
                for i in range(args.augment_count):
                    if i == 0:
                        augmented_waveform = waveform
                    else:
                        cache_file = os.path.join(cache_dir, f'{id_}_{i}.npz')
                        if os.path.exists(cache_file):
                            augmented_waveform = np.load(cache_file, allow_pickle=False)['waveform']
                            assert augmented_waveform.shape[0] > 0
                        else:
                            augmented_waveform = augmentation.augment(waveform, change_pace=True)
                            np.savez(cache_file, waveform=augmented_waveform)
                    melspec = vocoder.encode(augmented_waveform)
                    phone_f.write(bytes(memoryview(melspec)))
                    melspec_f.write(bytes(memoryview(audio)))

def compute_normalization(args, sample_rate=SAMPLE_RATE):
    from .data import IndexDataFileReader
    reader = IndexDataFileReader('data/%s-audio-%d' % (args.dataset, sample_rate))
    audio_list = []
    for index in tqdm(range(len(reader))):
        audio = np.frombuffer(reader[index], dtype=np.float32).reshape((-1, AUDIO_DIM))
        audio_list.append(audio)
    reader.close()
    audio = np.concatenate(audio_list, axis=0)

    f0 = audio[:, 0]
    f0mean = np.mean(f0[f0 >= 30])
    f0std = np.std(f0[f0 >= 30])

    mean = np.mean(audio, axis=0)
    std = np.std(audio, axis=0)
    mean[0] = f0mean
    std[0] = f0std
    x = np.stack([mean, std], axis=1)
    for i in range(x.shape[0]):
        print('    [%s, %s],' % (str(x[i, 0]), str(x[i, 1])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true', help='Analyze F0 of sampled data.')
    parser.add_argument('--ctc', action='store_true', help='Preprocess data for speech recognition task')
    parser.add_argument('--vc', action='store_true', help='Preprocess data for voice convertion task')
    parser.add_argument('--normalize', action='store_true', help='Compute normalization parameters.')
    parser.add_argument('--dataset', required=True, help='Dataset to process, css10ja, tsukuyomi_normal')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sampling rate')
    parser.add_argument('--augment_count', default=5, type=int, help='Number of augmented data for one clip.')
    args = parser.parse_args()

    if not (args.ctc or args.vc):
        raise ValueError('One of --ctc or --vc must be given.')

    if args.analyze:
        if args.dataset == 'css10ja':
            analyze_css10ja(args.dataset)
        else:
            analyze_jvs(args.dataset)
    elif args.vc:
        preprocess_vc_ljcorpus(args.dataset)
    elif args.normalize:
        compute_normalization(args)
    else:
        if args.dataset == 'css10ja' or args.dataset == 'css10ja_highpitch':
            preprocess_css10ja(args.dataset)
        elif args.dataset.startswith('kokoro_'):
            preprocess_ctc_ljcorpus(args)
        elif args.dataset.startswith('cv_'):
            preprocess_ctc_commonvoice(args)
        else:
            preprocess_jvs(args)
