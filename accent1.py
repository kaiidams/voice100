import os
from glob import glob
import numpy as np
import re
from tqdm import tqdm

import torchaudio
import pyworld

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


def read_metadata():
    with open('data/kokoro-speech-v1_3-large/metadata.csv', 'rt') as fp:
        for idx, line in enumerate(fp):
            parts = line.rstrip('\r\n').split('|')
            x = parts[1]
            x = x.replace(' ・ ', ' , ')
            x = x.replace('・', ' , ')
            x = x.replace('「', '')
            x = x.replace('」', '')
            x = re.sub(r'[―…（）　］『』ー‥]', '', x)
            x = re.sub(r' +', ' ', x)
            x = x.strip()
            x = x.split(' ')
            y = parts[2].replace(',', '_ , _')
            y = y.replace('.', '_ . _')
            y = y.replace('?', '_ ? _')
            y = y.replace('!', '_ ! _')
            y = y.strip()
            y = y.split(' _ ')
            if len(x) != len(y):
                print(idx, line)
                for x, y in zip(x, y):
                    print(x, y)
                assert False
            yield parts[0], [(x, y) for x, y in zip(x, y)]


def read_align():
    with open('data/kokoro_large-phone-align-train.txt', 'rt') as fp:
        for idx, line in enumerate(fp):
            parts = line.rstrip('\r\n').split('|')
            yield parts[1].split(' ')


def convert_text():
    clip_id = []
    res = []
    for a, (id_, x) in zip(read_align(), read_metadata()):
        a.append('@')
        b = []
        i = 0
        while a[i] == '-':
            b.append(('', 0, a[i]))
            i += 1
        while x:
            word, phone = x[0]
            x = x[1:]
            for j, p in enumerate(phone.split(' ')):
                while a[i] == p:
                    b.append((word, j, a[i]))
                    i += 1
                while a[i] == '-':
                    b.append(('', 0, a[i]))
                    i += 1
        assert a[i] == '@', a[i]
        clip_id.append(id_)
        res.append(b)
        # if len(res) >= 10:
        #     break

    with open('text.txt', 'w') as fp:
        for x in res:
            line = ' '.join([
                '%s/%d/%s' % tuple(y)
                for y in x
            ])
            for y in x:
                assert '/' not in y[0]
            fp.write(line + '\n')

    # clip_id = pa.array(clip_id, type=pa.string())
    # arr = pa.array(res, type=pa.list_(
    #     pa.struct([
    #         ('word', pa.string()),
    #         ('pos', pa.uint8()),
    #         ('token', pa.string()),
    #     ])))
    # return pa.table([clip_id, arr], names=['clip_id', 'text'])


def convert_pitch(clip_id):
    names = []
    data = []
    #os.makedirs('accent_data', exist_ok=True)
    for clip_id_ in tqdm(clip_id):
        fname = f'data/kokoro-speech-v1_3-large/wavs/{clip_id_}.flac'
        waveform, sr = torchaudio.load(fname)
        waveform = waveform.cpu().numpy().astype(np.double)
        assert waveform.shape[0] == 1
        fp, time_axis = pyworld.dio(
            waveform[0], sr, f0_floor=80.0,
            f0_ceil=400.0, frame_period=10.0)
        fp = fp.astype(np.float32)
        out_fname = os.path.basename(fname)
        out_fname, _ = os.path.splitext(out_fname)
        #out_fname = os.path.join('accent_data', f'{out_fname}.bin')
        names.append(out_fname)
        data.append(fp)
        # if len(data) >= 10:
        #     break

    clip_id = pa.array(names, type=pa.string())
    pitch = pa.array(data, type=pa.list_(pa.float32()))
    table = pa.table([clip_id, pitch], names=['clip_id', 'pitch'])
    return table


def preprocess1():
    with open('text.txt') as fp:
        data = []
        for line in fp:
            data.append(line.strip().split(' '))
    with np.load('pitch.npz') as arr:
        pitch_offsets = arr['offsets']
        pitch_values = arr['values']
    offsets = [0]
    values = []
    for idx, x in enumerate(data):
        s = pitch_offsets[idx]
        e = pitch_offsets[idx + 1]
        v = pitch_values[s:e]
        if v.shape[0] % 2 == 1:
            v = np.pad(v, [0, 1])
        v = v.reshape(-1, 2)
        c = np.sum((v > 1.0).astype(np.float32), axis=1)
        c = np.clip(c, a_min=1e-5, a_max=1e10)
        v = np.sum(v, axis=1) / c
        if v.shape[0] != len(x):
            v = np.pad(v, [0, 1])
        assert len(x) == v.shape[0], f'{len(x)} {v.shape[0]} {idx} {c}'
        values.append(v)
        offsets.append(offsets[-1] + v.shape[0])
    offsets = np.array(offsets, dtype=np.int32)
    values = np.concatenate(values)
    np.savez('pitch2.npz', offsets=offsets, values=values)
    # pitch = convert_pitch(text['clip_id'])

    # print(pc.all(text['clip_id'] == pitch['clip_id']))
    # # text.set_index('clip_id', inplace=True)
    # # text.update(pitch.set_index('clip_id'))
    # # text.reset_index()
    # # table = text.join(pitch, 'clip_id')
    # table = text.append_column('pitch', pitch['pitch'])

    # pq.write_table(table, 'accent.parquet')


def main():
    preprocess1()


main()
