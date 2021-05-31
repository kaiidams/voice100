import unittest
import os
import numpy as np
from tqdm import tqdm
from .vocoder import readwav, writewav, decode_audio, encode_audio, SAMPLE_RATE, AUDIO_DIM
from .encoder import decode_text

class TestPreprocess(unittest.TestCase):

    def test_data(self, name='cv_ja_kokoro_tiny', sample_rate=SAMPLE_RATE):
        from .data import IndexDataFileReader
        reader = IndexDataFileReader('data/%s-text-%d' % (name, sample_rate))
        reader2 = IndexDataFileReader('data/%s-audio-%d' % (name, sample_rate))
        assert len(reader) == len(reader2)
        l = len(reader)
        for index in tqdm(range(0, l, l // 10)):
            text = np.frombuffer(reader[index], dtype=np.uint8)
            print(decode_text(text))
            audio = np.frombuffer(reader2[index], dtype=np.float32).reshape((-1, AUDIO_DIM))
            x = decode_audio(audio)
            file = 'data/synthesized/%s_%04d.wav' % (name, index)
            os.makedirs(os.path.dirname(file), exist_ok=True)
            writewav(file, x)
        reader.close()
        reader2.close()

    def test_data2(self, name='cv_ja_kokoro_tiny', sample_rate=SAMPLE_RATE):
        from .data import IndexDataDataset
        text_file = 'data/%s-text-%d' % (name, sample_rate)
        audio_file = 'data/%s-audio-%d' % (name, sample_rate)
        ds = IndexDataDataset(
            [text_file, audio_file], [(-1,), (-1, AUDIO_DIM)], [np.uint8, np.float32])

        l = len(ds)
        for index in tqdm(range(1, l, l // 10)):
            text, audio = ds[index]
            print(decode_text(text))
            x = decode_audio(audio)
            file = 'data/synthesized/%s_%04d.wav' % (name, index)
            os.makedirs(os.path.dirname(file), exist_ok=True)
            writewav(file, x)
        ds.close()

def test_data2(name='kokoro_tiny', sample_rate=SAMPLE_RATE):
    from .data import IndexDataFileReader
    reader = IndexDataFileReader('data/%s-text-%d' % (name, sample_rate))
    reader2 = IndexDataFileReader('data/%s-align-%d' % (name, sample_rate))
    assert len(reader) == len(reader2)
    l = len(reader)
    for index in tqdm(range(0, l, l // 10)):
        text = np.array(reader[index], dtype=np.uint8)
        print(decode_text(text))
        align = np.array(reader2[index], dtype=np.uint8)
        print(decode_text(align))
        print(len(align))

def test_data3():
    x = readwav('data/tsuchiya_normal/tsuchiya_normal_001.wav', 22050)
    #y = encode_audio(x, 57.46701428196299, 196.7528135117272)
    y = encode_audio(x, 57.46701428196299, 396.7528135117272)
    x2 = decode_audio(y)
    writewav('data/a.wav', x2)

def main():
    test_data('cv_ja')
    #test_data2('cv_ja')
    #test_data3('cv_ja')

if __name__ == '__main__':
    unittest.main()