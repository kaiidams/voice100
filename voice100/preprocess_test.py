import soundfile as sf
import os
import numpy as np
from tqdm import tqdm
from .vocoder import readwav, writewav, decode_audio, encode_audio
from .encoder import decode_text

def test_data(name='css10ja'):
    file = 'data/%s_train.npz' % name
    f = np.load(file)
    data = {k: v for k, v in f.items()}
    for index in tqdm(range(10)):
        text_start = data['text_index'][index - 1] if index else 0
        text_end = data['text_index'][index]
        audio_start = data['audio_index'][index - 1] if index else 0
        audio_end = data['audio_index'][index]
        text = data['text_data'][text_start:text_end]
        audio = data['audio_data'][audio_start:audio_end, :]
        print(decode_text(text))
        x = decode_audio(audio)
        file = 'data/synthesized/%s_%04d.wav' % (name, index)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        writewav(file, x)

def test_data2(name='kokoro_tiny'):
    from .data import IndexDataFileReader
    reader = IndexDataFileReader('data/%s-text' % name)
    reader2 = IndexDataFileReader('data/%s_align' % name)
    for index in tqdm(range(10)):
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
    #test_data('tsukuyomi_normal')
    #test_data2()
    test_data3()

if __name__ == '__main__':
    main()