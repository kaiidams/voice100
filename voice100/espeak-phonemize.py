import os
from glob import glob
from argparse import ArgumentParser
import ctypes

AUDIO_OUTPUT_SYNCHRONOUS = 2
NULL = 0
espeakCHARS_UTF8 = 1
EE_OK = 0
EE_INTERNAL_ERROR = -1

espeak = ctypes.CDLL("libespeak-ng.so")
espeak.espeak_TextToPhonemes.restype = ctypes.c_char_p

def text_to_phonemes(text):
    text = ctypes.c_char_p(text.encode('utf-8'))
    phonemes = []
    while True:
        phonememode = ctypes.c_int(ord('\t') << 8)
        textmode = ctypes.c_int(espeakCHARS_UTF8)
        s = espeak.espeak_TextToPhonemes(ctypes.byref(text), textmode, phonememode)
        if not s:
            break
        s = s.decode('utf-8').replace(' ', '\t_\t').split('\t')
        phonemes.append(' '.join(s))    
    return ' _ '.join(phonemes)

def main():
    parser = ArgumentParser()
    args = parser.parse_args()
    args.voicename = 'English (America)'
    #args.voicename = 'Japanese'

    res = espeak.espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 500, NULL, 0)
    if res == EE_INTERNAL_ERROR:
        raise Exception("espeak_Initialize failed")

    res = espeak.espeak_SetVoiceByName(ctypes.c_char_p(args.voicename.encode('utf-8')))
    if res != EE_OK:
        raise Exception(f"espeak_SetVoiceByName failed with {res}")

    root = 'LibriSpeech/train-clean-100/'
    for file in glob(os.path.join(root, '**', '*.txt'), recursive=True):
        dirpath = os.path.dirname(file)
        assert dirpath.startswith(root)
        dirpath = dirpath[len(root):]
        with open(file) as f:
            for line in f:
                audio, _, text = line.rstrip('\r\n').partition(' ')
                audio = os.path.join(dirpath, audio + '.flac')
                text2 = text_to_phonemes(text)
                print(audio, text, text2)
    #convert_common_voice(root, tsv='validated.tsv')

if __name__ == '__main__':
    main()