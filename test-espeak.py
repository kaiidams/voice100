import os
import ctypes

from voice100.japanese.phonemizer import text2kata

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

def convert_common_voice(root, tsv):
    with open(os.path.join(root, 'voice100_transcript.txt'), 'w') as outf:
        with open(os.path.join(root, tsv)) as f:
            line = f.readline()
            assert line.startswith('client_id')
            for line in f:
                parts = line.rstrip('\r\n').split('\t')
                path = parts[1]
                sentence = parts[2]
                kata = text2kata(sentence)
                phonemes = text_to_phonemes(kata)
                outf.write(f"clips/{path}\t{phonemes}\n")

def main():

    voicename = 'English (America)'
    voicename = 'Japanese'
    ja = False

    res = espeak.espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 500, NULL, 0)
    if res == EE_INTERNAL_ERROR:
        raise Exception("espeak_Initialize failed")

    res = espeak.espeak_SetVoiceByName(ctypes.c_char_p(voicename.encode('utf-8')))
    if res != EE_OK:
        raise Exception(f"espeak_SetVoiceByName failed with {res}")

    from voice100.japanese.phonemizer import _CONVRULES
    import re
    for l in _CONVRULES:
        t = re.sub('/.*$', '', l)
        text = ctypes.c_char_p(t.encode('utf-8'))
        phonememode = ctypes.c_int(ord('\t') << 8)
        textmode = ctypes.c_int(espeakCHARS_UTF8)
        s = espeak.espeak_TextToPhonemes(ctypes.byref(text), textmode, phonememode)
        s = s.decode('utf-8').replace('\t', ' ')
        if '(en)' not in s:
            l = (t + '/ ' + s)
        print(f'    "{l}",')

    root = '/home/kaiida/data/cv-corpus-6.1-2020-12-11/ja'
    #convert_common_voice(root, tsv='validated.tsv')

if __name__ == '__main__':
    main()