# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import MeCab
from ._yomi2voca import yomi2voca
from typing import List, Tuple

_tagger = MeCab.Tagger()

_katakana = ''.join(chr(ch) for ch in range(ord('ァ'), ord('ン') + 1))
_hiragana = ''.join(chr(ch) for ch in range(ord('ぁ'), ord('ん') + 1))
_kata2hiratrans = str.maketrans(_katakana, _hiragana)
_symbols_tokens = set(['・', '、', '。', '？', '！'])
_no_yomi_tokens = set(['「', '」', '『', '』', '―', '（', '）', '［', '］', '[', ']', '　', '…'])

def kata2hira(text):
    text = text.translate(_kata2hiratrans)
    return text.replace('ヴ', 'う゛')

def getyomi(text) -> List[Tuple[str, str]]:
    parsed = _tagger.parse(text)
    res = []
    for line in parsed.split('\n'):
        if line == 'EOS':
            break
        parts = line.split('\t')

        word, yomi = parts[0], parts[1]
        if yomi:
            if yomi in ['Ｋ']:
                yomi = 'けい'
            else:
                yomi = kata2hira(yomi)
            res.append((word, yomi))
        else:
            if word in _symbols_tokens:
                res.append((word, word))
            elif word == 'っ' or word == 'ッ':
                res.append((word, 'っ'))
            elif word in _no_yomi_tokens:
                res.append((word, ''))
            else:
                res.append((word, word))
    return res

def text2voca(text: str, ignore_error: bool = False) -> List[Tuple[str, str]]:
    """Convert text to phonemes.
    """
    return [
        (text_, yomi2voca(yomi_, ignore_error=ignore_error))
        for text_, yomi_ in getyomi(text)
    ]
