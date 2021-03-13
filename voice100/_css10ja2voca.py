# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import re

_CSS_DOUBLE_RX = re.compile(r'([^-aiueo])\1')
_CSS_N_RX = re.compile(r'n(?![aiueo])')

def css10ja2voca(css):
    t = css.replace(' ', '')
    t = t.replace('。', '.')
    t = t.replace('、', ',')
    t = t.replace('？', '?')
    t = t.replace('jixy', 'j')
    t = t.replace('nixy', 'ny')
    t = t.replace('chixy', 'ch')
    t = t.replace('shixy', 'sh')
    t = t.replace('rexy', 'ry')
    t = t.replace('rixy', 'ry')
    t = t.replace('kexy', 'ky')
    t = t.replace('kixy', 'ky')
    t = t.replace('kuxy', 'ky')
    t = t.replace('v', 'b')
    t = t.replace('xtsu', 'q')
    t = _CSS_DOUBLE_RX.sub(r'q\1', t)
    t = _CSS_N_RX.sub(r'N', t)
    t = t.replace('wo', 'o')
    t = t.replace('-', ':')
    t = t.replace('―', ':')
    t = t.replace("'", '')
    return t
