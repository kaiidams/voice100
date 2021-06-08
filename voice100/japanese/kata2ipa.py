import re

_CONVRULES = [
    # Conversion of 2 letters
    "アァ/ a:",
    "イィ/ i:",
    "イェ/ j e",
    "イャ/ i a",
    "ウゥ/ u:",
    "エェ/ e:",
    "オォ/ o:",
    "カァ/ k a:",
    "キィ/ k i:",
    "クゥ/ k u:",
    "クャ/ k u a",
    "クュ/ k u u",
    "クョ/ k u o",
    "ケェ/ k e:",
    "コォ/ k o:",
    "ガァ/ g a:",
    "ギィ/ g i:",
    "グゥ/ g u:",
    "グャ/ g u a",
    "グュ/ g u u",
    "グョ/ g u o",
    "ゲェ/ g e:",
    "ゴォ/ g o:",
    "サァ/ s a:",
    "シィ/ s\\ i:",
    "スゥ/ s u:",
    "スャ/ s u a",
    "スュ/ s u u",
    "スョ/ s u o",
    "セェ/ s e:",
    "ソォ/ s o:",
    "ザァ/ z a:",
    "ジィ/ d_z\\ i:",
    "ズゥ/ z u:",
    "ズャ/ z u a",
    "ズュ/ z u u",
    "ズョ/ z u o",
    "ゼェ/ z e:",
    "ゾォ/ z o:",
    "タァ/ t a:",
    "チィ/ t_s\\ i:",
    "ツァ/ t_s a",
    "ツィ/ t_s w i",
    "ツゥ/ t_s u:",
    "ツャ/ t_s u a",
    "ツュ/ t_s u u",
    "ツョ/ t_s u o",
    "ツェ/ t_s e",
    "ツォ/ t_s o",
    "テェ/ t e:",
    "トォ/ t o:",
    "ダァ/ d a:",
    "ヂィ/ d_z\\ i:",
    "ヅゥ/ z u:",
    "ヅャ/ z u a",
    "ヅュ/ z u u",
    "ヅョ/ z u o",
    "デェ/ d e:",
    "ドォ/ d o:",
    "ナァ/ n a:",
    "ニィ/ n i:",
    "ヌゥ/ n u:",
    "ヌャ/ n u a",
    "ヌュ/ n u u",
    "ヌョ/ n u o",
    "ネェ/ n e:",
    "ノォ/ n o:",
    "ハァ/ h a:",
    "ヒィ/ C i:",
    "フゥ/ p\\ u:",
    "フャ/ p\\ u a",
    "フュ/ p\\ u u",
    "フョ/ p\\ u o",
    "ヘェ/ h e:",
    "ホォ/ h o:",
    "バァ/ b a:",
    "ビィ/ b i:",
    "ブゥ/ b u:",
    "フャ/ p\\ u a",
    "ブュ/ b u u",
    "フョ/ p\\ u o",
    "ベェ/ b e:",
    "ボォ/ b o:",
    "パァ/ p a:",
    "ピィ/ p i:",
    "プゥ/ p u:",
    "プャ/ p u a",
    "プュ/ p u u",
    "プョ/ p u o",
    "ペェ/ p e:",
    "ポォ/ p o:",
    "マァ/ m a:",
    "ミィ/ m i:",
    "ムゥ/ m u:",
    "ムャ/ m u a",
    "ムュ/ m u u",
    "ムョ/ m u o",
    "メェ/ m e:",
    "モォ/ m o:",
    "ヤァ/ j a:",
    "ユゥ/ j u:",
    "ユャ/ j u a",
    "ユュ/ j u u",
    "ユョ/ j u o",
    "ヨォ/ j o:",
    "ラァ/ r` a:",
    "リィ/ r` i:",
    "ルゥ/ r` u:",
    "ルャ/ r` u a",
    "ルュ/ r` u u",
    "ルョ/ r` u o",
    "レェ/ r` e:",
    "ロォ/ r` o:",
    "ワァ/ w a:",
    "ヲォ/ o:",
    "ディ/ d i",
    "デェ/ d e:",
    "デャ/ d e a",
    "デュ/ d _j u",
    "デョ/ d e o",
    "ティ/ t i",
    "テェ/ t e:",
    "テャ/ t e a",
    "テュ/ t _j u",
    "テョ/ t e o",
    "スィ/ s u",
    "ズァ/ z u",
    "ズィ/ z u",
    "ズゥ/ z u:",
    "ズャ/ z u a",
    "ズュ/ z u u",
    "ズョ/ z u o",
    "ズェ/ z u",
    "ズォ/ z u",
    "キャ/ k _j a",
    "キュ/ k _j u",
    "キョ/ k _j o",
    "シャ/ s\\ a",
    "シュ/ s\\ u",
    "シェ/ s\\ e:",
    "ショ/ s\\ o",
    "チャ/ t_s\\ a",
    "チュ/ t_s\\ u",
    "チェ/ t_s\\ e",
    "チョ/ t_s\\ o",
    "トゥ/ t u",
    "トャ/ t o a",
    "トュ/ t o u",
    "トョ/ t o o",
    "ドァ/ d o",
    "ドゥ/ d u",
    "ドャ/ d o a",
    "ドュ/ d o u",
    "ドョ/ d o o",
    "ドォ/ d o:",
    "ニャ/ n _j a",
    "ニュ/ n _j u",
    "ニョ/ n _j o",
    "ヒャ/ C a",
    "ヒュ/ C u",
    "ヒョ/ C o",
    "ミャ/ m _j a",
    "ミュ/ m _j u",
    "ミョ/ m _j o",
    "リャ/ r` _j a a",
    "リュ/ r` _j a u",
    "リョ/ r` _j a o",
    "ギャ/ g _j a",
    "ギュ/ g _j u",
    "ギョ/ g _j o",
    "ヂェ/ d_z\\ i",
    "ヂャ/ d_z\\ a",
    "ヂュ/ d_z\\ u",
    "ヂョ/ d_z\\ o",
    "ジェ/ d_z\\ e:",
    "ジャ/ d_z\\ a",
    "ジュ/ d_z\\ u",
    "ジョ/ d_z\\ o",
    "ビャ/ b _j a",
    "ビュ/ b _j u",
    "ビョ/ b _j o",
    "ピャ/ p _j a",
    "ピュ/ p _j u",
    "ピョ/ p _j o",
    "ウァ/ u",
    "ウィ/ w i",
    "ウェ/ w e",
    "ウォ/ w o",
    "ファ/ p\\ a",
    "フィ/ p\\ i",
    "フゥ/ p\\ u:",
    "フャ/ p\\ u a",
    "フュ/ p\\ u u",
    "フョ/ p\\ u o",
    "フェ/ p\\ e",
    "フォ/ p\\ o",
    "ヴァ/ v a",
    "ヴィ/ v i",
    "ヴェ/ v e",
    "ヴォ/ v o",
    "ヴュ/ v _j u",

    # Conversion of 1 letter
    "ア/ a",
    "イ/ i",
    "ウ/ u",
    "エ/ e",
    "オ/ o",
    "カ/ k a",
    "キ/ k i",
    "ク/ k u",
    "ケ/ k e",
    "コ/ k o",
    "サ/ s a",
    "シ/ s\\ i",
    "ス/ s u",
    "セ/ s e",
    "ソ/ s o",
    "タ/ t a",
    "チ/ t_s\\ i",
    "ツ/ t_s u",
    "テ/ t e",
    "ト/ t o",
    "ナ/ n a",
    "ニ/ n i",
    "ヌ/ n u",
    "ネ/ n e",
    "ノ/ n o",
    "ハ/ h a",
    "ヒ/ C i",
    "フ/ p\\ u",
    "ヘ/ h e",
    "ホ/ h o",
    "マ/ m a",
    "ミ/ m i",
    "ム/ m u",
    "メ/ m e",
    "モ/ m o",
    "ラ/ r` a",
    "リ/ r` i",
    "ル/ r` u",
    "レ/ r` e",
    "ロ/ r` o",
    "ガ/ g a",
    "ギ/ g i",
    "グ/ g u",
    "ゲ/ g e",
    "ゴ/ g o",
    "ザ/ z a",
    "ジ/ d_z\\ i",
    "ズ/ z u",
    "ゼ/ z e",
    "ゾ/ z o",
    "ダ/ d a",
    "ヂ/ d_z\\ i",
    "ヅ/ z u",
    "デ/ d e",
    "ド/ d o",
    "バ/ b a",
    "ビ/ b i",
    "ブ/ b u",
    "ベ/ b e",
    "ボ/ b o",
    "パ/ p a",
    "ピ/ p i",
    "プ/ p u",
    "ペ/ p e",
    "ポ/ p o",
    "ヤ/ j a",
    "ユ/ j u",
    "ヨ/ j o",
    "ワ/ w a",
    "ヰ/ i",
    "ヱ/ w e",
    "ヲ/ o",
    "ン/ N\\",
    "ッ/ ",
    "ヴ/ v u",
    "ー/:",

    # Try converting broken text
    "ァ/ a",
    "ィ/ i",
    "ゥ/ u",
    "ェ/ e",
    "ォ/ o",
    "ヮ/ w a",
    "ォ/ o",

    "、/ ",
    "。/ ",
    "！/ ",
    "？/ ",
    "・/ ",
]

_COLON_RX = re.compile(':+')

def _makerulemap():
    l = [tuple(x.split('/')) for x in _CONVRULES]
    return tuple(
        {k: v for k, v in l if len(k) == i}
        for i in (1, 2)
    )

_RULEMAP1, _RULEMAP2 = _makerulemap()

def kata2asciiipa(text: str) -> str:
    """Convert katakana text to ASCII IPA.
    """
    text = text.strip()
    res = ''
    while text:
        if len(text) >= 2:
            x = _RULEMAP2.get(text[:2])
            if x is not None:
                text = text[2:]
                res += x
                continue
        x = _RULEMAP1.get(text[0])
        if x is not None:
            text = text[1:]
            res += x
            continue
        res +=   + text[0]
        text = text[1:]
    res = _COLON_RX.sub(':', res)
    return res[1:]