# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import torch


def test_text_en():
    from voice100.text import BasicPhonemizer, CharTokenizer
    phonemizer = BasicPhonemizer()
    tokenizer = CharTokenizer()

    text = "Hello World!"
    phoneme = phonemizer(text)
    assert phoneme == "hello world"
    encoded = tokenizer(phoneme)
    assert encoded.shape == torch.Size([11])
    decoded = tokenizer.decode(encoded)
    assert decoded == "hello world"


def test_text_en_phone():
    from voice100.text import CMUPhonemizer, BasicTokenizer
    phonemizer = CMUPhonemizer()
    tokenizer = BasicTokenizer(language='en')

    text = "Hello World!"
    phoneme = phonemizer(text)
    assert phoneme == "HH/AH0/L/OW1/ /W/ER1/L/D/ /!"
    encoded = tokenizer(phoneme)
    assert encoded.shape == torch.Size([8])
    decoded = tokenizer.decode(encoded)
    assert decoded == "HH/AH0/L/OW1/W/ER1/L/D"


def test_text_ja():
    from voice100.japanese import JapanesePhonemizer
    from voice100.text import CharTokenizer
    phonemizer = JapanesePhonemizer()
    tokenizer = CharTokenizer()

    text = "こんにちは世界！"
    phoneme = phonemizer(text)
    assert phoneme == "kon'nichiwasekai"
    encoded = tokenizer(phoneme)
    assert encoded.shape == torch.Size([16])
    decoded = tokenizer.decode(encoded)
    assert decoded == "kon'nichiwasekai"

    text = "やっぱりヴォイス？"
    phoneme = phonemizer(text)
    assert phoneme == "ya'pariboisu"
    print(phoneme)
    encoded = tokenizer(phoneme)
    assert encoded.shape == torch.Size([12])
    decoded = tokenizer.decode(encoded)
    assert decoded == "ya'pariboisu"

    decoded = "yya'__pparri_"
    merged = tokenizer.merge_repeated(decoded)
    assert merged == "ya'pari"


def test_text_ja_phone():
    from voice100.japanese import JapanesePhonemizer
    from voice100.text import BasicTokenizer
    phonemizer = JapanesePhonemizer(use_phone=True)
    tokenizer = BasicTokenizer(language='ja')

    text = "こんにちは世界！"
    phoneme = phonemizer(text)
    assert phoneme == 'k o N n i ch i w a s e k a i !'
    encoded = tokenizer(phoneme)
    assert encoded.shape == torch.Size([15])
    decoded = tokenizer.decode(encoded)
    assert decoded == 'k o N n i ch i w a s e k a i !'

    text = "やっぱりヴォイス？"
    phoneme = phonemizer(text)
    assert phoneme == "y a q p a r i b o i s u ?"
    print(phoneme)
    encoded = tokenizer(phoneme)
    assert encoded.shape == torch.Size([13])
    decoded = tokenizer.decode(encoded)
    assert decoded == "y a q p a r i b o i s u ?"

    text = "「やっぱり」は★-Voice?"
    phoneme = phonemizer(text)
    assert phoneme == "y a q p a r i w a ★ b o i k e ?"
    encoded = tokenizer(phoneme)
    assert encoded.shape == torch.Size([15])
    decoded = tokenizer.decode(encoded)
    assert decoded == "y a q p a r i w a b o i k e ?"

    phoneme = "k o N n i - ch i あ w a C a v u"
    encoded = tokenizer(phoneme)
    assert encoded.shape == torch.Size([12])
    decoded = tokenizer.decode(encoded)
    assert decoded == 'k o N n i - ch i w a a u'

    decoded = "- - k o o N - n - - i - ch - i i w a - a -"
    merged = tokenizer.merge_repeated(decoded)
    assert merged == 'k o N n i ch i w a a'
