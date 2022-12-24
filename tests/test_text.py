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


def test_text_en_phone():
    from voice100.text import CMUPhonemizer, CMUTokenizer
    phonemizer = CMUPhonemizer()
    tokenizer = CMUTokenizer()

    text = "Hello World!"
    phoneme = phonemizer(text)
    assert phoneme == "HH/AH0/L/OW1/ /W/ER1/L/D/ /!"
    encoded = tokenizer(phoneme)
    assert encoded.shape == torch.Size([8])


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

    text = "やっぱりヴォイス？"
    phoneme = phonemizer(text)
    assert phoneme == "ya'pariboisu"
    print(phoneme)
    encoded = tokenizer(phoneme)
    assert encoded.shape == torch.Size([12])
