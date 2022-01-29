# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import unittest
import torch
from voice100.text import BasicPhonemizer, CharTokenizer


class TestEncoder(unittest.TestCase):
    def test(self):
        phonemizer = BasicPhonemizer()
        tokenizer = CharTokenizer()

        text = "Hello World!"
        phoneme = phonemizer(text)
        self.assertEqual("hello world", phoneme)
        encoded = tokenizer(phoneme)
        self.assertEqual(torch.Size([11]), encoded.shape)
        print(encoded)

    def test2(self):
        from voice100.japanese import JapanesePhonemizer
        phonemizer = JapanesePhonemizer()
        tokenizer = CharTokenizer()

        text = "こんにちは世界！"
        phoneme = phonemizer(text)
        self.assertEqual("kon'nichiwasekai", phoneme)
        encoded = tokenizer(phoneme)
        self.assertEqual(torch.Size([16]), encoded.shape)
        print(encoded)

        text = "やっぱりヴォイス？"
        phoneme = phonemizer(text)
        print(phoneme)


if __name__ == '__main__':
    unittest.main()
