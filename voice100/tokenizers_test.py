# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import unittest
from .tokenizers import *

class TestEncoder(unittest.TestCase):
    def test(self):
        tokenizer = CharTokenizer()
        print(tokenizer('hello world'))

if __name__ == '__main__':
    unittest.main()