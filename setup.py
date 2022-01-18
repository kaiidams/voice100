from setuptools import setup

setup(
    name="voice100",
    version="1.1.0",
    author="Katsuya Iida",
    author_email="katsuya.iida@gmail.com",
    description="Voice100",
    license="MIT",
    url="https://github.com/kaiidams/voice100",
    packages=['voice100'],
    long_description="Voice100 is a small TTS for English and Japanese.",
    entry_points={
        "console_scripts": [
            "voice100-align-text=voice100.align_text:cli_main",
            "voice100-calc-stat=voice100.calc_stat:cli_main",
            "voice100-cache-dataset=voice100.cache_dataset:cli_main",
            "voice100-export-onnx=voice100.export_onnx:cli_main",
            "voice100-train-align=voice100.train_align:cli_main",
            "voice100-train-asr=voice100.train_asr:cli_main",
            "voice100-train-ttsalign=voice100.train_ttsalign:cli_main",
            "voice100-train-ttsaudio=voice100.train_ttsaudio:cli_main",
        ]
    },
    install_requires=[
        'torch',
        'torchaudio',
        'pytorch_lightning>=1.4.0'
    ],
    extras_require={
        "align": [
        ],
        "asr": [
        ],
        "tts": [
            'pyworld>=0.2.12',
        ],
        "lang-en": [
        ],
        "lang-en_phone": [
            "g2p-en"
        ],
        "lang-ja": [
            'mecab-python3',
            'unidic-lite',
        ]
    })
