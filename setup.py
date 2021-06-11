from setuptools import setup

setup(
    name="voice100",
    version="0.3",
    author="Katsuya Iida",
    author_email="katsuya.iida@gmail.com",
    description="Voice100",
    license="MIT",
    url="https://github.com/kaiidams/voice100",
    packages=['voice100'],
    long_description="Voice100 is a small TTS for Japanese.",
    entry_points={
    },
    install_requires=[
        'torch',
        'pytorch_lightning'
    ],
    extras_require={
        "audio": [
            'torchaudio',
            #'librosa',
            #'soundfile',
            #'pyworld>=0.2.12',
            #'pysptk>=0.1.18',
        ],
        "lang-ja": [
            'mecab-python3',
            'unidic-lite',
        ]
    })
