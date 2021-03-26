from setuptools import setup

setup(
    name="voice100-tts",
    version="0.0.1",
    author="Katsuya Iida",
    author_email="katsuya.iida@gmail.com",
    description="Voice100 TTS",
    license="MIT",
    url="https://github.com/kaiidams/voice100",
    packages=['voice100'],
    long_description="Voice100 is a small TTS for Japanese.",
    entry_points={
        "console_scripts": [
            "voice100-preprocess = voice100.preprocess",
            "voice100-train = voice100.train"
        ]
    },
    install_requires=[
        'tensorflow'
    ],
    extras_require={
        "preprocess": [
            'librosa',
            'soundfile',
            'pyworld>=0.2.12',
            'pysptk>=0.1.18',
            'tqdm'
        ],
        "text": [
            'mecab-python3',
            'unidic-lite',
        ]
    })
