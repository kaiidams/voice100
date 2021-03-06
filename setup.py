from setuptools import setup

setup(
    name="voice100",
    version="0.1",
    author="Katsuya Iida",
    author_email="katsuya.iida@gmail.com",
    description="Voice100",
    license="MIT",
    url="https://github.com/kaiidams/voice100",
    packages=['voice100'],
    long_description="Voice100 is a speech-text aligner.",
    install_requires=[
        'torch',
        'torchaudio'
    ],
    extras_require={
        "text": [
            'mecab-python3',
            'unidic-lite',
            'beautifulsoup4',
        ]
    })
