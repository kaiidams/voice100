# Copyright (C) 2022 Katsuya Iida. All rights reserved.

import os
from typing import List, Text
import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from voice100.models import TextToAlignText
from voice100.models import AlignTextToAudio
from voice100.data_modules import get_tokenizer
from voice100.data_modules import get_audio_transform


def get_phonemizer(language: Text, use_phone: bool):
    if language == "en":
        if use_phone:
            from .text import CMUPhonemizer
            return CMUPhonemizer()
        else:
            from .text import BasicPhonemizer
            return BasicPhonemizer()
    elif language == "ja":
        from .japanese import JapanesePhonemizer
        return JapanesePhonemizer(use_phone=use_phone)
    else:
        raise ValueError(f"Unknown language `{language}'")


def make_samples(
    align_ckpt_path: Text,
    audio_ckpt_path: Text,
    sample_texts: List[Text],
    language: Text,
):
    align_model = TextToAlignText.load_from_checkpoint(align_ckpt_path, strict=False)
    _ = align_model.eval()

    phonemizer = get_phonemizer(language=language, use_phone=True)
    tokenizer = get_tokenizer(language=language, use_phone=True)
    phone = [phonemizer(t) for t in sample_texts]
    text = [tokenizer(p) for p in phone]
    text_len = torch.tensor([t.shape[0] for t in text])
    text = pad_sequence(text, batch_first=True)

    for i in range(len(sample_texts)):
        x = sample_texts[i]
        print(f"text {i}: {x}")

    with torch.no_grad():
        align, align_len = align_model.predict(text, text_len)

    for i in range(text.shape[0]):
        x = tokenizer.decode(text[i, :text_len[i]])
        print(f"decoded {i}: {x[:100]}...")

    aligntext = []
    aligntext_len = []
    for i in range(align.shape[0]):
        x = align_model.align(
            text[i, :text_len[i]],
            align[i, :align_len[i]])
        aligntext.append(x)
        aligntext_len.append(x.shape[0])

    aligntext_len = torch.tensor(aligntext_len, dtype=torch.int64)
    aligntext = pad_sequence(aligntext, batch_first=True)

    for i in range(aligntext.shape[0]):
        x = tokenizer.decode(aligntext[i, :aligntext_len[i]])
        print(f"aligntext {i}: {x[:100]}...")

    audio_model = AlignTextToAudio.load_from_checkpoint(audio_ckpt_path, strict=False)
    _ = audio_model.eval()

    with torch.no_grad():
        f0_hat, logspc_hat, codeap_hat = audio_model.predict(aligntext, aligntext_len)

    for i in range(f0_hat.shape[0]):
        audio_transform = get_audio_transform("world_mcep", 16000)
        audio_len = aligntext_len[i] * 2
        waveform_hat = audio_transform.vocoder.decode(
            f0_hat[i, :audio_len],
            logspc_hat[i, :audio_len],
            codeap_hat[i, :audio_len])
        waveform_hat = np.clip(waveform_hat, -0.8, 0.8)
        waveform_hat = (waveform_hat * 32765).astype(np.int16)
        waveform_hat = torch.from_numpy(waveform_hat)
        waveform_hat = torch.unsqueeze(waveform_hat, 0)
        torchaudio.save(f"sample-{language}-{i + 1}.wav", waveform_hat, 16000)


def cli_main():
    make_samples(
        align_ckpt_path="https://github.com/kaiidams/voice100/releases/download/v1.5.3/align_en_phone_base-20230407.ckpt",
        audio_ckpt_path="https://github.com/kaiidams/voice100/releases/download/v1.5.3/tts_en_phone_base-20230401.ckpt",
        sample_texts=[
            "beginnings are apt to be determinative and when reinforced by continuous applications of similar influence",
            "which had restored the courage of noirtier for ever since he had conversed with the priest his violent"
            " despair had yielded to a calm resignation which surprised all who knew his excessive affection"
        ],
        language="en",
    )

    make_samples(
        align_ckpt_path="https://github.com/kaiidams/voice100/releases/download/v1.5.1/align_ja_phone_base-20230203.ckpt",
        audio_ckpt_path="https://github.com/kaiidams/voice100/releases/download/v1.5.1/tts_ja_phone_base-20230204.ckpt",
        sample_texts=[
            'また、東寺のように五大明王と呼ばれる主要な明王の中央に配されることも多い。',
            'ニューイングランド風は牛乳をベースとした白いクリームスープでありボストンクラムチャウダーとも呼ばれる'
        ],
        language="ja"
    )


if __name__ == "__main__":
    cli_main()
