# Copyright (C) 2022 Katsuya Iida. All rights reserved.

import os
from typing import List, Text
import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from voice100.models.tts import TextToAlignTextModel
from voice100.models.tts import AlignTextToAudioModel
from voice100.models.tts import AlignTextToAudioMultiTaskModel
from voice100.data_modules import get_phonemizer, get_tokenizer
from voice100.data_modules import get_audio_transform


def make_samples(
    align_model: Text,
    audio_model: Text,
    sample_texts: List[Text],
    language: Text,
    mt: bool = False
):
    ckpt_path = os.path.join("model", align_model + ".ckpt")
    align_model = TextToAlignTextModel.load_from_checkpoint(ckpt_path, strict=False)
    _ = align_model.eval()

    phonemizer = get_phonemizer(language=language, use_align=False, use_phone=False)
    tokenizer = get_tokenizer(language=language, use_phone=False)
    phone = [phonemizer(t) for t in sample_texts]
    text = [tokenizer(p) for p in phone]
    text_len = torch.tensor([t.shape[0] for t in text])
    text = pad_sequence(text, batch_first=True)

    for i in range(len(sample_texts)):
        print(sample_texts[i])

    with torch.no_grad():
        pred = torch.relu(align_model.forward(text))
        align = torch.exp(pred) - 1.

    for i in range(text.shape[0]):
        print(tokenizer.decode(
            text[i, :text_len[i]]))

    aligntext = []
    aligntext_len = []
    for i in range(text.shape[0]):
        at = align_model.align(
            text[i, :text_len[i]],
            align[i, :text_len[i]],
            head=5, tail=5)
        aligntext.append(at)
        aligntext_len.append(at.shape[0])
    aligntext = pad_sequence(aligntext, batch_first=True)

    for i in range(aligntext.shape[0]):
        print(tokenizer.decode(
            aligntext[i, :aligntext_len[i]]))

    ckpt_path = os.path.join("model", audio_model + ".ckpt")
    if mt:
        audio_model = AlignTextToAudioMultiTaskModel.load_from_checkpoint(ckpt_path, strict=False)
    else:
        audio_model = AlignTextToAudioModel.load_from_checkpoint(ckpt_path, strict=False)
    _ = audio_model.eval()

    with torch.no_grad():
        if mt:
            hasf0_logits, f0_hat, logspc_hat, codeap_hat, _ = audio_model.forward(aligntext)
        else:
            hasf0_logits, f0_hat, logspc_hat, codeap_hat = audio_model.forward(aligntext)
        f0_hat, logspc_hat, codeap_hat = audio_model.norm.unnormalize(f0_hat, logspc_hat, codeap_hat)
        f0_hat[hasf0_logits < 0] = 0.0

    for i in range(f0_hat.shape[0]):
        audio_transform = get_audio_transform("world", 16000)
        audio_len = aligntext_len[i] * 2
        waveform_hat = audio_transform.vocoder.decode(
            f0_hat[i, :audio_len],
            logspc_hat[i, :audio_len],
            codeap_hat[i, :audio_len])
        waveform_hat = np.clip(waveform_hat, -0.8, 0.8)
        sf.write(f"sample-{language}-{i + 1}.wav", waveform_hat, 16000, subtype="PCM_16")


def cli_main():
    if False:
        make_samples(
            align_model="ttsalign_en_conv_base-20220409",
            audio_model="ttsaudio_en_mt_conv_base-20220107",
            sample_texts=[
                "beginnings are apt to be determinative and when reinforced by continuous applications of similar influence",
                "which had restored the courage of noirtier for ever since he had conversed with the priest his violent"
                " despair had yielded to a calm resignation which surprised all who knew his excessive affection"
            ],
            language="en"
        )
    else:
        make_samples(
            align_model="ttsalign_en_conv_base-20220409",
            audio_model="ttsaudio_en_mt_conv_base-20220316",
            sample_texts=[
                "beginnings are apt to be determinative and when reinforced by continuous applications of similar influence",
                "which had restored the courage of noirtier for ever since he had conversed with the priest his violent"
                " despair had yielded to a calm resignation which surprised all who knew his excessive affection"
            ],
            language="en",
            mt=True,
        )

    make_samples(
        align_model="ttsalign_ja_conv_base-20220411",
        audio_model="ttsaudio_ja_conv_base-20220416",
        sample_texts=[
            'また、東寺のように五大明王と呼ばれる主要な明王の中央に配されることも多い。',
            'ニューイングランド風は牛乳をベースとした白いクリームスープでありボストンクラムチャウダーとも呼ばれる'
        ],
        language="ja"
    )


if __name__ == "__main__":
    cli_main()
