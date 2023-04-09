# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import pytest
import torch

from voice100.data_modules import get_audio_transform, get_tokenizer


@pytest.mark.skip("Need ONNX file")
def test_onnx_asr():
    import onnxruntime as ort
    import librosa
    audio_file = './data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac'
    onnx_file = "model/onnx/asr_en_conv_base_ctc-20220126.onnx"
    sess = ort.InferenceSession(onnx_file)
    audio_transform = get_audio_transform(vocoder="mel", sample_rate=16000)
    text_transform = get_tokenizer(language="en", use_phone=False)

    waveform, sr = librosa.load(audio_file, 16000)
    waveform = torch.from_numpy(waveform)

    audio = torch.log(audio_transform.melspec(waveform).T + audio_transform.log_offset)

    ort_inputs = {
        "audio": audio[
            None, :, :
        ].numpy(),  # np.zeros(shape=[1, 123, 64]).astype(dtype=np.float32),
    }

    if False:
        import time
        s = time.time()
        for i in range(100):
            logits, logits_len = sess.run("logits", ort_inputs)
            t = time.time()
            print(t - s)
            s = t

    (logits,) = sess.run(["logits"], ort_inputs)
    pred = logits.argmax(-1)
    aligntext = text_transform.tokenizer.decode(pred[0])
    text = text_transform.tokenizer.merge_repeated(aligntext)
    print(text)


@pytest.mark.skip("Need ONNX file")
def test_onnx_tts():
    import onnxruntime as ort
    import numpy as np

    def align_text(
        text: np.ndarray,
        align: np.ndarray,
        head=5, tail=5
    ) -> np.ndarray:

        assert text.ndim == 1
        assert align.ndim == 2
        aligntext_len = head + int(np.sum(align)) + tail
        aligntext = np.zeros(aligntext_len, dtype=text.dtype)
        t = head
        for i in range(align.shape[0]):
            t += align[i, 0].item()
            s = round(t)
            t += align[i, 1].item()
            e = round(t)
            if s == e:
                e = max(0, e + 1)
            for j in range(s, e):
                aligntext[j] = text[i]
        return aligntext

    onnx_file = "../voice100-runtime/align_ja_phone_base-20230203.onnx"
    sess = ort.InferenceSession(onnx_file)
    audio_transform = get_audio_transform(vocoder="mel", sample_rate=16000)
    text_transform = get_tokenizer(language="ja", use_align=False, use_phone=True)
    text = 'k o N n i ch i w a'
    text = text_transform(text)[None, :].numpy()
    text_len = np.array([text.shape[1]], dtype=np.int64)
    #print(text)
    ort_inputs = {
        "text": text,
        "text_len": text_len
    }

    (align,) = sess.run(["align"], ort_inputs)
    align = np.exp(align) - 1
    print((align * 100 + 0.5).astype(int))
    aligntext = align_text(text[0], align[0])
    print(aligntext)

    aligntext = text_transform.tokenizer.decode(aligntext)
    print(aligntext)
    text = text_transform.tokenizer.merge_repeated(aligntext)
    #print(text)


if __name__ == '__main__':
    test_onnx_tts()
