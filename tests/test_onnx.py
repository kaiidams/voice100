# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import unittest
import torch
import onnxruntime as ort
import librosa

from voice100.datasets import get_audio_transform, get_text_transform


class TestEncoder(unittest.TestCase):
    @unittest.skip("Need ONNX file")
    def test(self):
        audio_file = './data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac'
        onnx_file = "model/onnx/asr_en_conv_base_ctc-20220126.onnx"
        sess = ort.InferenceSession(onnx_file)
        audio_transform = get_audio_transform(vocoder="mel", sample_rate=16000)
        text_transform = get_text_transform(language="en", use_phone=False)

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


if __name__ == '__main__':
    unittest.main()
