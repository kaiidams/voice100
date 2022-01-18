# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import unittest
import torch
import onnxruntime as ort
import librosa

from voice100.datasets import get_transform


class TestEncoder(unittest.TestCase):
    def test(self):

        sess = ort.InferenceSession("model/onnx/stt_en_conv_base_ctc-20211125.onnx")
        processor = get_transform(task="asr", sample_rate=16000, language="en", use_phone=False, infer=True)

        waveform, sr = librosa.load("test.flac", 16000)
        waveform = torch.from_numpy(waveform)

        encoder = processor.encoder
        audio = torch.log(processor.transform(waveform).T + processor.log_offset)

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
        print(encoder.merge_repeated(encoder.decode(pred[0])))


if __name__ == '__main__':
    unittest.main()
