import time
import onnxruntime as ort
import numpy as np
from voice100.vocoder import MelSpectrogramVocoder
from voice100.text import CharTokenizer

sess = ort.InferenceSession('test.onnx')
import librosa
waveform, sr = librosa.load('test.flac', 16000)
vocoder = MelSpectrogramVocoder()
encoder = CharTokenizer()
audio = vocoder.encode(waveform)

ort_inputs = {
    'audio': audio[None, :, :], #np.zeros(shape=[1, 123, 64]).astype(dtype=np.float32),
    'audio_len': np.array([audio.shape[0]], dtype=np.int32)
}

s = time.time()
for i in range(100):
    logits, logits_len = sess.run(['logits', 'logits_len'], ort_inputs)
    t = time.time()
    print(t - s)
    s = t

pred = logits.argmax(-1)
print(encoder.merge_repeated(encoder.decode(pred[0])))
