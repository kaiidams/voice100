import time
import torch
import onnxruntime as ort
import numpy as np

from voice100.datasets import AudioToCharProcessor
from voice100.text import CharTokenizer

MELSPEC_DIM = 64

def export_onnx(ckpt_path, output_file):
    from .models.align import AudioAlignCTC
    model = AudioAlignCTC.load_from_checkpoint(ckpt_path)
    audio = torch.rand(size=[1, 100, MELSPEC_DIM], dtype=torch.float32)
    model.eval()

    torch.onnx.export(
        model, audio,
        output_file,
        export_params=True,
        opset_version=13,
        verbose=True,
        do_constant_folding=True,
        input_names = ['audio'],
        output_names = ['logits'],
        dynamic_axes={'audio': {0: 'batch_size', 1: 'audio_len'},
                        'logits': {0: 'batch_size', 1: 'logits_len'}})

def export_onnx_asr(ckpt_path, output_file):
    from .models.align import AudioToCharCTC
    model = AudioToCharCTC.load_from_checkpoint(ckpt_path)
    audio = torch.rand(size=[1, 100, MELSPEC_DIM], dtype=torch.float32)
    model.eval()

    torch.onnx.export(
        model, audio,
        output_file,
        export_params=True,
        opset_version=13,
        verbose=True,
        do_constant_folding=True,
        input_names = ['audio'],
        output_names = ['logits'],
        dynamic_axes={'audio': {0: 'batch_size', 1: 'audio_len'},
                        'logits': {0: 'batch_size', 1: 'logits_len'}})


sess = ort.InferenceSession('model/onnx/stt_en_conv_base_ctc.onnx')
processor = AudioToCharProcessor(phonemizer='en')

import librosa
waveform, sr = librosa.load('test.flac', 16000)
waveform = torch.from_numpy(waveform)

encoder = processor._encoder
audio = torch.log(processor.transform(waveform).T + processor.log_offset)

ort_inputs = {
    'audio': audio[None, :, :].numpy(), #np.zeros(shape=[1, 123, 64]).astype(dtype=np.float32),
}

if False:
    s = time.time()
    for i in range(100):
        logits, logits_len = sess.run('logits', ort_inputs)
        t = time.time()
        print(t - s)
        s = t

logits, = sess.run(['logits'], ort_inputs)
pred = logits.argmax(-1)
print(encoder.merge_repeated(encoder.decode(pred[0])))
