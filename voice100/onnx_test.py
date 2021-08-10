# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import time
import torch
from torch import nn

MELSPEC_DIM = 64
BATCH_SIZE = 5

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

def export_onnx_ttsalign():
    from voice100.models.tts import TextToAlignTextModel
    ckpt_path = 'model/ttsalign_en_conv_base-20210808.ckpt'
    model = TextToAlignTextModel.load_from_checkpoint(ckpt_path)
    model.eval()

    text_len = 100

    # Input to the model
    x = torch.randint(low=0, high=27, size=(BATCH_SIZE, text_len), requires_grad=False)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "model/ttsalign_en_conv_base.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=13,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['text'],   # the model's input names
                    output_names = ['align'], # the model's output names
                    dynamic_axes={'text' : {0 : 'batch_size', 1: 'text_len'},    # variable length axes
                                    'align' : {0 : 'batch_size', 1: 'text_len'}})

class AlignTextToAudioPredictModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, aligntext):
        hasf0, f0, logspc, codeap = self.model(aligntext)
        f0, logspc, codeap = self.model.norm.unnormalize(f0, logspc, codeap)
        f0 = torch.where(hasf0 < 0, torch.zeros(size=(1,), dtype=f0.dtype, device=f0.device), f0)
        return f0, logspc, codeap

def export_onnx_ttsaudio():
    from voice100.models.tts import AlignTextToAudioModel
    ckpt_path = 'model/aligntts_en_conv_base.ckpt'
    model = AlignTextToAudioModel.load_from_checkpoint(ckpt_path)
    model.eval()
    model = AlignTextToAudioPredictModel(model)

    aligntext_len = 100

    # Input to the model
    x = torch.randint(low=0, high=27, size=(BATCH_SIZE, aligntext_len), requires_grad=False)
    #torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "model/ttsaudio_en_conv_base.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=13,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['aligntext'],   # the model's input names
                    output_names = ['f0', 'logspc', 'codeap'], # the model's output names
                    dynamic_axes={
                        'aligntext' : {0 : 'batch_size', 1: 'text_len'},    # variable length axes
                        #'hasf0' : {0 : 'batch_size', 1: 'audio_len'},
                        'f0' : {0 : 'batch_size', 1: 'audio_len'},
                        'logspc' : {0 : 'batch_size', 1: 'audio_len'},
                        'codeap' : {0 : 'batch_size', 1: 'audio_len'},
                        })

def test():
    import onnxruntime as ort
    import numpy as np

    from voice100.datasets import AudioToCharProcessor
    from voice100.text import CharTokenizer

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
