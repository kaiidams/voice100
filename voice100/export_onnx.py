# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import os
import torch
from torch import nn

MELSPEC_DIM = 64
BATCH_SIZE = 1


def export_onnx_asr(args):
    from .models import AudioToAlignText

    model = AudioToAlignText.load_from_checkpoint(args.ckpt_path)
    model.eval()

    audio = torch.rand(size=[1, 100, MELSPEC_DIM], dtype=torch.float32)
    audio_len = torch.tensor([audio.shape[1]], dtype=torch.int64)

    torch.onnx.export(
        model,
        (audio, audio_len),
        args.output,
        export_params=True,
        verbose=args.verbose,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["audio", "audio_len"],
        output_names=["logits", "logits_len"],
        dynamic_axes={
            "audio": {0: "batch_size", 1: "audio_len"},
            "audio_len": {0: "batch_size"},
            "logits": {0: "batch_size", 1: "logits_len"},
            "logits_len": {0: "batch_size"},
        },
    )


class TextToAlignTextPredict(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, text, text_len):
        return self.model.predict(text, text_len)


def export_onnx_align(args):
    from voice100.models import TextToAlignText

    model = TextToAlignText.load_from_checkpoint(args.ckpt_path)
    vocab_size = model.hparams["vocab_size"]
    model = TextToAlignTextPredict(model)
    model.eval()

    batch_size = 1
    text_len = 100
    text = torch.randint(low=0, high=vocab_size, size=(batch_size, text_len), requires_grad=False)
    text_len = torch.tensor([text_len], dtype=torch.int64, requires_grad=False)

    torch.onnx.export(
        model,
        (text, text_len),
        args.output,
        export_params=True,
        verbose=args.verbose,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["text", "text_len"],
        output_names=["align", "align_len"],
        dynamic_axes={
            "text": {0: "batch_size", 1: "text_len"},
            "text_len": {0: "batch_size"},
            "align": {0: "batch_size", 1: "text_len"},
            "aligntext_len": {0: "batch_size"},
        },
    )


class AlignTextToAudioPredict(nn.Module):
    def __init__(self, model) -> None:
        from voice100.vocoder import create_mc2sp_matrix
        super().__init__()
        self.model = model
        if model.logspc_size == 25:
            self.mc2sp_matrix = torch.from_numpy(create_mc2sp_matrix(512, 24, 0.410)).float()
        else:
            self.mc2sp_matrix = None

    def forward(self, aligntext, aligntext_len):
        f0, logspc_or_mcep, codeap = self.model.predict(aligntext, aligntext_len)
        if self.mc2sp_matrix is not None:
            logspc = logspc_or_mcep @ self.mc2sp_matrix
        else:
            logspc = logspc_or_mcep
        return f0, logspc, codeap


def export_onnx_tts(args):
    from voice100.models import AlignTextToAudio

    model = AlignTextToAudio.load_from_checkpoint(args.ckpt_path)
    vocab_size = model.hparams["vocab_size"]
    model = AlignTextToAudioPredict(model)
    model.eval()

    aligntext_len = 100
    aligntext = torch.randint(
        low=0, high=vocab_size, size=(BATCH_SIZE, aligntext_len), requires_grad=False
    )
    aligntext_len = torch.tensor([aligntext_len], dtype=torch.int64)

    torch.onnx.export(
        model,
        (aligntext, aligntext_len),
        args.output,
        export_params=True,
        verbose=args.verbose,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["aligntext", "aligntext_len"],
        output_names=["f0", "logspc", "codeap"],
        dynamic_axes={
            "aligntext": {0: "batch_size", 1: "aligntext_len"},
            "aligntext_len": {0: "batch_size"},
            "f0": {0: "batch_size", 1: "audio_len"},
            "logspc": {0: "batch_size", 1: "audio_len"},
            "codeap": {0: "batch_size", 1: "audio_len"},
        },
    )


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--model", type=str, choices=["asr", "align", "tts", "tts_mt"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--opset_version", type=int, default=13)

    args = parser.parse_args()
    if args.model is None:
        basename = os.path.basename(args.ckpt_path)
        args.model = basename.partition('_')[0]

    if args.output is None:
        basename = os.path.basename(args.ckpt_path)
        args.output, ext = os.path.splitext(basename)
        if ext != ".ckpt":
            raise ValueError(ext)
        args.output += ".onnx"

    if args.model == "asr":
        export_onnx_asr(args)
    elif args.model == "align":
        export_onnx_align(args)
    elif args.model == "tts":
        export_onnx_tts(args)
    else:
        raise ValueError()


if __name__ == "__main__":
    cli_main()
