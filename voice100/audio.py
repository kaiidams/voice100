# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from typing import Tuple
import math
import torch
from torch import nn
import random

__all__ = [
    "BatchSpectrogramAugumentation"
]

SPECTROGRAM_AUGUMENT_RATE = 0.2
LOG_OFFSET = 1e-6


class BatchSpectrogramAugumentation(nn.Module):
    """Augment mel-spectrogram data
    """

    def __init__(self, do_timestretch=True, log_offset=LOG_OFFSET):
        super().__init__()
        self.do_timestretch = do_timestretch
        self.log_offset = log_offset
        self.blank_audio = math.log(log_offset)

    @torch.no_grad()
    def forward(
        self, audio: torch.Tensor, audio_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(audio.shape) == 3
        assert audio.dtype == torch.float32

        if self.do_timestretch and random.random() < SPECTROGRAM_AUGUMENT_RATE:
            audio, audio_len = self.timestretch(audio, audio_len)
        if random.random() < SPECTROGRAM_AUGUMENT_RATE:
            audio = self.pitchshift(audio)
        if random.random() < SPECTROGRAM_AUGUMENT_RATE:
            audio = self.ampshift(audio)
        if random.random() < SPECTROGRAM_AUGUMENT_RATE:
            audio = self.timemask(audio)
        if random.random() < SPECTROGRAM_AUGUMENT_RATE:
            audio = self.freqmask(audio)
        if random.random() < SPECTROGRAM_AUGUMENT_RATE:
            audio = self.mixnoise(audio)
        if random.random() < SPECTROGRAM_AUGUMENT_RATE:
            audio = self.mixaudio(audio, audio_len)
        else:
            audio = self.maskaudio(audio, audio_len)
        return audio, audio_len

    def timestretch(self, audio, audio_len):
        rate = random.randrange(50, 150)
        max_audio_len = audio.shape[1] * rate // 100
        audio_len = torch.div(audio_len * rate, 100, rounding_mode='trunc')
        i = torch.div(torch.arange(max_audio_len, device=audio.device) * 100, rate, rounding_mode='trunc')
        audio = torch.index_select(audio, 1, i)
        return audio, audio_len

    def pitchshift(self, audio):
        rate = 1.0 + random.random() * 0.2
        i = (torch.arange(audio.shape[2], device=audio.device) * rate).int()
        i = torch.clamp(i, 0, audio.shape[2] - 1)
        return torch.index_select(audio, 2, i)

    def ampshift(self, audio):
        rate = 1.0 + random.random() * 3.0
        return audio - rate

    def timemask(self, audio):
        audio = audio.clone()
        n = random.randint(1, 3)
        for i in range(n):
            t = random.randrange(0, audio.shape[1])
            hw = random.randint(1, 3)
            s = int(t - hw)
            e = int(t + hw)
            a = random.uniform(-self.blank_audio, -5)
            audio[:, s:e, :] = a
        return audio

    def freqmask(self, audio):
        audio = audio.clone()
        t = random.randrange(0, audio.shape[2])
        hw = random.randint(1, 10)
        s = int(t - hw)
        e = int(t + hw)
        a = random.uniform(-self.blank_audio, -5)
        audio[:, :, s:e] = a
        return audio

    def mixnoise(self, audio):
        low = -5.0 + 5.0 * random.random()
        high = -5.0 + 5.0 * random.random()
        std = 5.0 * random.random()
        scale = torch.linspace(low, high, 64, device=audio.device)[None, :]
        noise = torch.rand(audio.shape, device=audio.device) * std + scale
        return torch.log(torch.clamp(torch.exp(audio) + torch.exp(noise), min=self.log_offset))

    def mixaudio(self, audio, audio_len):
        audio_mask = (torch.arange(audio.shape[1], device=audio.device)[None, :, None] < audio_len[:, None, None]).float()
        x = torch.exp(audio) * audio_mask
        y = torch.cat([x[1:], x[:1]], axis=0)
        return torch.log(torch.clamp((0.9 * x + 0.1 * y) * audio_mask, min=self.log_offset))

    def maskaudio(self, audio, audio_len):
        audio_mask = (torch.arange(audio.shape[1], device=audio.device)[None, :, None] < audio_len[:, None, None]).float()
        return torch.log(torch.clamp(torch.exp(audio) * audio_mask, min=self.log_offset))
