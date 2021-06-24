# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import torch
from torch import nn
import random

__all__ = [
    'SpectrogramAugumentation',
    'BatchSpectrogramAugumentation'
]

class SpectrogramAugumentation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, audio):
        assert len(audio.shape) == 2
        if random.random() < 0.2:
            pass # audio = self.timestretch(audio)
        if random.random() < 0.2:
            audio = self.pitchshift(audio)
        if random.random() < 0.2:
            audio = self.timemask(audio)
        if random.random() < 0.2:
            audio = self.freqmask(audio)
        if random.random() < 0.2:
            audio = self.mixnoise(audio)
        return audio        

    def timestretch(self, audio):
        rate = 1.0 + random.random() * 0.3
        i = (torch.arange(int(audio.shape[0] * rate)) / rate).int()
        return torch.index_select(audio, 0, i)

    def pitchshift(self, audio):
        rate = 1.0 + random.random() * 0.2
        i = rate * torch.arange(audio.shape[1])
        i = torch.clamp(i.int(), 0, audio.shape[1] - 1)
        return torch.index_select(audio, 1, i)

    def timemask(self, audio):
        audio = audio.clone()
        n = random.randint(1, 3)
        for i in range(n):
            t = random.randrange(0, audio.shape[0])
            hw = random.randint(1, 20)
            s = int(t - hw)
            e = int(t + hw)
            audio[s:e, :] = -20.0
        return audio

    def freqmask(self, audio):
        audio = audio.clone()
        t = random.randrange(0, audio.shape[1])
        hw = random.randint(1, 3)
        s = int(t - hw)
        e = int(t + hw)
        audio[:, s:e] = -20.0
        return audio

    def mixnoise(self, audio):
        low = -5.0 + 5.0 * random.random()
        high = -5.0 + 5.0 * random.random()
        std = 5.0 * random.random()
        scale = torch.linspace(low, high, 64)[None, :]
        noise = torch.rand(audio.shape) * std + scale
        return torch.log(torch.exp(audio) + torch.exp(noise))

class BatchSpectrogramAugumentation(nn.Module):

    def __init__(self, do_timestretch=True):
        super().__init__()
        self.do_timestretch = do_timestretch

    def forward(self, audio, audio_len):
        assert len(audio.shape) == 3
        AUGUMENT_RATE = 0.2

        if self.do_timestretch and random.random() < AUGUMENT_RATE:
            audio, audio_len = self.timestretch(audio, audio_len)
        if random.random() < AUGUMENT_RATE:
            audio = self.pitchshift(audio)
        if random.random() < AUGUMENT_RATE:
            audio = self.timemask(audio)
        if random.random() < AUGUMENT_RATE:
            audio = self.freqmask(audio)
        if random.random() < AUGUMENT_RATE:
            audio = self.mixnoise(audio)
        if random.random() < AUGUMENT_RATE:
            audio = self.mixaudio(audio, audio_len)

        return audio, audio_len

    def timestretch(self, audio, audio_len):
        rate = 1.0 + random.random() * 0.3
        i = (torch.arange(int(audio.shape[1] * rate), device=audio.device) / rate).int()
        audio = torch.index_select(audio, 1, i)
        audio_len = (audio_len * rate).int()
        return audio, audio_len

    def pitchshift(self, audio):
        rate = 1.0 + random.random() * 0.2
        i = (torch.arange(audio.shape[2], device=audio.device) * rate).int()
        i = torch.clamp(i, 0, audio.shape[2] - 1)
        return torch.index_select(audio, 2, i)

    def timemask(self, audio):
        audio = audio.clone()
        n = random.randint(1, 3)
        for i in range(n):
            t = random.randrange(0, audio.shape[1])
            hw = random.randint(1, 20)
            s = int(t - hw)
            e = int(t + hw)
            audio[:, s:e, :] = -20.0
        return audio

    def freqmask(self, audio):
        audio = audio.clone()
        t = random.randrange(0, audio.shape[2])
        hw = random.randint(1, 3)
        s = int(t - hw)
        e = int(t + hw)
        audio[:, :, s:e] = -20.0
        return audio

    def mixnoise(self, audio):
        low = -5.0 + 5.0 * random.random()
        high = -5.0 + 5.0 * random.random()
        std = 5.0 * random.random()
        scale = torch.linspace(low, high, 64, device=audio.device)[None, :]
        noise = torch.rand(audio.shape, device=audio.device) * std + scale
        return torch.log(torch.exp(audio) + torch.exp(noise))

    def mixaudio(self, audio, audio_len):
        audio_mask = (torch.arange(audio.shape[1], device=audio.device)[None, :, None] < audio_len[:, None, None]).float()
        x = torch.exp(audio) * audio_mask
        y = torch.cat([x[1:], x[:1]])
        return torch.log(0.9 * x + 0.1 * y) * audio_mask
