# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np
import pyworld
import torch
from torch import nn
from typing import Tuple

__all__ = [
    "WORLDVocoder",
    ]
    
class WORLDVocoder(nn.Module):

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_period: int = 10.0,
        n_fft: int = None,
        log_offset: float = 1e-15
        ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.n_fft = n_fft
        if sample_rate == 16000:
            self.codeap_dim = 1
            if self.n_fft is None: self.n_fft = 512
        elif sample_rate == 22050:
            self.codeap_dim = 2
            if self.n_fft is None: self.n_fft = 1024
        else:
            raise ValueError("Unsupported sample rate")
        self.log_offset = log_offset

    def forward(self, waveform: torch.Tensor):
        return self.encode(waveform)

    def encode(
        self,
        waveform: torch.Tensor,
        f0_floor: float = 80.0, f0_ceil: float = 400.0
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        waveform = waveform.numpy().astype(np.double)
        f0, time_axis = pyworld.dio(waveform, self.sample_rate, f0_floor=f0_floor,
            f0_ceil=f0_ceil, frame_period=self.frame_period)
        spc = pyworld.cheaptrick(waveform, f0, time_axis, self.sample_rate, fft_size=self.n_fft)
        logspc = np.log(spc + self.log_offset)
        ap = pyworld.d4c(waveform, f0, time_axis, self.sample_rate, fft_size=self.n_fft)
        codeap = pyworld.code_aperiodicity(ap, self.sample_rate)

        return (
            torch.from_numpy(f0.astype(np.float32)),
            torch.from_numpy(logspc.astype(np.float32)),
            torch.from_numpy(codeap.astype(np.float32))
        )

    def decode(
        self, f0: torch.Tensor, logspc: torch.Tensor, codeap: torch.Tensor
        ) -> torch.Tensor:
        f0 = f0.numpy().astype(np.double)
        logspc = logspc.numpy().astype(np.double)
        codeap = codeap.numpy().astype(np.double)
        spc = np.maximum(np.exp(logspc) - self.log_offset, 0)
        ap = pyworld.decode_aperiodicity(codeap, self.sample_rate, self.n_fft)
        waveform = pyworld.synthesize(f0, spc, ap, self.sample_rate, frame_period=self.frame_period)
        return waveform
