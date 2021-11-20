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
        use_mc: bool = False,
        log_offset: float = 1e-15
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.n_fft = n_fft
        if sample_rate == 16000:
            self.mcep_dim = 24
            self.mcep_alpha = 0.410
            self.codeap_dim = 1
            if self.n_fft is None:
                self.n_fft = 512
        elif sample_rate == 22050:
            self.mcep_dim = 34
            self.mcep_alpha = 0.455
            self.codeap_dim = 2
            if self.n_fft is None:
                self.n_fft = 1024
        else:
            raise ValueError("Unsupported sample rate")
        self.use_mc = use_mc
        if use_mc:
            self.sp2mc_matrix = create_sp2mc_matrix(self.n_fft, self.mcep_dim, alpha=self.mcep_alpha)
            self.mc2sp_matrix = create_mc2sp_matrix(self.n_fft, self.mcep_dim, alpha=self.mcep_alpha)
        else:
            self.sp2mc_matrix = None
            self.mc2sp_matrix = None
        self.log_offset = log_offset

    def forward(self, waveform: torch.Tensor):
        return self.encode(waveform)

    def encode(
        self,
        waveform: torch.Tensor,
        f0_floor: float = 80.0, f0_ceil: float = 400.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        waveform = waveform.cpu().numpy().astype(np.double)
        f0, time_axis = pyworld.dio(
            waveform, self.sample_rate, f0_floor=f0_floor,
            f0_ceil=f0_ceil, frame_period=self.frame_period)
        spc = pyworld.cheaptrick(waveform, f0, time_axis, self.sample_rate, fft_size=self.n_fft)
        logspc = np.log(spc + self.log_offset)
        ap = pyworld.d4c(waveform, f0, time_axis, self.sample_rate, fft_size=self.n_fft)
        codeap = pyworld.code_aperiodicity(ap, self.sample_rate)

        if self.use_mc:
            mc = logspc @ self.sp2mc_matrix
            return (
                torch.from_numpy(f0.astype(np.float32)),
                torch.from_numpy(mc.astype(np.float32)),
                torch.from_numpy(codeap.astype(np.float32))
            )
        else:
            return (
                torch.from_numpy(f0.astype(np.float32)),
                torch.from_numpy(logspc.astype(np.float32)),
                torch.from_numpy(codeap.astype(np.float32))
            )

    def decode(
        self, f0: torch.Tensor, logspc_or_mc: torch.Tensor, codeap: torch.Tensor
    ) -> torch.Tensor:
        f0 = f0.cpu().numpy().astype(np.double, order='C')
        if self.use_mc:
            mc = logspc_or_mc.cpu().numpy().astype(np.double)
            logspc = mc @ self.mc2sp_matrix
        else:
            logspc = logspc_or_mc.cpu().numpy().astype(np.double)
        codeap = codeap.cpu().numpy().astype(np.double, order='C')
        spc = np.maximum(np.exp(logspc) - self.log_offset, 0).copy(order='C')
        ap = pyworld.decode_aperiodicity(codeap, self.sample_rate, self.n_fft)
        waveform = pyworld.synthesize(f0, spc, ap, self.sample_rate, frame_period=self.frame_period)
        return waveform


def create_sp2mc_matrix(fftlen: int, order: int, alpha: float) -> np.ndarray:
    """PySPTK compatible mel-cepstrum transform matrix
    https://pysptk.readthedocs.io/en/latest/_modules/pysptk/conversion.html#sp2mc"""
    logsp = np.eye(fftlen // 2 + 1, dtype=np.float32)
    c = np.fft.irfft(logsp)
    c[:, 0] /= 2.0
    mc = freqt(c, order, alpha)
    return mc


def create_mc2sp_matrix(fftlen: int, order: int, alpha: float) -> np.ndarray:
    """PySPTK compatible mel-cepstrum invert transform matrix
    https://pysptk.readthedocs.io/en/latest/_modules/pysptk/conversion.html#mc2sp"""
    c = np.eye(order + 1, dtype=np.float32)
    c = freqt(c, fftlen // 2, -alpha)
    c[:, 0] *= 2.0
    c = np.concatenate([c[:, :], c[:, :0:-1]], axis=1)
    logsp = np.fft.rfft(c).real
    return logsp


def freqt(ceps: np.ndarray, order=25, alpha=0.0) -> np.ndarray:
    """PySPTK compatible frequency transform
    https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.freqt.html#pysptk.sptk.freqt
    """
    c = np.zeros([ceps.shape[0], order + 1])
    for i in range(ceps.shape[1]):
        d = alpha * c
        for j in range(c.shape[1]):
            if j == 0:
                d[:, j] += ceps[:, ceps.shape[1] - 1 - i]
            elif j == 1:
                d[:, j] += (1 - alpha ** 2) * c[:, j - 1]
            else:
                d[:, j] += c[:, j - 1] - alpha * d[:, j - 1]
        c = d
    return c
