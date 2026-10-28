"""Monostatic OFDM radar on a communication waveform (Integrated Sensing And Communication)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C = 299_792_458.0


@dataclass
class OFDMConfig:
    fc_hz: float = 10e9            # FR3 (7-24 GHz) carrier
    scs_hz: float = 120e3
    n_subcarriers: int = 1024
    n_symbols: int = 256
    cp_ratio: float = 0.07         # NR normal CP ~ 7 %

    @property
    def bandwidth_hz(self) -> float:
        return self.n_subcarriers * self.scs_hz

    @property
    def symbol_s(self) -> float:
        return (1 + self.cp_ratio) / self.scs_hz

    @property
    def wavelength_m(self) -> float:
        return C / self.fc_hz

    @property
    def range_resolution_m(self) -> float:
        return C / (2 * self.bandwidth_hz)

    @property
    def max_range_m(self) -> float:
        """Unambiguous range of the subcarrier grid; the CP limits ISI-free range further."""
        return C / (2 * self.scs_hz)

    @property
    def cp_limited_range_m(self) -> float:
        return C * self.cp_ratio / self.scs_hz / 2

    @property
    def velocity_resolution_mps(self) -> float:
        return self.wavelength_m / (2 * self.n_symbols * self.symbol_s)

    @property
    def max_velocity_mps(self) -> float:
        return self.wavelength_m / (4 * self.symbol_s)


@dataclass
class Target:
    range_m: float
    velocity_mps: float            # positive = approaching
    snr_db: float = 0.0            # per-resource-element echo SNR


@dataclass
class Detection:
    range_m: float
    velocity_mps: float
    power_db: float


@dataclass
class RangeDopplerMap:
    power: np.ndarray              # (range bins, doppler bins), linear
    range_axis: np.ndarray
    velocity_axis: np.ndarray
    detections: list = field(default_factory=list)


def qpsk_grid(cfg: OFDMConfig, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    bits = rng.integers(0, 2, (2, cfg.n_subcarriers, cfg.n_symbols))
    return ((1 - 2 * bits[0]) + 1j * (1 - 2 * bits[1])) / np.sqrt(2)


def simulate_echo(cfg: OFDMConfig, tx: np.ndarray, targets: list, rng=None) -> np.ndarray:
    """Received frequency-domain grid after CP removal (noise power = 1 per RE)."""
    rng = np.random.default_rng(rng)
    k = np.arange(cfg.n_subcarriers)[:, None]
    m = np.arange(cfg.n_symbols)[None, :]
    rx = np.zeros_like(tx)
    for t in targets:
        tau = 2 * t.range_m / C
        fd = 2 * t.velocity_mps / cfg.wavelength_m
        amp = 10 ** (t.snr_db / 20) * np.exp(2j * np.pi * rng.random())
        rx += amp * tx * np.exp(-2j * np.pi * k * cfg.scs_hz * tau) * np.exp(2j * np.pi * m * cfg.symbol_s * fd)
    noise = (rng.standard_normal(tx.shape) + 1j * rng.standard_normal(tx.shape)) / np.sqrt(2)
    return rx + noise


def range_doppler(cfg: OFDMConfig, tx: np.ndarray, rx: np.ndarray, window: bool = True) -> RangeDopplerMap:
    """Element-wise division by the known symbols, then IFFT (range) and FFT (Doppler)."""
    D = rx / tx
    if window:
        D = D * np.hanning(cfg.n_subcarriers)[:, None] * np.hanning(cfg.n_symbols)[None, :]
    rd = np.fft.ifft(D, axis=0)
    rd = np.fft.fftshift(np.fft.fft(rd, axis=1), axes=1)
    power = np.abs(rd) ** 2
    range_axis = np.arange(cfg.n_subcarriers) * cfg.range_resolution_m
    fd_axis = (np.arange(cfg.n_symbols) - cfg.n_symbols // 2) / (cfg.n_symbols * cfg.symbol_s)
    return RangeDopplerMap(power, range_axis, fd_axis * cfg.wavelength_m / 2)


def ca_cfar_2d(power: np.ndarray, guard=(2, 2), train=(8, 4), pfa: float = 1e-6) -> np.ndarray:
    """Cell-averaging CFAR with circular boundaries; returns a boolean detection mask."""
    gr, gd = guard
    tr, td = train
    kernel = np.zeros(power.shape)
    rows = np.arange(-(gr + tr), gr + tr + 1)
    cols = np.arange(-(gd + td), gd + td + 1)
    kernel[np.ix_(rows % power.shape[0], cols % power.shape[1])] = 1
    kernel[np.ix_(np.arange(-gr, gr + 1) % power.shape[0], np.arange(-gd, gd + 1) % power.shape[1])] = 0
    n_train = kernel.sum()
    noise = np.real(np.fft.ifft2(np.fft.fft2(power) * np.fft.fft2(kernel))) / n_train
    alpha = n_train * (pfa ** (-1 / n_train) - 1)
    return power > alpha * noise


def _local_maxima(power: np.ndarray) -> np.ndarray:
    peak = np.ones(power.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dd in (-1, 0, 1):
            if dr or dd:
                peak &= power >= np.roll(np.roll(power, dr, axis=0), dd, axis=1)
    return peak


def detect(cfg: OFDMConfig, tx: np.ndarray, rx: np.ndarray, pfa: float = 1e-8) -> RangeDopplerMap:
    rdm = range_doppler(cfg, tx, rx)
    mask = ca_cfar_2d(rdm.power, pfa=pfa) & _local_maxima(rdm.power)
    rows, cols = np.nonzero(mask)
    order = np.argsort(rdm.power[rows, cols])[::-1]
    rdm.detections = [
        Detection(rdm.range_axis[r], rdm.velocity_axis[c], 10 * np.log10(rdm.power[r, c]))
        for r, c in zip(rows[order], cols[order])
    ]
    return rdm
