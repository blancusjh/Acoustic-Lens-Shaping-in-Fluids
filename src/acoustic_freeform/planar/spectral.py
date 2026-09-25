"""Fourier-series conventions: centered real-space arrays, normalized coefficients."""

from dataclasses import dataclass

import numpy as np

from .config import Grid


@dataclass
class SpectralGrid:
    definition: Grid

    def __post_init__(self):
        d = self.definition
        self.x = (np.arange(d.nx) - d.nx // 2) * d.lx_m / d.nx
        self.y = (np.arange(d.ny) - d.ny // 2) * d.ly_m / d.ny
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.kx = 2 * np.pi * np.fft.fftfreq(d.nx, d.lx_m / d.nx)
        self.ky = 2 * np.pi * np.fft.fftfreq(d.ny, d.ly_m / d.ny)
        self.kxx, self.kyy = np.meshgrid(self.kx, self.ky)
        self.k = np.hypot(self.kxx, self.kyy)
        self.area = d.lx_m * d.ly_m
        self.count = d.nx * d.ny

    def forward(self, values: np.ndarray) -> np.ndarray:
        return np.fft.fft2(np.fft.ifftshift(values, axes=(-2, -1))) / self.count

    def inverse(self, coefficients: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(np.fft.ifft2(coefficients), axes=(-2, -1)) * self.count

    def gradient(self, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.inverse(1j * self.kxx * coefficients).real,
            self.inverse(1j * self.kyy * coefficients).real,
        )
