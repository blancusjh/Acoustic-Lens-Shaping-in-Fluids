"""Coherent square-patch pressure array; exact Fourier coefficients of each aperture.

The input is outgoing acoustic pressure on a source plane, not electrical watts
or total piezo face velocity. Returned sound is absorbed at that source plane.
An electromechanical transducer and cavity-return model are outside this baseline.
"""

import numpy as np

from .config import Array, Beam
from .spectral import SpectralGrid


class PressureArray:
    def __init__(self, definition: Array, grid: SpectralGrid, sound_speed_m_s: float):
        self.definition = definition
        self.grid = grid
        self.x = (np.arange(definition.nx) - (definition.nx - 1) / 2) * definition.pitch_m
        self.y = (np.arange(definition.ny) - (definition.ny - 1) / 2) * definition.pitch_m
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.wave_number = 2 * np.pi * definition.frequency_hz / sound_speed_m_s
        self._ex = np.exp(-1j * np.outer(self.x, grid.kx))
        self._ey = np.exp(-1j * np.outer(grid.ky, self.y))

    def beam_weights(self, beam: Beam) -> np.ndarray:
        weights = np.zeros(self.xx.shape, dtype=complex)
        for focus in beam.foci:
            distance = np.sqrt(
                (self.xx - focus.x_m) ** 2
                + (self.yy - focus.y_m) ** 2
                + self.definition.source_distance_m**2
            )
            weights += focus.weight * np.exp(-1j * self.wave_number * distance)
        peak = np.abs(weights).max()
        if peak == 0:
            raise ValueError("Focal contributions cancel every array element.")
        return weights / peak

    def spectrum(self, weights: np.ndarray) -> np.ndarray:
        if weights.shape != self.xx.shape or not np.isfinite(weights).all():
            raise ValueError("Invalid complex element weights.")
        d, g = self.definition, self.grid
        element_shape = np.outer(
            np.sinc(g.ky * d.element_width_m / (2 * np.pi)),
            np.sinc(g.kx * d.element_width_m / (2 * np.pi)),
        )
        return (
            d.pressure_amplitude_pa
            * d.element_width_m**2
            / g.area
            * element_shape
            * (self._ey @ weights @ self._ex)
        )
