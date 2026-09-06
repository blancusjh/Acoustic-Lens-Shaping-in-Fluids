"""Independent, unbounded-half-space Rayleigh integral for incident pressure.

This is a verification calculation, not the transient simulation's acoustic
solver. It integrates each square aperture by tensor Gauss-Legendre quadrature.
For outgoing Dirichlet data p(x,y,0), the half-space kernel is -2*dG/dz,
G=exp(ikr)/(4*pi*r): K=z*exp(ikr)*(1-ikr)/(2*pi*r^3).
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

from .array import PressureArray


def rayleigh_incident_pressure(
    array: PressureArray, weights: np.ndarray, xy_m: np.ndarray, order: int = 8
) -> np.ndarray:
    nodes, quadrature = leggauss(order)
    half = array.definition.element_width_m / 2
    x = array.xx[..., None, None] + half * nodes[None, None, None, :]
    y = array.yy[..., None, None] + half * nodes[None, None, :, None]
    area_weight = half**2 * quadrature[:, None] * quadrature[None, :]
    z, k = array.definition.source_distance_m, array.wave_number
    values = []
    for px, py in xy_m:
        distance = np.sqrt((px - x) ** 2 + (py - y) ** 2 + z * z)
        kernel = z * np.exp(1j * k * distance) * (1 - 1j * k * distance) / (2 * np.pi * distance**3)
        values.append(
            array.definition.pressure_amplitude_pa
            * np.sum(weights[..., None, None] * area_weight * kernel)
        )
    return np.asarray(values)
