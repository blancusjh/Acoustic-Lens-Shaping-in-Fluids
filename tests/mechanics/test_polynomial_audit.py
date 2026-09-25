import numpy as np

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface


def test_annular_extremum_against_independent_quartic():
    space = DualSurface(DualConfig(surface_elements=8))
    radius = space.config.radius_m
    x = space.r / radius
    amplitude = 1e-6
    # This quartic has zero radial volume, a pinned rim and a regular axis.
    height = amplitude * (1 - 4 * x**2 + 3 * x**4)
    coeff = np.linalg.solve(space.mass, space.b.T @ (space.weights * height))
    full = space.polynomial_maximum(coeff)
    outer = space.polynomial_maximum(coeff, lower_m=0.5 * radius)
    np.testing.assert_allclose(full["max_abs_m"], amplitude, rtol=1e-10)
    np.testing.assert_allclose(outer["max_abs_m"], amplitude / 3, rtol=1e-10)
    np.testing.assert_allclose(outer["radius_m"], radius * np.sqrt(2 / 3), rtol=1e-9)
