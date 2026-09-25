"""Check hydrostatics against volume gravitational energy, not a load formula."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface


@pytest.mark.parametrize(
    "densities", [(1200.0, 1100.0, 1000.0), (1000.0, 1100.0, 1200.0), (1100.0, 1100.0, 1100.0)]
)
def test_gravity_against_three_fluid_volume_energy(densities):
    cfg = DualConfig(surface_elements=8, density_kg_m3=densities)
    space = DualSurface(cfg)
    zero_g = DualSurface(replace(cfg, gravity_m_s2=0.0))
    rng = np.random.default_rng(913)
    q = rng.normal(size=(2, space.count)) * 2e-5
    direction = rng.normal(size=q.shape)
    # Independent quadrature over every spline interval and vertical fluid column.
    edges = np.unique(space.raw.t) * cfg.radius_m
    x, w = leggauss(18)
    r = (edges[:-1, None] + np.diff(edges)[:, None] * (x + 1) / 2).ravel()
    area = 2 * np.pi * r * (np.diff(edges)[:, None] * w / 2).ravel()
    zx, zw = leggauss(4)

    def volume_energy(coeff):
        heights = np.array([space.evaluate(face, r) for face in coeff])
        bounds = np.broadcast_to(np.array(cfg.levels_m)[:, None], (4, len(r))).copy()
        bounds[1:3] += heights
        total = 0.0
        for j, density in enumerate(densities):
            depth = bounds[j + 1] - bounds[j]
            z = bounds[j, :, None] + depth[:, None] * (zx + 1) / 2
            total += density * cfg.gravity_m_s2 * np.dot(area, z @ zw * depth / 2)
        return total

    grad, hess, energy = space.mechanics(q)
    grad0, hess0, energy0 = zero_g.mechanics(q)
    expected = volume_energy(q) - volume_energy(np.zeros_like(q))
    np.testing.assert_allclose(energy - energy0, expected, rtol=2e-9, atol=2e-18)
    step = 1e-7
    derivative = (volume_energy(q + step * direction) - volume_energy(q - step * direction)) / (
        2 * step
    )
    np.testing.assert_allclose(
        (grad - grad0) @ direction.ravel(), derivative, rtol=2e-7, atol=2e-12
    )
    curvature = direction.ravel() @ (hess - hess0) @ direction.ravel()
    if densities[0] > densities[1]:
        assert curvature > 0  # gravity restores a heavy-below configuration
    elif densities[0] < densities[1]:
        assert curvature < 0  # gravity destabilizes a heavy-above configuration
    else:
        assert curvature == 0
