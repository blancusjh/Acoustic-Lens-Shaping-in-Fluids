"""Internal derivative checks, separate from independent analytic model limits."""

import numpy as np

from acoustic_freeform.dual.acoustics import DualAcoustics
from acoustic_freeform.dual.config import DualConfig
from acoustic_freeform.dual.surface import DualSurface


def test_source_jacobian_matches_fresh_field_directional_difference():
    cfg = DualConfig(
        radial_cells=6,
        cells_per_layer=6,
        surface_elements=6,
        radial_ports=2,
        side_ports_per_layer=2,
    )
    space = DualSurface(cfg)
    wave = DualAcoustics(cfg, space)
    rng = np.random.default_rng(20260922)
    q = rng.normal(scale=1e-6, size=(2, space.count))
    drive = 0.03 * (rng.normal(size=cfg.channels) + 1j * rng.normal(size=cfg.channels))
    direction = rng.normal(size=cfg.channels) + 1j * rng.normal(size=cfg.channels)
    response = wave.solve(q)
    blocked = wave.solve(q, trace_only=True, source_block_size=3)
    assert blocked.solution.shape == (0, cfg.channels)
    np.testing.assert_allclose(blocked.force_kernels, response.force_kernels, rtol=1e-9, atol=1e-14)
    direct = wave.solve(q, drive)
    np.testing.assert_allclose(
        direct.solution[:, 0], response.solution @ drive, rtol=1e-10, atol=1e-6
    )
    for face in range(2):
        np.testing.assert_allclose(
            blocked.velocity[face], response.velocity[face], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            blocked.pressure[face], response.pressure[face], rtol=1e-9, atol=1e-5
        )
        np.testing.assert_allclose(
            direct.velocity[face][:, 0], response.velocity[face] @ drive, rtol=1e-9, atol=1e-10
        )
    analytic = response.force_jacobian(drive) @ np.r_[direction.real, direction.imag]
    step = 1e-5
    plus = wave.solve(q, drive + step * direction).force(np.ones(1))
    minus = wave.solve(q, drive - step * direction).force(np.ones(1))
    finite = (plus - minus) / (2 * step)
    np.testing.assert_allclose(analytic, finite, rtol=1e-7, atol=1e-12)
