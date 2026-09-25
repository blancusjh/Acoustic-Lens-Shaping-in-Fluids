"""Discrete shape derivative checks against independently perturbed wave solves."""

import numpy as np
import pytest

from acoustic_freeform.acoustics.helmholtz import DualAcoustics
from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface


@pytest.mark.parametrize("order", [2, 4])
def test_full_shape_derivative_matches_two_sided_wave_perturbations(order):
    cfg = DualConfig(
        radial_cells=8,
        cells_per_layer=4,
        radial_ports=4,
        side_ports_per_layer=2,
        acoustic_order=order,
        surface_elements=4,
    )
    space = DualSurface(cfg)
    rng = np.random.default_rng(52)
    q = rng.normal(size=(2, space.count)) * 2e-6
    drive = (rng.normal(size=cfg.channels) + 1j * rng.normal(size=cfg.channels)) * 0.1
    directions = np.eye(2 * space.count)[:, [0, space.count, 2 * space.count - 1]]
    wave = DualAcoustics(
        cfg, space, linear_solver="static_condensed", volume_assembly_chunk_size=31
    )
    _, analytic = wave.shape_force_jacobian(q, drive, directions, block_size=2)
    reference_wave = DualAcoustics(cfg, space, volume_assembly_chunk_size=None)
    for column, direction in enumerate(directions.T):
        for step in [1e-8, 5e-9]:
            delta = (step * direction).reshape(q.shape)
            plus = reference_wave.solve(q + delta, drive).force(np.ones(1, complex))
            minus = reference_wave.solve(q - delta, drive).force(np.ones(1, complex))
            finite = (plus - minus) / (2 * step)
            relative = np.linalg.norm(analytic[:, column] - finite) / np.linalg.norm(finite)
            assert relative < 2e-6


def test_zero_command_has_zero_acoustic_shape_derivative():
    cfg = DualConfig(
        radial_cells=4,
        cells_per_layer=2,
        radial_ports=2,
        side_ports_per_layer=1,
        acoustic_order=2,
        surface_elements=4,
    )
    space = DualSurface(cfg)
    _, derivative = DualAcoustics(cfg, space).shape_force_jacobian(
        np.zeros((2, space.count)),
        np.zeros(cfg.channels, complex),
        np.ones((2 * space.count, 1)),
    )
    np.testing.assert_array_equal(derivative, 0)
