"""Exact linear algebra and equivalence of independent wave factorizations."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from acoustic_freeform.acoustics.helmholtz import DualAcoustics
from acoustic_freeform.acoustics.linear import CondensedFactor, ReusedCondensedFactor
from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
def test_curved_wave_condensation_preserves_pressure_flux_and_load(order):
    cfg = replace(
        DualConfig(),
        acoustic_order=order,
        radial_cells=6,
        cells_per_layer=4,
        radial_ports=2,
        side_ports_per_layer=2,
        surface_elements=6,
        frequency_hz=300000,
    )
    space = DualSurface(cfg)
    rng = np.random.default_rng(416)
    q = rng.normal(scale=1e-5, size=(2, space.count))
    direct = DualAcoustics(cfg, space, local_flux_assembly=False).solve(q)
    condensed = DualAcoustics(cfg, space, "static_condensed").solve(q)
    for result in (direct, condensed):
        diagnostics = result.diagnostics(np.ones(cfg.channels, complex))
        assert json.loads(json.dumps(diagnostics))["factorized_pressure_dofs"] > 0
    np.testing.assert_allclose(condensed.solution, direct.solution, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(condensed.force_kernels, direct.force_kernels, rtol=1e-8, atol=1e-12)
    for a, b in zip(condensed.velocity, direct.velocity):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)
    assert condensed.residual < 1e-11
    blocked = DualAcoustics(cfg, space, "static_condensed").solve(q, trace_only=True)
    np.testing.assert_allclose(blocked.force_kernels, direct.force_kernels, rtol=1e-8, atol=1e-12)


def test_condensation_rejects_nonlocal_interior_blocks():
    matrix = np.eye(4)
    matrix[2, 3] = 1
    with pytest.raises(ValueError, match="inter-element"):
        CondensedFactor(csc_matrix(matrix), np.array([[2, 3]]))


def test_reused_preconditioner_solves_changed_geometry_not_frozen_wave():
    cfg = DualConfig(
        radial_cells=8,
        cells_per_layer=6,
        radial_ports=2,
        side_ports_per_layer=2,
        surface_elements=6,
        acoustic_order=4,
    )
    space = DualSurface(cfg)
    rng = np.random.default_rng(981)
    q = rng.normal(scale=1e-6, size=(2, space.count))
    drive = rng.normal(size=cfg.channels) + 1j * rng.normal(size=cfg.channels)
    wave = DualAcoustics(cfg, space, "reused_condensed")
    initial = wave.solve(q, drive)
    moved = q + rng.normal(scale=1e-7, size=q.shape)
    actual = wave.solve(moved, drive)
    reference = DualAcoustics(cfg, space).solve(moved, drive)
    assert np.linalg.norm(actual.solution - initial.solution) > 1
    assert actual.residual < 2e-12
    assert wave._reused_factor.iterations > 0
    np.testing.assert_allclose(actual.solution, reference.solution, rtol=1e-8, atol=1e-4)
    np.testing.assert_allclose(actual.force_kernels, reference.force_kernels, rtol=1e-8, atol=1e-12)


def test_failed_iterative_solve_refreshes_the_factor(monkeypatch):
    initial = csc_matrix(np.eye(4, dtype=complex))
    factor = ReusedCondensedFactor(initial, np.array([[2, 3]]))
    changed = 2 * initial
    factor.update(changed)
    monkeypatch.setattr(
        "acoustic_freeform.acoustics.linear.gmres", lambda matrix, rhs, **kwargs: (np.zeros_like(rhs), 1)
    )
    rhs = np.arange(4, dtype=complex)
    np.testing.assert_allclose(changed @ factor.solve(rhs), rhs)
    assert factor.refreshes == 1
