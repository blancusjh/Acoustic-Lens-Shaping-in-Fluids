"""Compare root algorithms, without interpreting either as physical dynamics."""

import numpy as np
import pytest

from acoustic_freeform.acoustics.helmholtz import DualAcoustics
from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.forward.equilibrium import coupled_equilibrium
from acoustic_freeform.mechanics.surface import DualSurface


@pytest.mark.parametrize("linear_solver", ["full", "reused_condensed"])
def test_matrix_free_root_matches_dense_on_shared_wave_model(linear_solver):
    cfg = DualConfig(
        radial_cells=8,
        cells_per_layer=6,
        surface_elements=6,
        radial_ports=2,
        side_ports_per_layer=2,
        frequency_hz=200000,
    )
    space = DualSurface(cfg)
    wave = DualAcoustics(cfg, space)
    rng = np.random.default_rng(20260923)
    drive = 0.04 * (rng.normal(size=cfg.channels) + 1j * rng.normal(size=cfg.channels))
    initial = np.zeros((2, space.count))
    dense, _ = coupled_equilibrium(wave, space, drive, initial)
    checked, balanced = coupled_equilibrium(wave, space, drive, dense)
    assert balanced["initial_seed_accepted"]
    assert balanced["wave_solves"] == 1
    np.testing.assert_array_equal(checked, dense)
    history = []
    wave = DualAcoustics(cfg, space, linear_solver)
    matrix_free, info = coupled_equilibrium(
        wave, space, drive, initial, method="krylov", progress=history.append
    )
    for dense_face, matrix_free_face in zip(dense, matrix_free):
        np.testing.assert_allclose(
            space.evaluate(dense_face, space.r),
            space.evaluate(matrix_free_face, space.r),
            atol=1e-11,
            rtol=1e-5,
        )
    assert info["compliance_residual_max_m"] < 1e-11
    assert len(history) == info["wave_solves"]
    np.testing.assert_allclose(history[-1]["trial_coefficients_m"], matrix_free)
    assert history[-1]["trial_compliance_residual_max_m"] < 1e-11
