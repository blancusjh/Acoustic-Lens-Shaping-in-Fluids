"""Independent mechanisms: three-layer scattering and capillary compliance."""

from dataclasses import replace

import numpy as np

from acoustic_freeform.dual.acoustics import DualAcoustics
from acoustic_freeform.dual.config import DualConfig
from acoustic_freeform.dual.surface import CartesianPatch, DualSurface
from acoustic_freeform.dual.verification import verify_capillary, verify_slab


def test_three_layer_traveling_wave_refines_to_analytic_solution():
    result = verify_slab(levels=(8, 16, 32))
    errors = [r["pressure_relative_max_error"] for r in result]
    assert errors[2] < errors[1] / 6 < errors[0] / 36
    assert errors[-1] < 1e-4
    assert max(r["power_relative_error"] for r in result) < 1e-11
    assert result[-1]["absolute_power_relative_error"] < 1e-5
    for face in range(2):
        velocity_errors = [r["interface_velocity_relative_max_error"][face] for r in result]
        assert velocity_errors[2] < velocity_errors[1] / 6 < velocity_errors[0] / 36
        assert velocity_errors[-1] < 1e-4


def test_nested_ports_preserve_the_physical_command_on_a_fixed_mesh():
    cfg = DualConfig(
        radial_cells=8,
        cells_per_layer=8,
        surface_elements=6,
        radial_ports=2,
        side_ports_per_layer=2,
    )
    space = DualSurface(cfg)
    q = np.zeros((2, space.count))
    rng = np.random.default_rng(20260922)
    drive = 0.01 * (rng.normal(size=cfg.channels) + 1j * rng.normal(size=cfg.channels))
    coarse = DualAcoustics(cfg, space).solve(q, drive)
    refined_cfg = replace(cfg, radial_ports=4, side_ports_per_layer=4)
    refined = DualAcoustics(refined_cfg, space).solve(q, np.repeat(drive, 2))
    np.testing.assert_allclose(refined.solution, coarse.solution, rtol=1e-10, atol=1e-7)
    np.testing.assert_allclose(
        refined.force(np.ones(1)), coarse.force(np.ones(1)), rtol=1e-10, atol=1e-15
    )


def test_volume_preserving_bessel_compliance():
    result = verify_capillary()
    assert result["height_max_error_m"] < 1e-14
    assert max(abs(v) for v in result["volume_error_m3"]) < 1e-25


def test_identical_acoustic_materials_have_no_radiation_contrast():
    cfg = DualConfig(
        density_kg_m3=(1000.0,) * 3,
        sound_speed_m_s=(1500.0,) * 3,
        radial_cells=4,
        cells_per_layer=4,
        surface_elements=6,
        radial_ports=2,
        side_ports_per_layer=2,
    )
    space = DualSurface(cfg)
    response = DualAcoustics(cfg, space).solve(np.zeros((2, space.count)))
    drive = np.random.default_rng(20260922).normal(size=cfg.channels).astype(complex)
    assert np.max(abs(response.force(drive))) == 0


def test_exact_cartesian_patch_projection_converges_without_changing_fill():
    cfg = DualConfig()
    optical = {"n_o": 1.33, "z_o_m": -0.1, "n_i": 1.5, "z_i_m": 0.2}
    target = CartesianPatch(cfg, 0, optical, -0.00015)
    r = np.linspace(0, cfg.clear_radius_m, 2001)
    errors = []
    for elements in (12, 24):
        space = DualSurface(replace(cfg, surface_elements=elements))
        q = space.project([target, target])[0]
        errors.append(np.max(abs(space.evaluate(q, r) - target.evaluate(r))))
        assert abs(space.weights @ (space.b @ q)) < 1e-20
        assert abs(space.evaluate(q, np.array([cfg.radius_m]))[0]) < 1e-15
    assert errors[1] < errors[0] / 10
    assert errors[1] < 1e-9
