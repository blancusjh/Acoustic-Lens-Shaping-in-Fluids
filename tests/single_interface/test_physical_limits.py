"""Physics checks with analytic solutions or conservation/dissipation laws."""

from dataclasses import replace

import numpy as np

from acoustic_freeform.single_interface.config import LensConfig
from acoustic_freeform.single_interface.hydrodynamics import step_surface, stokes_mobility
from acoustic_freeform.single_interface.optics import trace_surface
from acoustic_freeform.single_interface.surface import SurfaceSpace
from acoustic_freeform.single_interface.validation import flat_cavity_check


def test_cartesian_oval_has_equal_optical_path_and_focus():
    space = SurfaceSpace(LensConfig(surface_modes=20))
    target, _ = space.target()
    optics = trace_surface(space, target, count=709)
    assert optics["rms_spot_at_target_m"] < 1e-10
    assert optics["opd_rms_m"] < 1e-11
    assert abs(optics["best_focus_from_vertex_m"] - 0.020) < 1e-9


def test_zero_gravity_finite_cap_is_a_sphere():
    cfg = LensConfig(gravity_m_s2=0, surface_modes=22)
    space = SurfaceSpace(cfg)
    curvature_radius = 3.4  # dimensionless spherical radius
    heights = np.sqrt(curvature_radius**2 - space.r**2) - np.sqrt(curvature_radius**2 - 1)
    analytic_volume = np.dot(space.w, heights)
    equilibrium = space.equilibrium(np.zeros(space.count), analytic_volume)
    assert np.max(np.abs(space.b @ equilibrium - heights)) * cfg.radius_m < 2e-10
    assert abs(space.volume_vector @ equilibrium - analytic_volume) < 1e-13
    assert abs(space.evaluate(equilibrium, np.array([1]))[0]) < 1e-14


def test_stokes_motion_dissipates_capillary_energy_and_conserves_volume():
    cfg = LensConfig(surface_modes=12, mesh_radial=20, mesh_vertical=28)
    space = SurfaceSpace(cfg)
    target, volume = space.target()
    mobility = stokes_mobility(space, target)
    assert mobility.incompressibility_error < 1e-10
    assert np.linalg.eigvalsh(mobility.reduced).min() > 0
    next_state = step_surface(space, target, np.zeros(space.count), mobility, 0.001)
    assert space.energy(next_state) < space.energy(target)
    assert abs(space.volume_vector @ next_state - volume) < 1e-13
    # At the true unforced equilibrium the same solver must produce no motion.
    rest = space.equilibrium(np.zeros(space.count), volume)
    rest_mobility = stokes_mobility(space, rest)
    unchanged = step_surface(space, rest, np.zeros(space.count), rest_mobility, 0.001)
    assert np.linalg.norm(unchanged - rest) < 1e-11


def test_stokes_rate_obeys_inverse_viscosity_scaling():
    cfg = LensConfig(surface_modes=10, mesh_radial=16, mesh_vertical=20)
    space = SurfaceSpace(cfg)
    target, _ = space.target()
    mobility = stokes_mobility(space, target)
    slower = SurfaceSpace(replace(cfg, viscosity_pa_s=2 * cfg.viscosity_pa_s))
    a = step_surface(space, target, np.zeros(space.count), mobility, 0.001)
    b = step_surface(slower, target, np.zeros(space.count), mobility, 0.002)
    np.testing.assert_allclose(a, b, atol=1e-14)


def test_fem_acoustics_matches_independent_bessel_series():
    check = flat_cavity_check()
    assert check["pressure_relative_l2"] < 4e-4
    assert check["radiation_stress_relative_l2"] < 3e-3
    assert check["helmholtz_algebraic_residual"] < 1e-10
