"""Independent optical identities and finite/infinite conjugate checks."""

from dataclasses import replace

import numpy as np
import pytest

from acoustic_freeform.lens.cartesian import CartesianDiopter
from acoustic_freeform.lens.config import LensConfig
from acoustic_freeform.lens.excitation import normal_shape_load
from acoustic_freeform.lens.optics import trace_surface
from acoustic_freeform.lens.stationary import linear_stability
from acoustic_freeform.lens.surface import SurfaceSpace
from acoustic_freeform.lens.validation import fixed_drive_equilibrium


@pytest.mark.parametrize(
    "zo,zi", [(-0.010, 0.020), (-0.010, -0.020), (0.010, 0.020), (0.010, -0.020)]
)
def test_paper_four_conjugate_cases_agree_with_snell_and_parametric_formula(zo, zi):
    d = CartesianDiopter(1.0, zo, 1.7, zi)
    r = np.linspace(0, 0.0015, 211)
    z = d.sag(r)
    # Eq. (10) uses spherical distance rho; comparing r here would be wrong.
    rp, zp = d.parametric(np.hypot(r, z))
    np.testing.assert_allclose(rp, r, atol=2e-14)
    np.testing.assert_allclose(zp, z, atol=2e-14)
    transmitted = d.refract(r, z, d.slope(r, z))
    intercept = r + (zi - z) * transmitted[:, 0] / transmitted[:, 1]
    assert np.max(abs(intercept)) < 1e-13
    assert np.max(abs(d.fermat(r, z))) < 1e-14
    # Independently transcribed polynomial, paper Eq. (1), including squared branches.
    no, ni, rho2 = d.n_o, d.n_i, r * r + z * z
    f = ((ni * ni - no * no) * rho2 - 2 * z * (ni * ni * zi - no * no * zo)) ** 2
    f -= 4 * ni * no * (ni * zi - no * zo) * (no * zi - ni * zo) * rho2
    f -= 8 * ni * no * zi * zo * (ni * zi - no * zo) * (ni - no) * z
    assert np.max(abs(f)) < 1e-20


def test_infinite_object_recovers_existing_hyperboloid():
    no, ni, f = 1.52, 1.0, 0.020
    r = np.linspace(0, 0.004, 301)
    radius = f * (no - ni) / ni
    analytic = -r * r / (radius + np.sqrt(radius**2 + ((no / ni) ** 2 - 1) * r * r))
    d = CartesianDiopter(no, -np.inf, ni, f)
    np.testing.assert_allclose(d.sag(r), analytic, atol=2e-15)
    assert abs(d.form_parameters()["G"] + (no / ni) ** 2) < 1e-13
    distant = CartesianDiopter(no, -1e8, ni, f)
    np.testing.assert_allclose(distant.sag(r), analytic, atol=5e-13)


def test_reciprocity_and_collimated_image():
    d = CartesianDiopter(1.52, -0.050, 1.0, 0.020)
    reverse = CartesianDiopter(d.n_i, -d.z_i_m, d.n_o, -d.z_o_m)
    r = np.linspace(0, 0.003, 111)
    np.testing.assert_allclose(reverse.sag(r), -d.sag(r), atol=2e-15)
    collimator = CartesianDiopter(1.0, -0.020, 1.52, np.inf)
    z = collimator.sag(r)
    outgoing = collimator.refract(r, z, collimator.slope(r, z))
    np.testing.assert_allclose(outgoing[:, 0], 0, atol=1e-13)


def test_finite_conjugate_target_and_fixed_object_location():
    space = SurfaceSpace(LensConfig(object_distance_m=-0.050, surface_modes=32))
    target, volume = space.target()
    optics = trace_surface(space, target)
    assert optics["rms_spot_at_target_m"] < 1e-10
    assert optics["opd_rms_m"] < 1e-12
    rest = space.equilibrium(np.zeros(space.count), volume)
    traced = trace_surface(space, rest)
    r, h = traced["pupil_r_m"], traced["surface_z_m"]
    object_position = space.optical_vertex_m + space.config.object_distance_m
    reconstructed = h - r * traced["incident_direction_z"] / traced["incident_direction_r"]
    np.testing.assert_allclose(reconstructed, object_position, atol=1e-14)
    assert traced["rms_spot_at_target_m"] > 1e-5


def test_optical_indices_do_not_silently_change_the_fluid_model():
    with pytest.raises(ValueError, match="air above"):
        replace(LensConfig(), image_refractive_index=1.33).validate()


def test_single_surface_stigmatism_is_not_automatically_aplanatism():
    d = CartesianDiopter(1.52, -0.050, 1.0, 0.020)
    r = np.linspace(0.0001, 0.003, 501)
    z = d.sag(r)
    incident = d.incident_direction(r, z)
    outgoing = d.refract(r, z, d.slope(r, z))
    sine_ratio = incident[:, 0] / outgoing[:, 0]
    assert np.ptp(sine_ratio) / abs(sine_ratio.mean()) > 1e-3


def test_exported_shape_load_obeys_young_laplace_sphere_limit():
    cfg = LensConfig(gravity_m_s2=0, surface_modes=24)
    space = SurfaceSpace(cfg)
    radius = 0.011
    r = space.r * cfg.radius_m
    height = np.sqrt(radius**2 - r**2) - np.sqrt(radius**2 - cfg.radius_m**2)
    coefficients = space.fit(height / cfg.radius_m)
    load = normal_shape_load(space, coefficients, np.linspace(0, cfg.radius_m, 701))
    np.testing.assert_allclose(load, 2 * cfg.surface_tension_n_m / radius, atol=2e-7)


def test_unforced_equilibrium_has_only_decaying_stokes_modes():
    space = SurfaceSpace(LensConfig(surface_modes=8, mesh_radial=10, mesh_vertical=12))
    _, volume = space.target()
    rest = space.equilibrium(np.zeros(space.count), volume)
    stability = linear_stability(space, rest, np.zeros(space.config.array_rows, complex))
    assert stability["largest_real_growth_rate_s_inv"] < 0
    assert stability["unstable_modes"] == 0
    assert np.max(abs(np.array(stability["growth_rates_imag_s_inv"]))) < 1e-8


def test_fixed_drive_zero_limit_recovers_nonlinear_unforced_equilibrium():
    cfg = LensConfig(surface_modes=8, mesh_radial=8, mesh_vertical=12, acoustic_order=2)
    space = SurfaceSpace(cfg)
    _, volume = space.target()
    rest = space.equilibrium(np.zeros(space.count), volume)
    perturbed = rest + 0.002 * space.tangent[:, 0]
    balanced, _ = fixed_drive_equilibrium(
        space, np.zeros(cfg.array_rows, complex), perturbed, tolerance_m=1e-10
    )
    error_m = cfg.radius_m * np.sqrt(
        (balanced - rest) @ space.mass @ (balanced - rest) / np.sum(space.w)
    )
    assert error_m < 2e-10
    assert abs(space.volume_vector @ balanced - volume) < 1e-14


def test_stationary_algebraic_success_cannot_override_physical_balance(monkeypatch):
    from types import SimpleNamespace

    from acoustic_freeform.lens import validation

    cfg = LensConfig(surface_modes=8, mesh_radial=8, mesh_vertical=12, acoustic_order=2)
    space = SurfaceSpace(cfg)
    target, _ = space.target()
    monkeypatch.setattr(
        validation,
        "root",
        lambda fun, x, **kwargs: SimpleNamespace(x=x, success=True, message="mock solver success"),
    )
    # An optical target under zero drive is not a physical equilibrium, even
    # if an optimizer reports success without reducing the force residual.
    with pytest.raises(RuntimeError, match="correction tolerance"):
        fixed_drive_equilibrium(space, np.zeros(cfg.array_rows, complex), target)
