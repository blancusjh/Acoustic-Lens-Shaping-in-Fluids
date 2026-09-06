"""Analytic physical limits and independent coherence checks, in SI units."""

from dataclasses import replace

import numpy as np
import pytest

from acoustic_freeform.verification.planar.acoustics import PlanarAcoustics, RadiationBasis
from acoustic_freeform.verification.planar.array import PressureArray
from acoustic_freeform.verification.planar.config import Array, Beam, Fluid, Focus, Grid, Interface
from acoustic_freeform.verification.planar.interface import SurfaceDynamics
from acoustic_freeform.verification.planar.spectral import SpectralGrid

WATER = Fluid("water", 998.0, 1480.0, 0.001)
AIR = Fluid("air", 1.2, 343.0, 0.000018)


def grid():
    return SpectralGrid(Grid(64, 64, 0.028, 0.028))


def plane_wave(lower=WATER, upper=AIR, mode=(0, 0), amplitude=1000.0):
    g = grid()
    model = PlanarAcoustics(g, Array(), lower, upper)
    source = np.zeros(g.k.shape, dtype=complex)
    source[mode] = amplitude
    return model, model.solve(source)


@pytest.mark.parametrize("upper", [AIR, Fluid("liquid", 1100, 1700, 0.01)])
def test_normal_incidence_impedance_and_momentum(upper):
    _, s = plane_wave(upper=upper)
    z1 = WATER.density_kg_m3 * WATER.sound_speed_m_s
    z2 = upper.density_kg_m3 * upper.sound_speed_m_s
    expected_r = (z2 - z1) / (z2 + z1)
    np.testing.assert_allclose(s.reflected[0, 0] / s.incident[0, 0], expected_r, atol=1e-15)
    # Chesneau et al. PRE 106, 065104, Eq. 14. Incident peak pressure = 1000 Pa.
    intensity = 1000**2 / (2 * z1)
    expected_force = (
        2
        * intensity
        / WATER.sound_speed_m_s
        * (z1 * z1 + z2 * z2 - 2 * WATER.sound_speed_m_s / upper.sound_speed_m_s * z1 * z2)
        / (z1 + z2) ** 2
    )
    np.testing.assert_allclose(s.radiation_pressure, expected_force, rtol=2e-13)
    assert (
        abs(sum([s.powers_w["reflected"], s.powers_w["transmitted"]]) / s.powers_w["incident"] - 1)
        < 1e-14
    )


def test_pressure_release_limit_has_force_despite_small_pressure():
    light = Fluid("vanishing impedance limit", 1e-9, 343.0, 1e-12)
    _, s = plane_wave(upper=light)
    assert np.max(np.abs(s.below.pressure)) / 1000 < 1e-12
    expected = 1000**2 / (WATER.density_kg_m3 * WATER.sound_speed_m_s**2)
    np.testing.assert_allclose(s.radiation_pressure, expected, rtol=1e-10)


def test_identical_fluids_produce_no_interface_traction():
    g = grid()
    rng = np.random.default_rng(914)
    source = rng.normal(size=g.k.shape) + 1j * rng.normal(size=g.k.shape)
    s = PlanarAcoustics(g, Array(), WATER, WATER).solve(source)
    np.testing.assert_allclose(s.radiation_pressure, 0, atol=1e-17)


def test_oblique_snell_transmission_and_flux():
    upper = Fluid("faster", 1100, 1700, 0.01)
    model, s = plane_wave(upper=upper, mode=(2, 5))
    index = (2, 5)
    sin1 = model.grid.k[index] / (model.omega / WATER.sound_speed_m_s)
    sin2 = model.grid.k[index] / (model.omega / upper.sound_speed_m_s)
    np.testing.assert_allclose(sin1 / WATER.sound_speed_m_s, sin2 / upper.sound_speed_m_s)
    c1, c2 = np.sqrt(1 - sin1**2), np.sqrt(1 - sin2**2)
    z1 = WATER.density_kg_m3 * WATER.sound_speed_m_s
    z2 = upper.density_kg_m3 * upper.sound_speed_m_s
    r = (z2 * c1 - z1 * c2) / (z2 * c1 + z1 * c2)
    np.testing.assert_allclose(s.reflected[index] / s.incident[index], r, atol=1e-14)
    np.testing.assert_allclose(s.below.pressure, s.above.pressure, atol=1e-10)
    np.testing.assert_allclose(s.below.velocity[2], s.above.velocity[2], atol=1e-12)
    assert (
        abs((s.powers_w["reflected"] + s.powers_w["transmitted"]) / s.powers_w["incident"] - 1)
        < 1e-14
    )


def test_total_internal_reflection_carries_no_outgoing_upper_power():
    upper = Fluid("fast fluid", 1000, 3000, 0.01)
    model, s = plane_wave(upper=upper, mode=(0, 12))
    assert model.kz2[0, 12].imag > 0
    assert abs(abs(model.reflection[0, 12]) - 1) < 1e-14
    assert abs(s.powers_w["transmitted"]) < 1e-18
    np.testing.assert_allclose(s.powers_w["incident"], s.powers_w["reflected"], rtol=1e-14)


def test_evanescent_interference_conserves_total_normal_flux():
    g = grid()
    model = PlanarAcoustics(g, replace(Array(), retain_evanescent=True), WATER, AIR)
    source = np.zeros(g.k.shape, dtype=complex)
    source[0, 22] = 1000
    s = model.solve(source)
    assert model.kz1[0, 22].imag > 0 and model.kz2[0, 22].real > 0
    assert s.powers_w["incident"] == 0
    assert s.powers_w["transmitted"] > 0
    np.testing.assert_allclose(s.powers_w["lower_net"], s.powers_w["transmitted"], rtol=1e-12)
    np.testing.assert_allclose(
        s.powers_w["evanescent_interference"], s.powers_w["transmitted"], rtol=1e-12
    )


def test_coherent_quadratic_basis_and_global_phase():
    g = grid()
    array = PressureArray(Array(), g, WATER.sound_speed_m_s)
    model = PlanarAcoustics(g, Array(), WATER, AIR)
    w1 = array.beam_weights(Beam("a", (Focus(-0.001, 0.001),)))
    w2 = array.beam_weights(Beam("b", (Focus(0.001, -0.001),)))
    s1, s2 = [model.solve(array.spectrum(w)) for w in [w1, w2]]
    basis = RadiationBasis([s1, s2], WATER, AIR)
    exact = model.solve(array.spectrum(0.6 * w1 + 0.8 * w2)).radiation_pressure
    np.testing.assert_allclose(basis.pressure(np.array([0.6, 0.8])), exact, atol=2e-16)
    shifted = model.solve(array.spectrum(np.exp(1.2j) * w1))
    np.testing.assert_allclose(shifted.radiation_pressure, s1.radiation_pressure, atol=2e-16)
    cancellation = RadiationBasis([s1, model.solve(-array.spectrum(w1))], WATER, AIR)
    np.testing.assert_allclose(cancellation.pressure(np.ones(2)), 0, atol=1e-17)
    # The interference cross term must actually matter for this experiment.
    incoherent = 0.6**2 * s1.radiation_pressure + 0.8**2 * s2.radiation_pressure
    assert np.linalg.norm(exact - incoherent) / np.linalg.norm(exact) > 0.1


@pytest.mark.parametrize("model", ["stokes", "weakly_damped"])
def test_static_sinusoidal_shape_and_volume_constraint(model):
    g = grid()
    d = SurfaceDynamics(g, WATER, AIR, Interface(model=model), 0.0002)
    pressure = 0.1 * np.cos(2 * np.pi * 3 * g.xx / 0.028) + 0.7
    p = g.forward(pressure)
    h = np.zeros(g.k.shape, dtype=complex)
    k = 2 * np.pi * 3 / 0.028
    static = 0.1 * np.cos(k * g.xx) / (0.072 * k * k + (998 - 1.2) * 9.81)
    h = g.forward(static)
    updated, v = d.step(h, np.zeros_like(h), p)
    np.testing.assert_allclose(g.inverse(updated).real, static, atol=1e-20)
    np.testing.assert_allclose(v, 0, atol=1e-16)
    assert updated[0, 0] == 0


def test_inviscid_capillary_dispersion_and_energy_conservation():
    g = grid()
    water = replace(WATER, viscosity_pa_s=0)
    d = SurfaceDynamics(g, water, AIR, Interface(), 0.00017)
    h0 = 1e-6 * np.cos(2 * np.pi * 4 * g.xx / 0.028)
    h = g.forward(h0)
    v = np.zeros_like(h)
    e0 = sum(d.energy(h, v)[:2])
    k = 2 * np.pi * 4 / 0.028
    omega = np.sqrt(((998 - 1.2) * 9.81 * k + 0.072 * k**3) / (998 + 1.2))
    for _ in range(157):
        h, v = d.step(h, v, np.zeros_like(h))
    np.testing.assert_allclose(g.inverse(h).real, h0 * np.cos(omega * d.dt * 157), atol=1e-19)
    np.testing.assert_allclose(sum(d.energy(h, v)[:2]), e0, rtol=5e-14)


def test_two_fluid_stokes_relaxation_matches_mobility():
    g = grid()
    f1, f2 = Fluid("a", 1000, 1500, 0.2), Fluid("b", 1000, 1700, 0.3)
    d = SurfaceDynamics(g, f1, f2, Interface(0.01, 9.81, "stokes"), 0.0017)
    k = 2 * np.pi * 3 / 0.028
    h0 = 1e-6 * np.cos(k * g.xx)
    h = g.forward(h0)
    v = np.zeros_like(h)
    rate = 0.01 * k / (2 * (0.2 + 0.3))
    for _ in range(71):
        before = sum(d.energy(h, v)[:2])
        h, v = d.step(h, v, np.zeros_like(h))
        assert sum(d.energy(h, v)[:2]) <= before * (1 + 1e-14)
    np.testing.assert_allclose(g.inverse(h).real, h0 * np.exp(-rate * d.dt * 71), atol=1e-20)


@pytest.mark.parametrize("model", ["stokes", "weakly_damped"])
@pytest.mark.parametrize("z", [-0.001, 0.001])
def test_reconstructed_slow_flow_is_incompressible(model, z):
    g = grid()
    d = SurfaceDynamics(g, WATER, AIR, Interface(model=model), 0.0001)
    v = g.forward(
        0.001 * np.cos(2 * np.pi * 3 * g.xx / 0.028) * np.cos(2 * np.pi * 2 * g.yy / 0.028)
    )
    flow = d.flow_plane(v, z)
    dz = 1e-7
    dw_dz = (d.flow_plane(v, z + dz)[2] - d.flow_plane(v, z - dz)[2]) / (2 * dz)
    divergence = g.gradient(g.forward(flow[0]))[0] + g.gradient(g.forward(flow[1]))[1] + dw_dz
    assert np.max(np.abs(divergence)) < 1e-8
    np.testing.assert_allclose(d.flow_plane(v, 0)[2], g.inverse(v).real, atol=1e-18)
