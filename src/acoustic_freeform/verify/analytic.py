"""Independent analytic slab and Bessel compliance limits."""

from dataclasses import replace

import numpy as np
from scipy.special import jn_zeros, jv

from ..acoustics.helmholtz import DualAcoustics
from ..apparatus.config import DualConfig
from ..mechanics.surface import DualSurface


def slab_coefficients(config, incident_pa=1e5):
    """Six traveling-wave amplitudes from independent 1D matching equations."""
    omega = 2 * np.pi * config.frequency_hz
    k = omega / np.asarray(config.sound_speed_m_s)
    impedance = np.asarray(config.sound_speed_m_s) * np.asarray(config.density_kg_m3)
    matrix = np.zeros((6, 6), complex)
    rhs = np.zeros(6, complex)
    matrix[0, 0] = 1
    rhs[0] = incident_pa
    matrix[1, 5] = 1
    for j in range(2):
        distance = config.levels_m[j + 1] - config.levels_m[j]
        wave = np.exp(1j * k[j] * distance * np.array([1, -1]))
        matrix[2 + 2 * j, 2 * j : 2 * j + 2] = wave
        matrix[2 + 2 * j, 2 * j + 2 : 2 * j + 4] = -1
        matrix[3 + 2 * j, 2 * j : 2 * j + 2] = wave * np.array([1, -1]) / impedance[j]
        matrix[3 + 2 * j, 2 * j + 2 : 2 * j + 4] = np.array([-1, 1]) / impedance[j + 1]
    return np.linalg.solve(matrix, rhs).reshape(3, 2)


def verify_slab(config=None, levels=(6, 12, 18), linear_solver="full"):
    base = config or DualConfig()
    records = []
    for cells in levels:
        cfg = replace(
            base,
            radial_cells=4,
            cells_per_layer=cells,
            radial_ports=2,
            side_ports_per_layer=2,
            side_admittance_factor=0,
            surface_elements=6,
        )
        surface = DualSurface(cfg)
        wave = DualAcoustics(cfg, surface, linear_solver=linear_solver)
        incident = 1e5
        drive = np.zeros(cfg.channels, complex)
        drive[: cfg.radial_ports] = -2 * incident / (cfg.density_kg_m3[0] * cfg.sound_speed_m_s[0])
        response = wave.solve(np.zeros((2, surface.count)), drive)
        coeff = slab_coefficients(cfg, incident)
        xyz = response.basis.doflocs * cfg.radius_m
        region = np.searchsorted(cfg.levels_m[1:-1], xyz[1])
        dz = xyz[1] - np.asarray(cfg.levels_m)[region]
        phase = 2 * np.pi * cfg.frequency_hz * dz / np.asarray(cfg.sound_speed_m_s)[region]
        exact = coeff[region, 0] * np.exp(1j * phase) + coeff[region, 1] * np.exp(-1j * phase)
        relative = float(np.max(abs(response.solution[:, 0] - exact)) / incident)
        source_exact = (
            np.pi
            * cfg.radius_m**2
            * incident
            * np.real(coeff[0].sum())
            / (cfg.density_kg_m3[0] * cfg.sound_speed_m_s[0])
        )
        diagnostics = response.diagnostics(np.ones(1, complex))
        velocity_errors = []
        for interface in range(2):
            thickness = cfg.levels_m[interface + 1] - cfg.levels_m[interface]
            phase_at = 2 * np.pi * cfg.frequency_hz * thickness / cfg.sound_speed_m_s[interface]
            exact_velocity = coeff[interface, 0] * np.exp(1j * phase_at) - coeff[
                interface, 1
            ] * np.exp(-1j * phase_at)
            exact_velocity /= cfg.density_kg_m3[interface] * cfg.sound_speed_m_s[interface]
            velocity_errors.append(
                float(
                    np.max(abs(response.velocity[interface][:, 0] - exact_velocity))
                    / abs(exact_velocity)
                )
            )
        records.append(
            {
                "cells_per_layer": cells,
                "pressure_relative_max_error": relative,
                "analytic_source_work_w": float(source_exact),
                "interface_velocity_relative_max_error": velocity_errors,
                "absolute_power_relative_error": float(
                    abs(diagnostics["source_power_w"] / source_exact - 1)
                ),
                **diagnostics,
            }
        )
    return records


def verify_capillary(config=None):
    cfg = config or DualConfig()
    space = DualSurface(cfg)
    kr = jn_zeros(2, 1)[0]
    k = kr / cfg.radius_m
    amplitude = 1e-9
    exact = amplitude * (jv(0, k * space.r) - jv(0, kr))
    forces = []
    for j in range(2):
        drho = cfg.density_kg_m3[j] - cfg.density_kg_m3[j + 1]
        # The Bessel Laplacian is known independently, with J2(kR)=0 giving volume zero.
        pressure = cfg.surface_tension_n_m[j] * k**2 * amplitude * jv(0, k * space.r)
        pressure += drho * cfg.gravity_m_s2 * exact
        forces.append(space.b.T @ (space.weights * pressure))
    q = space.prescribed_equilibrium(np.concatenate(forces))
    r = np.linspace(0, cfg.radius_m, 2001)
    reference = amplitude * (jv(0, k * r) - jv(0, kr))
    return {
        "bessel_amplitude_m": amplitude,
        "height_max_error_m": float(max(np.max(abs(space.evaluate(v, r) - reference)) for v in q)),
        "volume_error_m3": [float(space.weights @ (space.b @ v)) for v in q],
    }
