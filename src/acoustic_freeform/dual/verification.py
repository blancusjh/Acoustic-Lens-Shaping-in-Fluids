"""Independent analytic slab and Bessel compliance limits."""

from dataclasses import replace

import numpy as np
from scipy.special import jn_zeros, jv

from .acoustics import DualAcoustics
from .config import DualConfig
from .surface import DualSurface


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


def main():
    """Independent analytic resolution audit, preserving earlier run evidence."""
    import argparse
    import json
    from pathlib import Path

    from ..provenance import capture_execution

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--linear-solver", choices=["full", "static_condensed"], default="full")
    args = parser.parse_args()
    inputs = json.loads(args.config.read_text())
    cfg = DualConfig(**inputs["apparatus"])
    cfg.validate()
    args.out.mkdir(parents=True, exist_ok=False)
    capture_execution(args.out)
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "apparatus": cfg.as_dict(),
                "analytic_axial_cells": args.levels,
                "wave_linear_solver": args.linear_solver,
                "source_configuration": str(args.config),
                "scope": "Planar one-dimensional analytic limit; not curved-candidate convergence.",
            },
            indent=2,
        )
        + "\n"
    )
    records = verify_slab(cfg, levels=args.levels, linear_solver=args.linear_solver)
    (args.out / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = [
        "# Independent three-layer analytic resolution audit",
        "",
        "This validates a planar limit, not a shaped lens or 10 nm accuracy.",
        "",
        "| Axial cells/layer | Pressure relative error | Front/back velocity relative error |",
        "|---:|---:|---:|",
    ]
    for r in records:
        v = r["interface_velocity_relative_max_error"]
        lines.append(
            f"| {r['cells_per_layer']} | {r['pressure_relative_max_error']:.6g} "
            f"| {v[0]:.6g} / {v[1]:.6g} |"
        )
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
