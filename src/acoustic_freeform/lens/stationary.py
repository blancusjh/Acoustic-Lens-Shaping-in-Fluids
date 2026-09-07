"""Reproducible stationary inverse design, excitation export and stability."""

import csv
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..provenance import capture_execution
from .acoustics import CavityAcoustics
from .config import MATERIAL_PROVENANCE, LensConfig
from .design import design_stationary
from .excitation import normal_shape_load
from .hydrodynamics import stokes_mobility
from .optics import trace_surface
from .surface import SurfaceSpace
from .validation import fixed_drive_equilibrium


def linear_stability(space, coefficients, drive, step=1e-6):
    """Continuous, fixed-drive Stokes/interface growth rates at an equilibrium.

    Includes acoustic shape sensitivity by central finite differences. No
    controller, imposed shape interpolation, or discrete-time freezing of the
    radiation force enters this linearization. Rates are model predictions,
    not experimental measurements. High-rate modes may require fluid inertia.
    """
    q = space.tangent
    acoustic = CavityAcoustics(space.config, space)
    derivative = []
    for mode in range(q.shape[1]):
        perturbation = step * q[:, mode]
        plus = acoustic.solve_basis(coefficients + perturbation).force(drive)
        minus = acoustic.solve_basis(coefficients - perturbation).force(drive)
        derivative.append(q.T @ (plus - minus) / (2 * step))
        if mode % 8 == 0:
            print(f"stability: acoustic shape derivative {mode + 1}/{q.shape[1]}", flush=True)
    force_jacobian = np.stack(derivative, axis=1)
    _, capillary = space.derivatives(coefficients)
    mobility = stokes_mobility(space, coefficients)
    rate_matrix = (
        mobility.reduced @ (force_jacobian - q.T @ capillary @ q) / space.config.capillary_time_s
    )
    values = np.linalg.eigvals(rate_matrix)
    return {
        "scope": "Local continuous-time stability with each complex wall drive held fixed; creeping flow and instantaneous cycle-averaged acoustics, without feedback.",
        "dimensionless_difference_step": step,
        "growth_rates_real_s_inv": values.real.tolist(),
        "growth_rates_imag_s_inv": values.imag.tolist(),
        "largest_real_growth_rate_s_inv": float(values.real.max()),
        "unstable_modes": int(np.count_nonzero(values.real > 1e-5)),
        "rate_matrix_s_inv": rate_matrix.tolist(),
    }


def export_stationary(config, destination, coefficients, drive, history, stability=False):
    """Save an actual stationary solution; no synthetic physical timeline."""
    out = Path(destination)
    if (out / "report.json").exists():
        raise FileExistsError(f"Completed stationary result already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config.validate()
    space = SurfaceSpace(config)
    target, volume = space.target()
    initial = space.equilibrium(np.zeros(space.count), volume)
    c, residual = fixed_drive_equilibrium(space, drive, coefficients)
    field = CavityAcoustics(config, space).solve_basis(c)
    optics = trace_surface(space, c)
    r = np.linspace(0, config.radius_m, 1001)
    height = lambda state: config.radius_m * space.evaluate(state, r / config.radius_m)
    pressure = (
        config.density_kg_m3
        * 2
        * np.pi
        * config.frequency_hz
        * config.radius_m
        * (field.solutions @ drive)
    )
    report = {
        "kind": "stationary",
        "configuration": config.as_dict(),
        "material": MATERIAL_PROVENANCE,
        "diopter": config.diopter.as_dict(),
        "scope": "Joint stationary surface/array design, independently checked with fixed drives. No physical approach trajectory is represented by the design iterations.",
        "illumination": "Prescribed spherical incident wavefront inside the resin; external source and window refraction are not designed.",
        "liquid_volume_m3": float(
            np.pi * config.radius_m**2 * config.depth_m + 2 * np.pi * config.radius_m**3 * volume
        ),
        "coupled_equilibrium_residual": residual,
        "equilibrium_correction_tolerance_m": 1e-10,
        "optics": {k: v for k, v in optics.items() if np.isscalar(v)},
        "shape_rms_to_target_m": float(
            np.sqrt(
                np.mean(
                    (
                        config.radius_m
                        * space.evaluate(c - target, optics["pupil_r_m"] / config.radius_m)
                    )
                    ** 2
                )
            )
        ),
        "computed_vertex_above_rim_m": float(height(c)[0]),
        "target_vertex_above_rim_m": space.optical_vertex_m,
        "peak_cavity_pressure_pa": float(np.max(abs(pressure))),
        "source_power_w": float(
            -np.pi
            * config.radius_m**2
            * np.real(np.vdot(-1j * (field.wall_loads @ drive), pressure))
        ),
        "max_wall_velocity_peak_m_s": float(np.max(abs(drive))),
        "max_wall_displacement_peak_m": float(
            np.max(abs(drive)) / (2 * np.pi * config.frequency_hz)
        ),
        "design_iterations": history,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    np.savez_compressed(
        out / "stationary.npz", coefficients=c, target=target, initial=initial, drive_m_s=drive
    )
    (out / "configuration.json").write_text(json.dumps(config.as_dict(), indent=2) + "\n")
    with (out / "hold-drive.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_from_bottom_1based",
                "row_center_z_m",
                "frequency_hz",
                "velocity_real_m_s",
                "velocity_imag_m_s",
                "velocity_peak_m_s",
                "phase_deg",
                "phase_relative_to_row1_deg",
                "displacement_peak_m",
            ]
        )
        for j, w in enumerate(drive):
            writer.writerow(
                [
                    j + 1,
                    -config.depth_m + (j + 0.5) * config.depth_m / config.array_rows,
                    config.frequency_hz,
                    w.real,
                    w.imag,
                    abs(w),
                    np.degrees(np.angle(w)),
                    np.degrees(np.angle(w * drive[0].conjugate())),
                    abs(w) / (2 * np.pi * config.frequency_hz),
                ]
            )
    np.savetxt(
        out / "surface-and-load.csv",
        np.c_[
            r,
            height(initial),
            height(target),
            height(c),
            height(target) - height(initial),
            normal_shape_load(space, target, r),
        ],
        delimiter=",",
        header="radius_m,initial_height_m,target_height_m,computed_height_m,target_perturbation_m,target_sigma_kappa_plus_rho_g_h_pa",
        comments="",
    )
    convention = {
        "phasor": "v_normal(z,t)=taper(z)*Re[w_row*exp(-i*2*pi*f*t)]; normal radially out of liquid.",
        "amplitude": "Peak velocity at the center of the axial cosine taper; not RMS or voltage.",
        "taper": "0.5*(1+cos(pi*(z-z_center)/half_width)) inside the row; zero outside. half_width=element_fill*depth/(2*array_rows).",
        "phase": "Common phase is arbitrary. All azimuthal sectors in each row share its complex drive.",
        "time": "These are stationary holding excitations. No approach program or stability claim follows from force balance alone.",
        "calibration": "The wall-speed envelope is a provisional design choice. Voltages and hardware feasibility require measured loaded transducer response.",
    }
    (out / "excitation-definition.json").write_text(json.dumps(convention, indent=2) + "\n")
    if stability:
        report["stability"] = linear_stability(space, c, drive)
    report["states_sha256"] = hashlib.sha256((out / "stationary.npz").read_bytes()).hexdigest()
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_stationary(config, destination, stability=False, seed_drive=None, starts=3):
    config.validate()
    if (Path(destination) / "report.json").exists():
        raise FileExistsError("Choose a new destination for the stationary design.")
    capture_execution(destination)
    space = SurfaceSpace(config)
    c, drive, _, history = design_stationary(
        space, initial_drive=seed_drive, starts=starts, tolerance_m=1e-10
    )
    return export_stationary(config, destination, c, drive, history, stability=stability)


def regrid_stationary(source, destination, resolution):
    """Transfer fixed physical drives to another mesh and re-solve at equal volume."""
    source, out = Path(source), Path(destination)
    if (out / "report.json").exists():
        raise FileExistsError(f"Completed stationary result already exists: {out}")
    capture_execution(out)
    original = LensConfig(**json.loads((source / "configuration.json").read_text()))
    data = np.load(source / "stationary.npz")
    cfg = replace(
        original,
        mesh_radial=resolution[0],
        mesh_vertical=resolution[1],
        surface_modes=resolution[2],
    )
    old_space, space = SurfaceSpace(original), SurfaceSpace(cfg)
    volume = float(old_space.volume_vector @ data["coefficients"])
    seed = np.zeros(space.count)
    seed[: min(len(seed), len(data["coefficients"]))] = data["coefficients"][: len(seed)]
    seed += (
        space.volume_vector
        * (volume - space.volume_vector @ seed)
        / np.dot(space.volume_vector, space.volume_vector)
    )
    return export_stationary(
        cfg,
        out,
        seed,
        data["drive_m_s"],
        [
            {
                "operation": "fixed_drive_regrid",
                "source": str(source.resolve()),
                "source_state_sha256": hashlib.sha256(
                    (source / "stationary.npz").read_bytes()
                ).hexdigest(),
            }
        ],
    )


def verify_stationary(result_directory):
    """Fixed-drive spatial checks, including additional surface modes."""
    result = Path(result_directory)
    from .config import LensConfig

    cfg = LensConfig(**json.loads((result / "configuration.json").read_text()))
    data = np.load(result / "stationary.npz")
    records = []
    previous = data["coefficients"]
    resolutions = [
        (cfg.mesh_radial, cfg.mesh_vertical, cfg.surface_modes),
        (
            int(np.ceil(1.25 * cfg.mesh_radial)),
            int(np.ceil(4 * cfg.mesh_vertical / 3)),
            int(np.ceil(1.5 * cfg.surface_modes)),
        ),
    ]
    for nr, nz, modes in resolutions:
        fine = replace(cfg, acoustic_order=4, mesh_radial=nr, mesh_vertical=nz, surface_modes=modes)
        space = SurfaceSpace(fine)
        baseline = np.pad(data["coefficients"], (0, modes - cfg.surface_modes))
        initial = np.pad(previous, (0, modes - len(previous)))
        saved = result / f"refined-P4-{nr}-{nz}-{modes}.npz"
        if saved.exists():
            cache = np.load(saved)
            if np.array_equal(cache["drive_m_s"], data["drive_m_s"]):
                initial = cache["coefficients"]
        begin = time.perf_counter()
        c, residual = fixed_drive_equilibrium(space, data["drive_m_s"], initial, method="hybr")
        ray = trace_surface(space, c)
        r = ray["pupil_r_m"] / cfg.radius_m
        record = {
            "order": 4,
            "mesh": [nr, nz],
            "surface_modes": modes,
            "scope": "Same physical drive, independently re-solved stationary surface; no refit.",
            "equilibrium_solver": "Finite-difference-Jacobian Powell hybrid root solve.",
            "equilibrium_residual": residual,
            "surface_change_rms_m": float(
                cfg.radius_m * np.sqrt(np.mean(space.evaluate(c - baseline, r) ** 2))
            ),
            "ray_rms_at_target_m": ray["rms_spot_at_target_m"],
            "ray_rms_at_best_focus_m": ray["rms_spot_at_best_focus_m"],
            "opd_rms_m": ray["opd_rms_m"],
            "seconds": time.perf_counter() - begin,
            "equilibrium_correction_tolerance_m": 1e-10,
        }
        records.append(record)
        np.savez_compressed(
            result / f"refined-P4-{nr}-{nz}-{modes}.npz",
            coefficients=c,
            drive_m_s=data["drive_m_s"],
        )
        (result / "spatial-convergence.json").write_text(json.dumps(records, indent=2) + "\n")
        print(record, flush=True)
        previous = c
    return records
