"""Physical Stokes time integration with explicitly declared ideal shape feedback."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .acoustics import CavityAcoustics
from .config import MATERIAL_PROVENANCE, LensConfig
from .design import fit_array
from .hydrodynamics import step_surface, stokes_mobility
from .optics import best_fit_conic, best_fit_sphere, spherical_optical_reference, trace_surface
from .surface import SurfaceSpace


def optical_scalars(space, coefficients):
    return {k: v for k, v in trace_surface(space, coefficients).items() if np.isscalar(v)}


def run_lens(config: LensConfig, destination, seed_drive=None, replay=None):
    """Integrate the fluid; a desired surface is never substituted for a state.

    In feedback mode the ideal controller observes the full current surface
    and fits complex array velocities on that actual acoustic geometry. Replay
    mode holds each recorded drive until its next update, without fitting or
    observation, matching the original controller's sample-and-hold program.
    Acoustic oscillations are cycle averaged; times below are fluid times.
    """
    config.validate()
    out = Path(destination)
    if (out / "report.json").exists():
        raise FileExistsError(f"Completed result already exists: {out}. Choose a new destination.")
    out.mkdir(parents=True, exist_ok=True)
    space = SurfaceSpace(config)
    acoustic = CavityAcoustics(config, space)
    target, volume = space.target()
    rest = space.equilibrium(np.zeros(space.count), volume)
    c = rest.copy()
    if seed_drive is None and replay is None:
        seed_drive = fit_array(space, acoustic.solve_basis(target), target).drive_m_s
    drive = np.zeros(config.array_rows, complex)
    count = int(np.ceil(config.end_s / config.step_s))
    times = np.linspace(0, config.end_s, count + 1)
    coefficients, drives, diagnostics = [], [], []
    acoustic_slices, fluid_slices, radiation_slices = [], [], []
    begin = time.perf_counter()
    for step, t in enumerate(times):
        field = acoustic.solve_basis(c)
        s = np.clip(t / config.ramp_s, 0, 1)
        gain = s * s * (3 - 2 * s)
        if replay is not None:
            index = np.searchsorted(replay["times_s"], t + 1e-12, side="right") - 1
            drive = replay["drive_m_s"][np.clip(index, 0, len(replay["times_s"]) - 1)].copy()
        elif step > 0:
            desired = rest + gain * (target - rest)
            warm = drive if np.max(np.abs(drive)) > 1e-5 else seed_drive * np.sqrt(gain)
            drive = fit_array(space, field, desired, starts=1, initial_drive=warm).drive_m_s
        force = field.force(drive)
        mobility = stokes_mobility(space, c)
        pressure = (
            config.density_kg_m3
            * 2
            * np.pi
            * config.frequency_hz
            * config.radius_m
            * (field.solutions @ drive)
        )
        stress = field.surface_pressure(drive, config.density_kg_m3)
        pupil = config.clear_radius_m / config.radius_m * np.sqrt((np.arange(401) + 0.5) / 401)
        gradient, _ = space.derivatives(c)
        velocity = (
            config.surface_tension_n_m
            / config.viscosity_pa_s
            * (mobility.velocity_operator @ (space.tangent.T @ (force - gradient)))
        )
        # Vector P2 coefficients are interleaved r,z.
        max_speed = float(np.max(np.linalg.norm(velocity.reshape(-1, 2), axis=1)))
        row = {
            "time_s": float(t),
            "setpoint_fraction": float(gain),
            "shape_rms_to_target_m": float(
                np.sqrt(np.mean((config.radius_m * space.evaluate(c - target, pupil)) ** 2))
            ),
            "volume_error_m3": float(
                2 * np.pi * config.radius_m**3 * (space.volume_vector @ c - volume)
            ),
            "helmholtz_relative_residual": field.residual,
            "stokes_incompressibility_error": mobility.incompressibility_error,
            "peak_acoustic_pressure_pa": float(np.max(np.abs(pressure))),
            "radiation_pressure_min_pa": float(stress.min()),
            "radiation_pressure_max_pa": float(stress.max()),
            "source_power_w": float(
                -np.pi
                * config.radius_m**2
                * np.real(np.vdot(-1j * (field.wall_loads @ drive), pressure))
            ),
            "maximum_fluid_speed_m_s": max_speed,
            "flow_reynolds_number": config.density_kg_m3
            * max_speed
            * config.radius_m
            / config.viscosity_pa_s,
            **optical_scalars(space, c),
        }
        coefficients.append(c.copy())
        drives.append(drive.copy())
        diagnostics.append(row)
        acoustic_slices.append(np.abs(pressure[field.basis.nodal_dofs[0]]).astype(np.float32))
        fluid_slices.append(velocity[mobility.velocity_basis.nodal_dofs].astype(np.float32))
        order = np.argsort(field.radial_samples)
        radiation_slices.append(
            np.interp(np.linspace(0, 1, 161), field.radial_samples[order], stress[order]).astype(
                np.float32
            )
        )
        if step % 10 == 0 or step == count:
            print(
                f"t={t:.3f} s; shape={row['shape_rms_to_target_m'] * 1e6:.4f} um; "
                f"spot={row['rms_spot_at_target_m'] * 1e6:.3f} um; "
                f"elapsed={time.perf_counter() - begin:.1f} s",
                flush=True,
            )
        if step < count:
            c = step_surface(space, c, force, mobility, times[step + 1] - t)
    np.savez_compressed(
        out / "trajectory.npz",
        times_s=times,
        coefficients=np.array(coefficients),
        drive_m_s=np.array(drives),
        target=target,
        initial=rest,
        acoustic_pressure_pa=np.array(acoustic_slices),
        fluid_velocity_m_s=np.array(fluid_slices),
        radiation_pressure_pa=np.array(radiation_slices),
    )
    report = {
        "configuration": config.as_dict(),
        "diopter": config.diopter.as_dict(),
        "illumination": (
            "Collimated incident wave inside the resin."
            if config.object_distance_m is None
            else "Prescribed spherical incident wavefront inside the resin, centered at the fixed object conjugate. External illumination and refraction through the base window are not designed."
        ),
        "material": MATERIAL_PROVENANCE,
        "control": (
            "Replay of recorded complex wall velocities; no feedback."
            if replay is not None
            else "Ideal full-surface feedback; array fit uses the actual current cavity."
        ),
        "numerics": f"Curved P{config.acoustic_order} Helmholtz; axisymmetric P2/P1 Stokes; nonlinear capillary modal evolution.",
        "duration_wall_s": time.perf_counter() - begin,
        "liquid_volume_m3": np.pi * config.radius_m**2 * config.depth_m
        + 2 * np.pi * config.radius_m**3 * volume,
        "target": {**optical_scalars(space, target), **best_fit_sphere(space, target)},
        "initial": {**optical_scalars(space, rest), **best_fit_sphere(space, rest)},
        "final": {**diagnostics[-1], **best_fit_sphere(space, c)},
        "conic_fit": best_fit_conic(space, c),
        "spherical_reference": spherical_optical_reference(space, c),
        "history": diagnostics,
        "provenance": {
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "trajectory_sha256": hashlib.sha256((out / "trajectory.npz").read_bytes()).hexdigest(),
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "configuration.json").write_text(json.dumps(config.as_dict(), indent=2) + "\n")
    return report
