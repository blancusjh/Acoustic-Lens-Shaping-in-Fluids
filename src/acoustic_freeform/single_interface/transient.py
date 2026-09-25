"""Recorded nonlinear ALE trajectories for formation and maintained-state recovery."""

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..core.provenance import capture_execution
from .acoustics import CavityAcoustics
from .config import MATERIAL_PROVENANCE, LensConfig
from .design import fit_array
from .hydrodynamics import assemble_fluid
from .optics import best_fit_conic, best_fit_sphere, spherical_optical_reference
from .simulation import optical_scalars
from .streaming import absorption_rhs
from .surface import SurfaceSpace
from .time_step import advance_fluid


def run_transient(
    source,
    destination,
    *,
    step_s=0.005,
    end_s=0.4,
    initial="perturbation",
    perturbation_m=1e-6,
    perturbation_mode=0,
    controller_directory=None,
    noise_rms_m=0.0,
    ramp_s=1.2,
    formation_period_s=0.02,
    resolution=None,
):
    source, out = Path(source), Path(destination)
    if (out / "report.json").exists():
        raise FileExistsError(f"Completed trajectory exists: {out}")
    if initial not in ("perturbation", "rest", "rest_fixed", "hold") or step_s <= 0 or end_s <= 0:
        raise ValueError("Invalid initial state or timing.")
    if initial == "rest" and ramp_s >= end_s:
        raise ValueError("An approach run must include a hold after its ramp.")
    out.mkdir(parents=True, exist_ok=True)
    execution = capture_execution(out)
    source_cfg = LensConfig(**json.loads((source / "configuration.json").read_text()))
    cfg = replace(
        source_cfg,
        step_s=step_s,
        end_s=end_s,
        ramp_s=min(ramp_s, end_s),
    )
    if resolution is not None:
        cfg = replace(
            cfg, mesh_radial=resolution[0], mesh_vertical=resolution[1], surface_modes=resolution[2]
        )
        if cfg.surface_modes < source_cfg.surface_modes:
            raise ValueError("Refinement must retain every mode of the source surface.")
    data = np.load(source / "stationary.npz")
    goal, hold_drive = data["coefficients"], data["drive_m_s"]
    goal = np.pad(goal, (0, cfg.surface_modes - len(goal)))
    space = SurfaceSpace(cfg)
    target, volume = space.target()
    rest = space.equilibrium(np.zeros(space.count), volume)
    c = rest.copy() if initial in ("rest", "rest_fixed") else goal.copy()
    if initial == "perturbation":
        original_space = SurfaceSpace(source_cfg)
        original_direction = original_space.tangent[:, perturbation_mode]
        rms = cfg.radius_m * np.sqrt(
            original_direction @ original_space.mass @ original_direction / original_space.w.sum()
        )
        direction = np.pad(original_direction, (0, space.count - len(original_direction)))
        c += direction * perturbation_m / rms
    first_c = c.copy()
    velocity = np.zeros(assemble_fluid(space, c).velocity_basis.N)
    drive = np.zeros_like(hold_drive) if initial == "rest" else hold_drive.copy()
    pending_drive = hold_drive.copy()
    controller = None
    controller_mode = "fixed_drive"
    acoustic_jacobian = None
    if controller_directory is not None:
        controller = dict(np.load(Path(controller_directory) / "controller.npz"))
        period = float(controller["sample_period_s"])
        control_report = json.loads((Path(controller_directory) / "report.json").read_text())
        controller_mode = control_report.get("mode", "surface_feedback")
        linearization = np.load(Path(control_report["stability_source"]) / "linearization.npz")
        count_reference = len(linearization["coefficients"])
        reference_space = SurfaceSpace(replace(cfg, surface_modes=count_reference))
        embedded = np.zeros((space.count, space.count))
        embedded[:count_reference, :count_reference] = (
            reference_space.tangent
            @ linearization["acoustic_stiffness"]
            @ reference_space.tangent.T
        )
        acoustic_jacobian = space.tangent.T @ embedded @ space.tangent
        np.savez_compressed(out / "implicit-radiation-jacobian.npz", reduced=acoustic_jacobian)
        if controller_mode != "fixed_drive" and (
            period / step_s < 1 or abs(period / step_s - round(period / step_s)) > 1e-9
        ):
            raise ValueError("Fluid steps must divide the controller sample period.")
    count = int(np.ceil(end_s / step_s))
    times = np.linspace(0, end_s, count + 1)
    if abs(times[1] - step_s) > 1e-12:
        raise ValueError("End time must be an integer multiple of the fluid step.")
    states, drives, diagnostics, acoustic_slices, fluid_slices, radiation_slices = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    rng = np.random.default_rng(9062026)
    next_control, next_formation = 0.0, formation_period_s
    clipped_updates = 0
    start = time.perf_counter()
    acoustic_cfg = replace(cfg, bulk_streaming=False)
    for step, t in enumerate(times):
        field = CavityAcoustics(acoustic_cfg, space).solve_basis(c)
        formation = initial == "rest" and t < ramp_s
        if formation and t >= next_formation - 1e-12:
            fraction = np.clip(t / ramp_s, 0, 1)
            fraction = fraction * fraction * (3 - 2 * fraction)
            desired = rest + fraction * (goal - rest)
            warm = drive if np.max(abs(drive)) > 1e-5 else hold_drive * np.sqrt(fraction)
            # The formation inverse model omits bulk flow, while the forward plant includes it.
            drive = fit_array(
                space, field, desired, starts=1, initial_drive=warm, linearization_state=c
            ).drive_m_s
            next_formation += formation_period_s
        elif not formation:
            if controller is None or controller_mode == "fixed_drive":
                drive = hold_drive.copy()
            elif t >= next_control - 1e-12:
                # Apply the previous sample's command: one complete update of delay.
                drive = pending_drive.copy()
                r = controller["radial_stations_m"] / cfg.radius_m
                observed = cfg.radius_m * space.evaluate(c - goal, r)
                observed += rng.normal(0, noise_rms_m, len(r))
                xi = controller["reconstruction"] @ observed
                command = -controller["gain"] @ xi
                pending_drive = (
                    hold_drive + command[: cfg.array_rows] + 1j * command[cfg.array_rows :]
                )
                limit = np.maximum(1.0, abs(pending_drive) / cfg.max_wall_speed_m_s)
                clipped_updates += int(np.any(limit > 1.0))
                pending_drive /= limit
                next_control = t + period
        fluid = assemble_fluid(space, c)
        # The updated ALE domain changes the discrete divergence constraint.
        # Its kinetic-energy orthogonal pressure projection is part of the
        # first-order splitting, not a modification of the surface state.
        if np.linalg.norm(fluid.divergence @ velocity) > 1e-11 * max(
            np.linalg.norm(velocity), 1e-30
        ):
            velocity = fluid.solve(fluid.mass @ velocity, shift=1.0, viscosity_scale=0.0)
        body = (
            absorption_rhs(space, field, drive, fluid)
            if cfg.bulk_streaming
            else np.zeros(fluid.velocity_basis.N)
        )
        radiation = field.radiation_force(drive)
        if initial == "hold" and step == 0:
            gradient, _ = space.derivatives(c)
            velocity = fluid.solve(
                fluid.coupling.T @ (space.tangent.T @ (radiation - gradient)) + body
            )
        pressure = (
            cfg.density_kg_m3
            * 2
            * np.pi
            * cfg.frequency_hz
            * cfg.radius_m
            * (field.solutions @ drive)
        )
        physical_velocity = velocity * cfg.surface_tension_n_m / cfg.viscosity_pa_s
        speed = float(np.linalg.norm(physical_velocity.reshape(-1, 2), axis=1).max())
        stress = field.surface_pressure(drive, cfg.density_kg_m3)
        pupil = cfg.clear_radius_m / cfg.radius_m * np.sqrt((np.arange(401) + 0.5) / 401)
        error = cfg.radius_m * np.sqrt(np.mean(space.evaluate(c - goal, pupil) ** 2))
        row = {
            "time_s": float(t),
            "shape_rms_to_target_m": float(
                cfg.radius_m * np.sqrt(np.mean(space.evaluate(c - target, pupil) ** 2))
            ),
            "shape_rms_to_holding_state_m": float(error),
            "volume_error_m3": float(
                2 * np.pi * cfg.radius_m**3 * (space.volume_vector @ c - volume)
            ),
            "maximum_fluid_speed_m_s": speed,
            "flow_reynolds_number": float(
                cfg.density_kg_m3 * speed * cfg.radius_m / cfg.viscosity_pa_s
            ),
            "incompressibility_relative_error": float(
                np.linalg.norm(fluid.divergence @ velocity) / max(np.linalg.norm(velocity), 1e-30)
            ),
            "helmholtz_relative_residual": field.residual,
            "peak_acoustic_pressure_pa": float(abs(pressure).max()),
            "source_power_w": float(
                -np.pi
                * cfg.radius_m**2
                * np.real(np.vdot(-1j * (field.wall_loads @ drive), pressure))
            ),
            "radiation_pressure_min_pa": float(stress.min()),
            "radiation_pressure_max_pa": float(stress.max()),
            "maximum_wall_velocity_m_s": float(abs(drive).max()),
            **optical_scalars(space, c),
        }
        states.append(c.copy())
        drives.append(drive.copy())
        diagnostics.append(row)
        acoustic_slices.append(abs(pressure[field.basis.nodal_dofs[0]]).astype(np.float32))
        fluid_slices.append(physical_velocity[fluid.velocity_basis.nodal_dofs].astype(np.float32))
        order = np.argsort(field.radial_samples)
        radiation_slices.append(
            np.interp(np.linspace(0, 1, 161), field.radial_samples[order], stress[order]).astype(
                np.float32
            )
        )
        if step % 10 == 0 or step == count:
            print(
                f"ALE t={t:.4f} s; hold error={error * 1e6:.5f} um; ray RMS={row['rms_spot_at_target_m'] * 1e6:.4f} um; elapsed={time.perf_counter() - start:.1f} s",
                flush=True,
            )
            np.savez_compressed(
                out / "checkpoint.npz",
                times_s=times[: step + 1],
                coefficients=np.array(states),
                drive_m_s=np.array(drives),
                velocity_coefficients=velocity,
            )
            (out / "progress.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        if step < count:
            c, velocity = advance_fluid(
                space,
                c,
                velocity,
                fluid,
                radiation,
                body,
                times[step + 1] - t,
                acoustic_jacobian=None if formation else acoustic_jacobian,
            )
    np.savez_compressed(
        out / "trajectory.npz",
        times_s=times,
        coefficients=np.array(states),
        drive_m_s=np.array(drives),
        target=target,
        initial=first_c,
        holding_state=goal,
        acoustic_pressure_pa=np.array(acoustic_slices),
        fluid_velocity_m_s=np.array(fluid_slices),
        radiation_pressure_pa=np.array(radiation_slices),
    )
    report = {
        "configuration": cfg.as_dict(),
        "diopter": cfg.diopter.as_dict(),
        "material": MATERIAL_PROVENANCE,
        "scope": "Nonlinear axisymmetric ALE flow with inertia, convection, capillarity, gravity and instantaneous cycle-averaged acoustics. Bulk absorption is included when declared. No thermal feedback or wall boundary-layer streaming.",
        "control": "Fixed physical holding drives after the approach ramp; ideal surface observations are used during the ramp."
        if initial == "rest" and (controller is None or controller_mode == "fixed_drive")
        else "Fixed physical holding drives; no feedback corrections."
        if controller is None or controller_mode == "fixed_drive"
        else "Surface-height feedback with sample-and-hold commands and one full sample of sensing/computation delay.",
        "controller_source": None
        if controller_directory is None
        else str(Path(controller_directory).resolve()),
        "sensor_noise_height_rms_m": noise_rms_m,
        "saturated_control_updates": clipped_updates,
        "initial_condition": initial,
        "perturbation_full_aperture_rms_m": perturbation_m if initial == "perturbation" else 0.0,
        "perturbation_volume_null_mode": perturbation_mode,
        "initial_mean_velocity": "Stationary Stokes circulation" if initial == "hold" else "Zero",
        "formation_controller": None
        if initial != "rest"
        else {
            "ramp_s": ramp_s,
            "update_period_s": formation_period_s,
            "observation": "Ideal current full surface; zero delay during formation; radiation-only inverse model controlling the declared forward plant.",
        },
        "illumination": "Prescribed spherical wavefront inside resin; external source/window optics are not designed.",
        "duration_wall_s": time.perf_counter() - start,
        "numerics": "First-order ALE P2/P1 fluid with implicit capillarity. When an implicit radiation Jacobian is supplied, the frozen shape derivative is included for fixed-drive operation and feedback holding (W method). Actual acoustic force and body load are re-solved each step. The inverse-controlled approach ramp instead uses explicit radiation. Acoustic order follows configuration.",
        "liquid_volume_m3": float(
            np.pi * cfg.radius_m**2 * cfg.depth_m + 2 * np.pi * cfg.radius_m**3 * volume
        ),
        "target": {**optical_scalars(space, target), **best_fit_sphere(space, target)},
        "initial": {**diagnostics[0], **best_fit_sphere(space, first_c)},
        "final": {**diagnostics[-1], **best_fit_sphere(space, c)},
        "conic_fit": best_fit_conic(space, c),
        "spherical_reference": spherical_optical_reference(space, c),
        "history": diagnostics,
        "provenance": {
            "execution": execution,
            "source": str(source.resolve()),
            "completion_source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "trajectory_sha256": hashlib.sha256((out / "trajectory.npz").read_bytes()).hexdigest(),
        },
    }
    (out / "configuration.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
