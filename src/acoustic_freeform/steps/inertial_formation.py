"""Declared flat-domain potential inertia, NOT a viscous three-fluid solver."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from ..acoustics.helmholtz import DualAcoustics
from ..apparatus.config import DualConfig
from ..apparatus.sources import export_sources
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..forward.formation import envelope, inertia, midpoint_step
from ..mechanics.surface import DualSurface


def run(config_path, out):
    config = json.loads(Path(config_path).read_text())
    cfg = DualConfig(**config["apparatus"])
    space = DualSurface(cfg)
    source = data_path(config["source_state"])
    drive = np.load(source)["source_velocity_m_s"]
    if len(drive) != cfg.channels:
        raise ValueError("Source count mismatch")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    capture_execution(out)
    export_sources(out, cfg, drive)
    mass = inertia(space, config["inertia_radial_modes"])
    fine_mass = inertia(space, 2 * config["inertia_radial_modes"])
    _, stiffness, _ = space.mechanics(np.zeros((2, space.count)))
    eigen, basis = eigh(stiffness, mass, subset_by_index=[0, config["dynamic_modes"] - 1])

    def mechanics(x):
        grad, hess, energy = space.mechanics((basis @ x).reshape(2, space.count))
        return basis.T @ grad, basis.T @ hess @ basis, energy

    wave = DualAcoustics(cfg, space, linear_solver="static_condensed")
    dt, duration = config["step_s"], config["duration_s"]
    steps = round(duration / dt)
    if not np.isclose(steps * dt, duration):
        raise ValueError("Duration must be an integer multiple of step")
    x, v = np.zeros(len(eigen)), np.zeros(len(eigen))
    times, states, velocities, energy, diagnostics = [0.0], [basis @ x], [basis @ v], [0.0], []
    work = [0.0]
    for step in range(steps):
        t = (step + 0.5) * dt
        predicted = (basis @ (x + 0.5 * dt * v)).reshape(2, space.count)
        response = wave.solve(predicted, drive * envelope(t, config["ramp_s"]))
        force = basis.T @ response.force(np.ones(1, complex))
        xn, vn = midpoint_step(x, v, dt, force, mechanics)
        full = (basis @ xn).reshape(2, space.count)
        heights = np.array([space.b @ face for face in full])
        gaps = np.diff(
            np.vstack(
                [
                    np.full(len(space.r), cfg.levels_m[0]),
                    np.array(cfg.levels_m[1:3])[:, None] + heights,
                    np.full(len(space.r), cfg.levels_m[3]),
                ]
            ),
            axis=0,
        )
        if np.min(gaps) <= 0 or np.max(abs(heights)) > config["maximum_displacement_m"]:
            failure = {
                "completed": False,
                "reason": "Reduced-model displacement/gap guard exceeded",
                "last_accepted_time_s": times[-1],
                "rejected_time_s": (step + 1) * dt,
                "rejected_maximum_displacement_m": float(np.max(abs(heights))),
                "ten_nm_certified": False,
            }
            (out / "validation.json").write_text(json.dumps(failure, indent=2) + "\n")
            (out / "report.md").write_text(
                "# Stopped formation diagnostic\n\n" + json.dumps(failure, indent=2) + "\n"
            )
            (out / "wave-midpoint-diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2) + "\n"
            )
            raise RuntimeError(
                "Reduced-model displacement/gap guard exceeded; no trajectory manufactured"
            )
        work.append(work[-1] + float(force @ (xn - x)))
        x, v = xn, vn
        times.append((step + 1) * dt)
        states.append(basis @ x)
        velocities.append(basis @ v)
        energy.append(float(0.5 * v @ v + mechanics(x)[2]))
        diagnostics.append(response.diagnostics(np.ones(1, complex)))
        print(
            f"{step + 1}/{steps}: t={times[-1]:.6f} s, max |h|={np.max(abs(heights)) * 1e6:.3f} um",
            flush=True,
        )
        np.savez_compressed(
            out / "trajectory.npz",
            time_s=times,
            coefficients_m=np.array(states).reshape(-1, 2, space.count),
            velocity_coefficients_m_s=np.array(velocities).reshape(-1, 2, space.count),
            energy_j=energy,
            acoustic_work_j=work,
            modal_frequency_hz=np.sqrt(eigen) / (2 * np.pi),
        )
    (out / "wave-midpoint-diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    check = {
        "completed": True,
        "model": "Axisymmetric flat-domain potential inertia with nonlinear capillarity and moving-graph acoustics",
        "viscosity_streaming_thermal_3d_implemented": False,
        "experimental_validation": False,
        "ten_nm_certified": False,
        "target_used_in_trajectory_rhs": False,
        "inertia_relative_change_on_doubling_modes": float(
            np.linalg.norm(basis.T @ (fine_mass - mass) @ basis, 2)
        ),
        "maximum_wave_relative_residual": max(d["wave_linear_residual"] for d in diagnostics),
        "maximum_work_energy_defect_j": float(np.max(abs(np.array(energy) - work))),
        "time_step_convergence": "Requires independent fixed-command comparison",
    }
    (out / "validation.json").write_text(json.dumps(check, indent=2) + "\n")
    (out / "report.md").write_text(
        "# Two-face forward formation diagnostic\n\n" + json.dumps(check, indent=2) + "\n"
    )
