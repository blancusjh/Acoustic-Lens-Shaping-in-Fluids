"""Fixed-state, fixed-command force convergence and physical sensitivity.

At one saved held-command state, each trial changes either the numerical
discretization (wave mesh, element order, quadrature, surface space) or one
declared physical parameter, and recomputes the acoustic and mechanical
generalized forces. The force-balance residual r is mapped to a height change
twice: the frozen compliance K^-1 r used by the historical inverses, and the
coupled first-order equilibrium shift (K - A)^-1 r, with A the analytic acoustic
shape stiffness at the same command. No new nonlinear equilibrium is solved;
the coupled shift is a linear prediction about the saved state.
"""

import hashlib
import json
from dataclasses import replace
from itertools import pairwise

import numpy as np

from ..acoustics.helmholtz import DualAcoustics
from ..apparatus.config import DualConfig
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..mechanics.surface import DualSurface, targets_from_record

NUMERICAL = {"radial_cells", "cells_per_layer", "acoustic_order", "surface_elements"}
PHYSICAL = {"sound_speed_m_s", "density_kg_m3", "frequency_hz", "surface_tension_n_m"}


def load_state(directory):
    """Saved held-command root: geometry, command, apparatus and target faces."""
    directory = data_path(directory)
    cfg = DualConfig(**json.loads((directory / "config.json").read_text()))
    with np.load(directory / "state.npz") as saved:
        q, drive = saved["coefficients_m"], saved["source_velocity_m_s"]
    campaign = json.loads((directory.parent / "config.json").read_text())
    synthesis_path = data_path(campaign["source_file"]).parent / "config.json"
    synthesis = json.loads(synthesis_path.read_text())
    inputs = [directory / "state.npz", directory / "config.json", synthesis_path]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    return cfg, q, drive, synthesis, hashes


def reproject(space, q, target_space):
    """The same geometry in another surface space, and the largest height change."""
    heights = np.array([space.height(j, v, target_space.r) for j, v in enumerate(q)])
    moved = np.stack(
        [
            np.linalg.solve(
                target_space.mass,
                target_space.b.T
                @ (target_space.weights * (h - target_space.base(j, target_space.r))),
            )
            for j, h in enumerate(heights)
        ]
    )
    radii = np.linspace(0, space.config.radius_m, 2001)
    change = max(
        float(np.max(abs(target_space.height(j, a, radii) - space.height(j, b, radii))))
        for j, (a, b) in enumerate(zip(moved, q))
    )
    return moved, change


def run(configuration, output):
    inputs = json.loads(configuration.read_text())
    cfg, q, drive, synthesis, hashes = load_state(inputs["state_directory"])
    space = DualSurface(cfg, allow_volume_offsets=True)
    if q.shape != (2, space.count) or drive.shape != (cfg.channels,):
        raise ValueError("Saved geometry/command does not match the saved apparatus")
    targets = targets_from_record(cfg, synthesis)
    radii = np.linspace(0, cfg.radius_m, inputs.get("radial_samples", 801))
    pupil = radii <= cfg.clear_radius_m
    target_h = np.array([t.evaluate(radii) for t in targets])
    state_h = np.array([space.height(j, v, radii) for j, v in enumerate(q)])
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    inputs["input_sha256"] = hashes
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    rows, reference = [], None
    # "reference": one analytic tangent per surface space, at the first trial's
    # wave discretization and physics that uses it. The coupled shift then
    # changes only at second order. "per_trial" recomputes A everywhere.
    tangent_mode = inputs.get("acoustic_tangent", "reference")
    if tangent_mode not in ("reference", "per_trial"):
        raise ValueError("acoustic_tangent must be reference or per_trial")
    tangents = {}
    for trial in inputs["trials"]:
        overrides = trial.get("overrides", {})
        unknown = set(overrides) - NUMERICAL - PHYSICAL
        if unknown:
            raise ValueError(f"Trial {trial['name']} changes fixed quantities: {sorted(unknown)}")
        kind = "physical" if set(overrides) & PHYSICAL else "numerical"
        if set(overrides) & PHYSICAL and set(overrides) & NUMERICAL:
            raise ValueError("A trial changes either the discretization or the physics, not both")
        trial_cfg = replace(
            cfg, **{k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
        )
        # The surface space carries tension, densities and gravity, so every trial
        # builds its own; geometry is reprojected only when the basis changes.
        trial_space = DualSurface(trial_cfg, allow_volume_offsets=True)
        if trial_cfg.surface_elements != cfg.surface_elements:
            trial_q, geometry_change = reproject(space, q, trial_space)
        else:
            trial_q, geometry_change = q, 0.0
        wave = DualAcoustics(trial_cfg, trial_space, linear_solver="static_condensed")
        wave.order = trial.get("integration_order", wave.order)
        cached = tangents.get(trial_cfg.surface_elements)
        if tangent_mode == "per_trial" or cached is None:
            response, acoustic = wave.shape_force_jacobian(trial_q, drive)
            tangents[trial_cfg.surface_elements] = (acoustic, trial["name"])
            tangent_source = trial["name"]
        else:
            response = wave.solve(trial_q, drive)
            acoustic, tangent_source = cached
        force = response.force(np.ones(1, complex))
        mechanical, stiffness, _ = trial_space.mechanics(trial_q)
        residual = force - mechanical
        shifts = {
            "frozen": np.linalg.solve(stiffness, residual).reshape(2, -1),
            "coupled": np.linalg.solve(stiffness - acoustic, residual).reshape(2, -1),
        }
        heights = {
            name: np.array([trial_space.evaluate(v, radii) for v in shift])
            for name, shift in shifts.items()
        }
        if reference is None:
            reference = heights
        predicted = state_h + heights["coupled"]
        row = {
            "name": trial["name"],
            "kind": kind,
            "overrides": overrides,
            "integration_order": wave.order,
            "sequence": trial.get("sequence"),
            "geometry_reprojection_change_m": geometry_change,
            "frozen_shift_pupil_max_m": np.max(abs(heights["frozen"][:, pupil]), axis=1).tolist(),
            "coupled_shift_pupil_max_m": np.max(abs(heights["coupled"][:, pupil]), axis=1).tolist(),
            "coupled_shift_full_max_m": np.max(abs(heights["coupled"]), axis=1).tolist(),
            "change_from_first_trial_pupil_max_m": {
                name: np.max(abs(h - reference[name])[:, pupil], axis=1).tolist()
                for name, h in heights.items()
            },
            "predicted_target_error_pupil_max_m": np.max(
                abs(predicted - target_h)[:, pupil], axis=1
            ).tolist(),
            "relative_force_change_from_mechanical_load": float(
                np.linalg.norm(residual) / np.linalg.norm(mechanical)
            ),
            "acoustic_to_mechanical_stiffness_norm_ratio": float(
                np.linalg.norm(acoustic, 2) / np.linalg.norm(stiffness, 2)
            ),
            "coupled_condition_number": float(np.linalg.cond(stiffness - acoustic)),
            "wave_diagnostics": response.diagnostics(np.ones(1, complex)),
            "acoustic_tangent_from_trial": tangent_source,
            "linear_prediction_only": True,
        }
        rows.append(row)
        np.savez_compressed(
            output / f"{trial['name']}.npz",
            radius_m=radii,
            frozen_shift_m=heights["frozen"],
            coupled_shift_m=heights["coupled"],
            generalized_force_n=force,
            acoustic_jacobian_n_m=acoustic,
        )
        print(json.dumps({k: row[k] for k in ("name", "coupled_shift_pupil_max_m")}), flush=True)
        (output / "validation.json").write_text(json.dumps(rows, indent=2) + "\n")
    sequences = {}
    for row in rows:
        if row["sequence"]:
            sequences.setdefault(row["sequence"], []).append(row)
    convergence = {}
    for label, members in sequences.items():
        steps = []
        for a, b in pairwise(members):
            fa = np.load(output / f"{a['name']}.npz")["coupled_shift_m"]
            fb = np.load(output / f"{b['name']}.npz")["coupled_shift_m"]
            steps.append(np.max(abs(fb - fa)[:, pupil], axis=1))
        ratios = [(s0 / np.maximum(s1, 1e-300)).tolist() for s0, s1 in pairwise(steps)]
        convergence[label] = {
            "levels": [m["name"] for m in members],
            "successive_coupled_shift_change_pupil_max_m": [s.tolist() for s in steps],
            "successive_change_ratios": ratios,
        }
    summary = {
        "trials": rows,
        "sequences": convergence,
        "fixed_geometry_and_source_command": True,
        "new_nonlinear_equilibria_solved": False,
        "physical_accuracy_certified": False,
    }
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Fixed-state force convergence and sensitivity",
        "",
        (
            "Height shifts are first-order predictions about the saved state: frozen "
            "K^-1 r and coupled (K - A)^-1 r. No new equilibrium is solved."
        ),
        "",
        "| Trial | Kind | Frozen pupil shift front/back (nm) | Coupled pupil shift front/back (nm) | Coupled change from first trial (nm) | ‖A‖/‖K‖ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        f, c = row["frozen_shift_pupil_max_m"], row["coupled_shift_pupil_max_m"]
        d = row["change_from_first_trial_pupil_max_m"]["coupled"]
        lines.append(
            f"| {row['name']} | {row['kind']} | {f[0] * 1e9:.3f} / {f[1] * 1e9:.3f} "
            f"| {c[0] * 1e9:.3f} / {c[1] * 1e9:.3f} | {d[0] * 1e9:.3f} / {d[1] * 1e9:.3f} "
            f"| {row['acoustic_to_mechanical_stiffness_norm_ratio']:.3g} |"
        )
    for label, data in convergence.items():
        lines += ["", f"Sequence `{label}`: {json.dumps(data)}"]
    (output / "report.md").write_text("\n".join(lines) + "\n")
