"""Fixed-command low-mode acoustic shape feedback at a declared reference shape.

Central differences recompute the wave problem, not an interpolated field.
This is a truncated static response audit, not a stability certificate.
"""

import hashlib
import json

import numpy as np
from scipy.linalg import block_diag, eigh

from ..acoustics.helmholtz import DualAcoustics
from ..apparatus.config import DualConfig
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..mechanics.surface import DualSurface
from ..verify.coupled_response import low_rank_compliance, orthogonal_feedback_direction


def run(configuration, output):
    inputs = json.loads(configuration.read_text())
    cache_dir = data_path(inputs["cache_directory"])
    cache_path = cache_dir / "frozen-operator.npz"
    source_path = data_path(inputs["source_file"])
    cfg = DualConfig(**json.loads((cache_dir / "resolved-config.json").read_text()))
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    capture_execution(output)
    (output / "input-provenance.json").write_text(
        json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [cache_path, source_path]},
            indent=2,
        )
        + "\n"
    )
    with np.load(cache_path) as cache:
        q = cache["target_coefficients_m"]
    drive = np.load(source_path)["source_velocity_m_s"]
    space = DualSurface(cfg)
    required, stiffness, _ = space.mechanics(q)
    mass = block_diag(space.mass, space.mass)
    ranks = inputs["ranks"]
    family = inputs.get("direction_family", "mechanical")
    if family not in ("mechanical", "load_krylov") or max(ranks) > len(stiffness):
        raise ValueError("Invalid direction family or requested rank")
    if family == "mechanical":
        _, directions = eigh(stiffness, mass, subset_by_index=(0, max(ranks) - 1))
    else:
        directions = np.zeros((len(stiffness), max(ranks)))
    r = np.linspace(0, cfg.radius_m, 2001)
    height = block_diag(space.basis(r), space.basis(r))
    if family == "mechanical":
        directions /= np.max(abs(height @ directions), axis=0)
    wave = DualAcoustics(cfg, space, linear_solver="reused_condensed")

    def force(coefficients):
        response = wave.solve(coefficients, drive)
        return response.force(np.ones(1, complex))

    actual = force(q)
    if family == "load_krylov":
        first = np.linalg.solve(stiffness, actual - required)
        directions[:, 0] = first / np.max(abs(height @ first))
    columns = []
    differences = []
    step = inputs["difference_step_m"]
    for j, direction in enumerate(directions.T):
        delta = (step * direction).reshape(q.shape)
        column = (force(q + delta) - force(q - delta)) / (2 * step)
        columns.append(column)
        if j < inputs.get("half_step_check_modes", 0):
            half = (force(q + delta / 2) - force(q - delta / 2)) / step
            changes = height @ np.linalg.solve(stiffness, half - column)
            differences.append(
                {"mode": j, "max_compliance_derivative_change": float(max(abs(changes)))}
            )
        if family == "load_krylov" and j + 1 < max(ranks):
            new = orthogonal_feedback_direction(stiffness, directions[:, : j + 1], column)
            size = np.max(abs(height @ new))
            if size < 1e-12:
                raise RuntimeError("Krylov direction exhausted; inspect the saved lower-rank audit")
            directions[:, j + 1] = new / size
        np.savez_compressed(
            output / "tangent-checkpoint.npz",
            coefficients_m=q,
            source_velocity_m_s=drive,
            directions=directions[:, : j + 1],
            acoustic_force_derivatives_n_m=np.stack(columns, axis=1),
            stiffness_n_m=stiffness,
            force_defect_n=actual - required,
        )
        (output / "progress.json").write_text(
            json.dumps({"completed_modes": j + 1, "requested_modes": max(ranks)}) + "\n"
        )
        print(f"Fixed-command acoustic shape derivative {j + 1}/{max(ranks)}", flush=True)
    derivative = np.stack(columns, axis=1)
    rows = []
    for rank in ranks:
        compliance, small = low_rank_compliance(
            stiffness, directions[:, :rank], derivative[:, :rank]
        )
        displacement = (compliance @ (actual - required)).reshape(q.shape)
        rows.append(
            {
                "rank": rank,
                "projected_static_condition_number": float(np.linalg.cond(small)),
                "predicted_pupil_displacement_max_m": [
                    space.polynomial_maximum(x, upper_m=cfg.clear_radius_m)["max_abs_m"]
                    for x in displacement
                ],
                "new_equilibrium_solved": False,
            }
        )
        if inputs.get("check_nonlinear_prediction", False):
            trial = q + displacement
            trial_required, _, _ = space.mechanics(trial)
            trial_defect = np.linalg.solve(stiffness, force(trial) - trial_required).reshape(
                q.shape
            )
            rows[-1]["fresh_nonlinear_prediction_defect_full_surface_m"] = [
                space.polynomial_maximum(x)["max_abs_m"] for x in trial_defect
            ]
        np.savez_compressed(
            output / f"response-rank-{rank}.npz",
            compliance_m_n=compliance,
            predicted_displacement_m=displacement,
        )
    report = {
        "rank_study": rows,
        "half_step_checks": differences,
        "direction_family": family,
        "scope": "Low-mode static acoustic feedback at fixed target and command; no stability or formation claim.",
        "three_dimensional_disturbances_included": False,
        "streaming_and_thermal_feedback_included": False,
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Coupled response audit\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n"
    )
    print(json.dumps(report, indent=2), flush=True)
