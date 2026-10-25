"""Quasi-static start-up branch of a saved held command, from the unforced state.

Follows sigma F_ac(q; c) = grad E(q) for sigma = s^2 from 0 to 1 with fresh
nonlinear force balances, records the fold indicator and creeping-flow growth
rate at every point, and compares the end of the branch with the target and,
optionally, with an independently integrated start-up state.
"""

import json

import numpy as np

from ..acoustics.helmholtz import DualAcoustics
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..forward.branch import startup_branch
from ..mechanics.surface import CartesianPatch, DualSurface
from ..mechanics.viscous import assemble_mobility
from .force_convergence import load_state


def run(configuration, output):
    inputs = json.loads(configuration.read_text())
    cfg, _, drive, synthesis, hashes = load_state(inputs["state_directory"])
    material = json.loads(data_path(synthesis["material_reference"]).read_text())
    space = DualSurface(cfg, allow_volume_offsets=True)
    targets = [CartesianPatch(cfg, j, **f) for j, f in enumerate(synthesis["case"]["faces"])]
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    inputs["input_sha256"] = hashes
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    wave = DualAcoustics(cfg, space, linear_solver="static_condensed")
    fluid = inputs.get("fluid_mesh")
    viscosity = np.array(material["viscosities_pa_s"])

    def mobility(q):
        return assemble_mobility(
            space, q, viscosity, fluid["radial_cells"], fluid["cells_per_layer"]
        ).mobility

    rows = []

    def progress(point):
        row = {k: v for k, v in point.items() if k != "coefficients_m"}
        rows.append(row)
        print(json.dumps(row), flush=True)
        (output / "branch.json").write_text(json.dumps(rows, indent=2) + "\n")

    branch = startup_branch(
        wave,
        space,
        drive,
        inputs["sigmas"],
        tolerance_m=inputs.get("tolerance_m", 1e-13),
        mobility=mobility if fluid else None,
        progress=progress,
    )
    end = branch["points"][-1]["coefficients_m"]
    radii = np.linspace(0, cfg.radius_m, 801)
    pupil = radii <= cfg.clear_radius_m
    end_h = np.array([space.height(j, v, radii) for j, v in enumerate(end)])
    target_h = np.array([t.evaluate(radii) for t in targets])
    report = {
        "reached_end": branch["reached_end"],
        "stop_reason": branch["stop_reason"],
        "final_sigma": branch["points"][-1]["sigma"],
        "end_pupil_error_to_target_max_m": np.max(abs(end_h - target_h)[:, pupil], axis=1).tolist(),
        "end_vertex_heights_m": end_h[:, 0].tolist(),
        "branch": [{k: v for k, v in p.items() if k != "coefficients_m"} for p in branch["points"]],
        "model": "Quasi-static axisymmetric start-up branch; creeping-flow growth rates if a fluid mesh is given",
    }
    if inputs.get("compare_state_directory"):
        _, other, _, _, _ = load_state(inputs["compare_state_directory"])
        other_h = np.array([space.height(j, v, radii) for j, v in enumerate(other)])
        report["end_difference_to_compared_state_max_m"] = np.max(
            abs(end_h - other_h), axis=1
        ).tolist()
    np.savez_compressed(
        output / "branch.npz",
        sigma=[p["sigma"] for p in branch["points"]],
        coefficients_m=[p["coefficients_m"] for p in branch["points"]],
        source_velocity_m_s=drive,
    )
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Start-up branch\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "branch"}, indent=2), flush=True)
