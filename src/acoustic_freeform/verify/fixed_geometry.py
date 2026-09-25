"""Fixed-command, fixed-geometry wave/traction convergence; no re-optimization."""

import argparse
import gc
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..acoustics.helmholtz import DualAcoustics
from ..apparatus.config import DualConfig
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..mechanics.surface import DualSurface


def run(config_path, output):
    inputs = json.loads(Path(config_path).read_text())
    cfg = DualConfig(**inputs["apparatus"])
    state = data_path(inputs["state_file"])
    saved = np.load(state)
    q, drive = saved["coefficients_m"], saved["source_velocity_m_s"]
    space = DualSurface(cfg)
    if q.shape != (2, space.count) or drive.shape != (cfg.channels,):
        raise ValueError("Saved geometry/command does not match this configuration")
    required, stiffness, _ = space.mechanics(q)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs["state_sha256"] = hashlib.sha256(state.read_bytes()).hexdigest()
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    capture_execution(output)
    records, reference = [], None
    if inputs.get("reference_force_file"):
        reference_path = data_path(inputs["reference_force_file"])
        reference_config_path = reference_path.parent / "config.json"
        reference_cfg = DualConfig(**json.loads(reference_config_path.read_text())["apparatus"])
        for key in cfg.as_dict():
            if key not in ("radial_cells", "cells_per_layer", "acoustic_order") and (
                getattr(reference_cfg, key) != getattr(cfg, key)
            ):
                raise ValueError("Reference force changes the physical apparatus or surface space")
        with np.load(reference_path) as prior:
            if not np.array_equal(prior["coefficients_m"], q) or not np.array_equal(
                prior["source_velocity_m_s"], drive
            ):
                raise ValueError("Reference force must use exactly the same geometry and command")
            reference = prior["generalized_force_n"]
        (output / "reference-provenance.json").write_text(
            json.dumps(
                {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [reference_path, reference_config_path]
                },
                indent=2,
            )
            + "\n"
        )
    for trial in inputs["trials"]:
        numerical = replace(cfg, **trial.get("apparatus_overrides", {}))
        # Surface, materials, frequency and command supports must remain fixed.
        for key in cfg.as_dict():
            if key not in ("radial_cells", "cells_per_layer", "acoustic_order") and (
                getattr(numerical, key) != getattr(cfg, key)
            ):
                raise ValueError("Only wave-mesh/order refinement is allowed here")
        wave = DualAcoustics(numerical, space, linear_solver="static_condensed")
        # Explicit diagnostic override, recorded separately from polynomial order.
        wave.order = trial.get("integration_order", wave.order)
        response = wave.solve(q, drive)
        force = response.force(np.ones(1, complex))
        if reference is None:
            reference = force.copy()
        defect = np.linalg.solve(stiffness, force - required).reshape(2, -1)
        change = np.linalg.solve(stiffness, force - reference).reshape(2, -1)
        row = {
            "name": trial["name"],
            "apparatus": numerical.as_dict(),
            "integration_order": wave.order,
            "frozen_compliance_force_balance_defect_max_m": [
                space.polynomial_maximum(v)["max_abs_m"] for v in defect
            ],
            "frozen_compliance_force_change_pupil_max_m": [
                space.polynomial_maximum(v, upper_m=cfg.clear_radius_m)["max_abs_m"] for v in change
            ],
            "diagnostics": response.diagnostics(np.ones(1, complex)),
            "fixed_geometry_and_source_command": True,
            "new_equilibrium_solved": False,
            "physical_accuracy_certified": False,
        }
        records.append(row)
        np.savez_compressed(
            output / f"{trial['name']}.npz",
            generalized_force_n=force,
            frozen_compliance_change_m=change,
            coefficients_m=q,
            source_velocity_m_s=drive,
        )
        (output / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
        (output / "report.md").write_text(
            "# Fixed-geometry convergence audit\n\nCompliance changes are load diagnostics, "
            "not newly achieved surface errors. No inverse is executed.\n\n```json\n"
            + json.dumps(records, indent=2)
            + "\n```\n"
        )
        print(json.dumps(row), flush=True)
        del response, wave
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)
