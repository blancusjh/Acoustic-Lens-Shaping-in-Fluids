"""Full discrete acoustic shape Jacobian and an independent static-response audit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..acoustics.helmholtz import DualAcoustics
from ..apparatus.config import DualConfig
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..mechanics.surface import DualSurface


def run(configuration, output):
    inputs = json.loads(configuration.read_text())
    cache_dir = data_path(inputs["cache_directory"])
    cache_path = cache_dir / "frozen-operator.npz"
    source_path = data_path(inputs["source_file"])
    cfg = DualConfig(**json.loads((cache_dir / "resolved-config.json").read_text()))
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    provenance = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [cache_path, source_path]
    }
    (output / "input-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with np.load(cache_path) as cached:
        q = cached["target_coefficients_m"]
    drive = np.load(source_path)["source_velocity_m_s"]
    space = DualSurface(cfg)
    required, stiffness, _ = space.mechanics(q)
    wave = DualAcoustics(cfg, space, linear_solver="reused_condensed")

    def progress(completed, total, derivative):
        np.savez_compressed(
            output / "jacobian-checkpoint.npz",
            acoustic_jacobian_n_m=derivative,
            coefficients_m=q,
            source_velocity_m_s=drive,
        )
        (output / "progress.json").write_text(
            json.dumps({"completed_columns": completed, "total_columns": total}) + "\n"
        )
        print(f"Analytic acoustic shape Jacobian: {completed}/{total} columns", flush=True)

    response, acoustic = wave.shape_force_jacobian(
        q, drive, block_size=inputs.get("block_size", 8), progress=progress
    )
    defect = response.force(np.ones(1, complex)) - required
    coupled = stiffness - acoustic
    compliance = np.linalg.solve(coupled, np.eye(len(coupled)))
    displacement = (compliance @ defect).reshape(q.shape)
    np.savez_compressed(
        output / "linearization.npz",
        acoustic_jacobian_n_m=acoustic,
        mechanical_stiffness_n_m=stiffness,
        compliance_m_n=compliance,
        coefficients_m=q,
        source_velocity_m_s=drive,
        force_defect_n=defect,
        predicted_displacement_m=displacement,
    )
    checks = []
    reference = inputs.get("finite_difference_audit")
    if reference and (Path(reference) / "validation.json").exists():
        reference_path = Path(reference) / "tangent-checkpoint.npz"
        content = reference_path.read_bytes()
        (output / "independent-finite-differences.npz").write_bytes(content)
        with np.load(output / "independent-finite-differences.npz") as prior:
            np.testing.assert_array_equal(prior["coefficients_m"], q)
            np.testing.assert_array_equal(prior["source_velocity_m_s"], drive)
            predicted = acoustic @ prior["directions"]
            measured = prior["acoustic_force_derivatives_n_m"]
            checks = [
                float(np.linalg.norm(p - m) / max(np.linalg.norm(m), 1e-30))
                for p, m in zip(predicted.T, measured.T, strict=True)
            ]
        provenance[str(reference_path)] = hashlib.sha256(content).hexdigest()
        (output / "input-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    trial = q + displacement
    trial_required, _, _ = space.mechanics(trial)
    trial_response = wave.solve(trial, drive)
    nonlinear_defect = np.linalg.solve(
        stiffness, trial_response.force(np.ones(1, complex)) - trial_required
    ).reshape(q.shape)
    report = {
        "full_shape_dimension": len(coupled),
        "acoustic_shape_directions_truncated": False,
        "fixed_source_command": True,
        "independent_directional_relative_errors": checks,
        "preconditioned_static_condition_number": float(
            np.linalg.cond(np.linalg.solve(stiffness, coupled))
        ),
        "inverse_identity_max_error": float(
            np.max(abs(coupled @ compliance - np.eye(len(coupled))))
        ),
        "predicted_pupil_displacement_max_m": [
            space.polynomial_maximum(x, upper_m=cfg.clear_radius_m)["max_abs_m"]
            for x in displacement
        ],
        "fresh_nonlinear_prediction_defect_full_surface_m": [
            space.polynomial_maximum(x)["max_abs_m"] for x in nonlinear_defect
        ],
        "target_wave_diagnostics": response.diagnostics(np.ones(1, complex)),
        "scope": "Complete discrete static derivative for the declared axisymmetric weak-flux model. Predicted correction is not an accepted equilibrium; no dynamic stability, streaming, thermal or 3D claim.",
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Full analytic static-response audit\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)
