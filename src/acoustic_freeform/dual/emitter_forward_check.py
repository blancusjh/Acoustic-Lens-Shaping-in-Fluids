"""Held-source coupled Newton checks with full analytic shape feedback."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import block_diag

from ..provenance import capture_execution
from .acoustics import DualAcoustics
from .campaign import maximum_error
from .config import DualConfig
from .optics import trace_pair
from .sources import export_sources
from .surface import CartesianPatch, DualSurface


def project_acoustic_tangent(reference_space, space, acoustic):
    """Galerkin transfer of a reference derivative for root preconditioning."""
    radius = space.config.radius_m
    if reference_space.config.radius_m != radius:
        raise ValueError("Tangent transfer requires the same physical radius")
    edges = np.unique(np.r_[reference_space.raw.t, space.raw.t]) * radius
    x, w = leggauss(12)
    r = (edges[:-1, None] + np.diff(edges)[:, None] * (x + 1) / 2).ravel()
    weights = (np.diff(edges)[:, None] * w / 2).ravel() * 2 * np.pi * r
    cross = reference_space.basis(r).T @ (weights[:, None] * space.basis(r))
    single = np.linalg.solve(reference_space.mass, cross)
    transfer = block_diag(single, single)
    return transfer.T @ acoustic @ transfer


def run(configuration, output):
    inputs = json.loads(configuration.read_text())
    source_path = Path(inputs["source_file"])
    synthesis = json.loads((source_path.parent / "config.json").read_text())
    material = json.loads(Path(synthesis["material_reference"]).read_text())
    drive = np.load(source_path)["source_velocity_m_s"]
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    inputs["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    rows = []
    for grid in inputs["grids"]:
        directory = output / grid["name"]
        directory.mkdir()
        cfg = DualConfig(**{**synthesis["apparatus"], **grid["overrides"]})
        (directory / "config.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
        space = DualSurface(cfg)
        reference_acoustic = None
        if inputs.get("reference_linearization_file"):
            path = Path(inputs["reference_linearization_file"])
            audit = json.loads((path.parent / "config.json").read_text())
            origin = json.loads((Path(audit["cache_directory"]) / "config.json").read_text())
            before, after = dict(origin["apparatus"]), cfg.as_dict()
            for key in (
                "radial_cells",
                "cells_per_layer",
                "acoustic_order",
                "surface_elements",
                "max_source_speed_m_s",
            ):
                before.pop(key)
                after.pop(key)
            if before != json.loads(json.dumps(after)) or origin["case"] != synthesis["case"]:
                raise ValueError("Reference derivative physical apparatus/target mismatch")
            reference_space = DualSurface(DualConfig(**origin["apparatus"]))
            with np.load(path) as data:
                reference_acoustic = project_acoustic_tangent(
                    reference_space, space, data["acoustic_jacobian_n_m"]
                )
            (directory / "reference-tangent-provenance.json").write_text(
                json.dumps(
                    {
                        "file": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "scope": "Transferred reference Jacobian for root iteration only. Full fresh nonlinear force remains the acceptance test.",
                    },
                    indent=2,
                )
                + "\n"
            )
        targets = [
            CartesianPatch(cfg, j, **face) for j, face in enumerate(synthesis["case"]["faces"])
        ]
        q = space.project(targets)
        wave = DualAcoustics(cfg, space, linear_solver="static_condensed")
        export_sources(directory, cfg, drive)
        history = []
        converged = False
        secant = np.eye(q.size)
        previous_q = previous_residual = None
        for iteration in range(inputs["maximum_newton_steps"]):
            mechanical, stiffness, _ = space.mechanics(q)
            response = wave.solve(q, drive)
            defect = response.force(np.ones(1, complex)) - mechanical
            correction = np.linalg.solve(stiffness, defect).reshape(q.shape)
            norm = max(space.polynomial_maximum(v)["max_abs_m"] for v in correction)
            entry = {"iteration": iteration, "full_mechanical_compliance_defect_m": norm}
            history.append(entry)
            print(grid["name"], json.dumps(entry), flush=True)
            (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            np.savez_compressed(
                directory / "latest-state.npz", coefficients_m=q, source_velocity_m_s=drive
            )
            if norm < inputs["force_compliance_tolerance_m"]:
                converged = True
                break
            method = inputs.get("root_method", "newton")
            if reference_acoustic is not None:
                delta = np.linalg.solve(stiffness - reference_acoustic, defect).reshape(q.shape)
            elif method == "broyden":
                residual = -correction.ravel()
                if previous_q is not None:
                    displacement = q.ravel() - previous_q
                    denominator = displacement @ displacement
                    if denominator > 0:
                        secant += (
                            np.outer(
                                residual - previous_residual - secant @ displacement,
                                displacement,
                            )
                            / denominator
                        )
                delta = np.linalg.solve(secant, -residual).reshape(q.shape)
                previous_q, previous_residual = q.ravel().copy(), residual.copy()
            elif method == "mechanical_preconditioned":
                # A stationary iteration, not physical time stepping. Every
                # trial still solves the acoustic field on the changed faces.
                delta = correction
            else:
                modes = inputs.get("acoustic_feedback_modes", q.size)
                directions = None
                if modes < q.size:
                    # Low-rank feedback is a root-solver preconditioner only.
                    # Acceptance always tests the unprojected nonlinear force.
                    _, vectors = np.linalg.eigh(stiffness)
                    directions = vectors[:, :modes]
                response, acoustic = wave.shape_force_jacobian(
                    q,
                    drive,
                    directions=directions,
                    block_size=8,
                    progress=lambda count, total, _, name=grid["name"]: print(
                        f"{name} tangent {count}/{total}", flush=True
                    ),
                )
                if directions is not None:
                    acoustic = acoustic @ directions.T
                delta = np.linalg.solve(stiffness - acoustic, defect).reshape(q.shape)
            max_delta = max(space.polynomial_maximum(v)["max_abs_m"] for v in delta)
            factor = min(1.0, inputs["maximum_step_m"] / max(max_delta, 1e-30))
            for _ in range(12):
                trial = q + factor * delta
                try:
                    trial_response = wave.solve(trial, drive)
                    trial_mech, _, _ = space.mechanics(trial)
                    trial_defect = np.linalg.solve(
                        stiffness, trial_response.force(np.ones(1, complex)) - trial_mech
                    )
                    trial_norm = max(
                        space.polynomial_maximum(v)["max_abs_m"]
                        for v in trial_defect.reshape(q.shape)
                    )
                except ValueError:
                    trial_norm = np.inf
                if trial_norm < norm:
                    q = trial
                    break
                factor *= 0.5
            else:
                entry["line_search_failed"] = True
                break
        # The final accepted step may exhaust the iteration budget. Audit its
        # actual field, never the field/residual from the previous geometry.
        if not converged:
            mechanical, stiffness, _ = space.mechanics(q)
            response = wave.solve(q, drive)
            correction = np.linalg.solve(
                stiffness, response.force(np.ones(1, complex)) - mechanical
            ).reshape(q.shape)
            norm = max(space.polynomial_maximum(v)["max_abs_m"] for v in correction)
            converged = norm < inputs["force_compliance_tolerance_m"]
        radii = np.linspace(0, cfg.clear_radius_m, inputs["ray_count"])
        trace = trace_pair(
            space,
            q,
            material["indices"],
            material["stigmatic_z_m"][0],
            material["stigmatic_z_m"][2],
            launch_radius_m=radii,
            launch_height_m=cfg.levels_m[1] + targets[0].evaluate(radii),
        )
        errors = [maximum_error(space, v, t)["max_error_m"] for v, t in zip(q, targets)]
        row = {
            "grid": grid["name"],
            "root_method": inputs.get("root_method", "newton"),
            "acoustic_feedback_modes": inputs.get("acoustic_feedback_modes", q.size),
            "coupled_root_converged": converged,
            "mean_pupil_max_error_m": errors,
            "max_spot_radius_m": trace["max_radius_m"],
            "all_rays_transmitted": trace["all_rays_transmitted"],
            "sampled_joint_gate": bool(
                converged
                and max(errors) <= 1e-8
                and trace["max_radius_m"] < 1e-6
                and trace["all_rays_transmitted"]
            ),
            "wave_diagnostics": response.diagnostics(np.ones(1, complex)),
            "force_compliance_defect_m": norm,
            "scope": "Held-source stationary mean, nominal lossless longitudinal acoustic model; no formation/stability/experimental claim",
        }
        np.savez_compressed(
            directory / "state.npz",
            coefficients_m=q,
            source_velocity_m_s=drive,
            spots_m=trace["spots_m"],
            transmitted=trace["transmitted"],
        )
        (directory / "validation.json").write_text(json.dumps(row, indent=2) + "\n")
        (directory / "report.md").write_text(
            "# Held-source forward check\n\n```json\n" + json.dumps(row, indent=2) + "\n```\n"
        )
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    (output / "validation.json").write_text(
        json.dumps({"grids": rows, "physical_accuracy_certified": False}, indent=2) + "\n"
    )
    (output / "report.md").write_text(
        "# Emitter-driven coupled checks\n\n```json\n" + json.dumps(rows, indent=2) + "\n```\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)
