"""Coupled stationary inverse with fixed pupils and free non-optical annuli."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, null_space
from scipy.optimize import least_squares

from ..provenance import capture_execution
from .acoustics import DualAcoustics
from .config import DualConfig
from .precision import PrecisionObjective, WhitenedObjective
from .sources import export_sources, source_regions
from .surface import DualSurface


def annulus_spaces(space):
    # More than degree+1 points in every pupil knot interval; a spline that
    # vanishes at these points vanishes identically there, up to roundoff.
    radii = np.linspace(0, space.config.clear_radius_m, 12 * space.count + 1)
    single = null_space(space.basis(radii), rcond=1e-12)
    annulus = block_diag(single, single)
    complement = null_space(annulus.T, rcond=1e-12)
    return annulus, complement


class JointAnnulusObjective:
    """All force rows retained; annulus motion is an explicit unknown."""

    def __init__(self, source, annulus, current, penalty=1e-3, scale_m=1e-6):
        self.source, self.annulus, self.current = source, annulus, current
        self.penalty, self.scale_m = penalty, scale_m
        self.source_size = 2 * source.channels
        self.shape_rows = annulus.shape[0]

    def residual(self, x):
        result = self.source.residual(x[: self.source_size])
        change = x[self.source_size :] * self.scale_m
        result[: self.shape_rows] -= self.source.base.shape_weight * (self.annulus @ change) / 1e-8
        return np.r_[result, self.penalty * (self.current + change) / self.scale_m]

    def jacobian(self, x):
        source = self.source.jacobian(x[: self.source_size])
        result = np.zeros((source.shape[0] + self.annulus.shape[1], len(x)))
        result[: source.shape[0], : self.source_size] = source
        result[: self.shape_rows, self.source_size :] = (
            -self.source.base.shape_weight[:, None] * self.annulus * self.scale_m / 1e-8
        )
        result[source.shape[0] :, self.source_size :] = self.penalty * np.eye(self.annulus.shape[1])
        return result


def run(configuration, output):
    settings = json.loads(configuration.read_text())
    seed_path = Path(settings["source_file"])
    synthesis = json.loads((seed_path.parent / "config.json").read_text())
    apparatus = dict(synthesis["apparatus"])
    if settings.get("max_source_speed_m_s"):
        apparatus["max_source_speed_m_s"] = settings["max_source_speed_m_s"]
    cfg = DualConfig(**apparatus)
    space = DualSurface(cfg)
    with np.load(seed_path) as data:
        drive = data["source_velocity_m_s"].copy()
        target = data["target_coefficients_m"].copy()
        q = data["coefficients_m"].copy() if "coefficients_m" in data else target.copy()
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    inputs = {
        **synthesis,
        "apparatus": cfg.as_dict(),
        "free_annulus_inverse": settings,
        "seed_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
    }
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    active = np.array(
        [
            r["boundary"] == "side" or r["r_min_m"] >= synthesis["clear_endcap_radius_m"]
            for r in source_regions(cfg)
        ]
    )
    annulus, complement = annulus_spaces(space)
    print(
        f"{annulus.shape[1]} annulus coordinates; {complement.shape[1]} pupil constraints",
        flush=True,
    )
    history = []
    wave = DualAcoustics(cfg, space)
    reference_acoustic = None
    if settings.get("reference_linearization_file"):
        path = Path(settings["reference_linearization_file"])
        validation = json.loads((path.parent / "validation.json").read_text())
        if (
            validation["acoustic_shape_directions_truncated"]
            or not validation["independent_directional_relative_errors"]
        ):
            raise ValueError("A complete independently checked derivative is required")
        audit = json.loads((path.parent / "config.json").read_text())
        origin = json.loads((Path(audit["cache_directory"]) / "config.json").read_text())
        if any(
            origin[key] != synthesis[key]
            for key in ("apparatus", "case", "material_reference_sha256", "wave_model")
        ):
            raise ValueError("Reference derivative physics mismatch")
        with np.load(path) as data:
            np.testing.assert_array_equal(data["coefficients_m"], target)
            reference_acoustic = data["acoustic_jacobian_n_m"].copy()
        (output / "reference-derivative-provenance.json").write_text(
            json.dumps(
                {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                indent=2,
            )
            + "\n"
        )
    for iteration in range(settings["outer_iterations"]):
        mechanical, stiffness, _ = space.mechanics(q)
        if iteration == 0 and settings.get("initial_operator"):
            cache_path = Path(settings["initial_operator"])
            origin = json.loads((cache_path.parent / "config.json").read_text())
            if any(
                origin[key] != synthesis[key]
                for key in ("apparatus", "case", "material_reference_sha256", "wave_model")
            ):
                raise ValueError("Initial operator physics mismatch")
            with np.load(cache_path) as saved:
                if not np.array_equal(saved["target_coefficients_m"], q):
                    raise ValueError("Initial operator geometry mismatch")
                kernels = saved["force_kernels"].copy()
            (output / "initial-operator-provenance.json").write_text(
                json.dumps(
                    {
                        "file": str(cache_path),
                        "sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            print(f"Outer {iteration}: fresh full-source wave at changed annuli", flush=True)
            kernels = wave.solve(q, trace_only=True).force_kernels
        tangent = stiffness
        if reference_acoustic is not None:
            tangent = stiffness - reference_acoustic
        elif settings.get("acoustic_feedback_modes"):
            _, vectors = np.linalg.eigh(stiffness)
            directions = vectors[:, : settings["acoustic_feedback_modes"]]
            _, derivative = wave.shape_force_jacobian(
                q,
                drive,
                directions=directions,
                block_size=8,
                progress=lambda n, total, _, iteration=iteration: print(
                    f"Outer {iteration}: feedback {n}/{total}", flush=True
                ),
            )
            tangent = stiffness - derivative @ directions.T
        joint = settings.get("joint_annulus", False)
        observation = np.eye(q.size) if joint else complement
        cache = {
            "force_kernels": kernels[:, active][:, :, active],
            "required_force_n": mechanical,
            "compliance_observation_m_n": np.linalg.solve(tangent.T, observation).T,
            "observation_weights": np.ones(observation.shape[1]),
            "carrier_height_response_s": np.zeros((1, int(active.sum())), complex),
        }
        base = PrecisionObjective(cache, 0, cfg.max_source_speed_m_s / np.sqrt(2), 1e-6)
        objective = WhitenedObjective(
            base,
            cutoff=settings["source_gram_cutoff"],
            spectral_method=settings.get("source_spectral_method", "gram"),
        )
        calls = [0]
        source_objective = objective
        initial = source_objective.encode(drive[active])
        if joint:
            objective = JointAnnulusObjective(
                source_objective,
                annulus,
                annulus.T @ (q - target).ravel(),
                penalty=settings.get("annulus_penalty", 1e-3),
            )
            initial = np.r_[initial, np.zeros(annulus.shape[1])]

        def residual(x, iteration=iteration, calls=calls, objective=objective):
            calls[0] += 1
            if calls[0] % 100 == 0:
                print(f"Outer {iteration}: source evaluation {calls[0]}", flush=True)
            return objective.residual(x)

        fit = least_squares(
            residual,
            initial,
            jac=objective.jacobian,
            x_scale="jac",
            max_nfev=settings["source_evaluations"],
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-10,
        )
        drive[:] = 0
        source_size = 2 * source_objective.channels
        drive[active] = source_objective.unpack(fit.x[:source_size])
        drive *= min(1.0, cfg.max_source_speed_m_s / max(abs(drive)))
        defect = np.linalg.solve(
            stiffness, np.einsum("a,iab,b->i", drive.conj(), kernels, drive).real - mechanical
        )
        full_peak = max(space.polynomial_maximum(v)["max_abs_m"] for v in defect.reshape(q.shape))
        leakage = np.max(
            abs(space.basis(np.linspace(0, cfg.clear_radius_m, 1003)) @ (q - target).T)
        )
        row = {
            "iteration": iteration,
            "whole_interface_force_compliance_defect_m": full_peak,
            "pupil_complement_defect_m": float(np.max(abs(complement.T @ defect))),
            "pupil_change_from_target_projection_m": float(leakage),
            "source_peak_m_s": float(max(abs(drive))),
            "source_optimizer_success": bool(fit.success),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        np.savez_compressed(
            output / f"iteration-{iteration:03d}.npz",
            source_velocity_m_s=drive,
            target_coefficients_m=target,
            coefficients_m=q,
        )
        np.savez_compressed(
            output / "best-source.npz",
            source_velocity_m_s=drive,
            target_coefficients_m=target,
            coefficients_m=q,
        )
        if full_peak < settings["force_compliance_tolerance_m"]:
            break
        if joint:
            change = (annulus @ fit.x[source_size:] * objective.scale_m).reshape(q.shape)
        else:
            coupled_step = np.linalg.solve(tangent, stiffness @ defect)
            change = (annulus @ (annulus.T @ coupled_step)).reshape(q.shape)
        peak = max(space.polynomial_maximum(v)["max_abs_m"] for v in change)
        factor = min(settings["relaxation"], settings["maximum_annulus_step_m"] / max(peak, 1e-30))
        if iteration + 1 < settings["outer_iterations"]:
            q += factor * change
    export_sources(output, cfg, drive)
    report = {
        "history": history,
        "whole_force_balance_converged": bool(full_peak < settings["force_compliance_tolerance_m"]),
        "held_command_refinement_verified": False,
        "scope": "Stationary inverse iteration, not formation. Both pupils constrained; all forces checked. Existing nominal lossless harmonic model.",
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Free-annulus emitter inverse\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)
