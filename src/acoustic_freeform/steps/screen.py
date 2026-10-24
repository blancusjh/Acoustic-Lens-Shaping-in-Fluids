"""Carrier-aware frozen-target inverse screening; never a coupled accuracy pass.

Run: python -m acoustic_freeform.inverse.screening CONFIG --out NEW_DIRECTORY
Reuse a declared frozen operator using --cache-from PREVIOUS_DIRECTORY.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import least_squares

from ..acoustics.helmholtz import DualAcoustics
from ..apparatus.config import DualConfig
from ..apparatus.sources import export_sources
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..inverse.screening import (
    PrecisionObjective,
    WhitenedObjective,
    add_joint_spot_observations,
    add_pupil_slope_observations,
    constrained_optimize,
)
from ..mechanics.surface import CartesianPatch, DualSurface
from ..verify.analytic import verify_capillary, verify_slab
from ..verify.metrics import maximum_error


def build_cache(inputs, output):
    cfg = DualConfig(**inputs["apparatus"])
    space = DualSurface(cfg)
    targets = [CartesianPatch(cfg, j, **f) for j, f in enumerate(inputs["case"]["faces"])]
    q = space.project(targets)
    validation = {
        "three_layer_slab": verify_slab(cfg, levels=(8, 16, 32)),
        "bessel_compliance": verify_capillary(cfg),
        "target_projection": [maximum_error(space, v, t) for v, t in zip(q, targets)],
        "scope": "Analytic limits and geometry projection, not coupled accuracy.",
    }
    print(f"Building frozen operator for {cfg.channels} shared sources", flush=True)
    response = DualAcoustics(cfg, space).solve(q, trace_only=True)
    required, stiffness, _ = space.mechanics(q)
    r = np.unique(
        np.r_[
            np.linspace(0, cfg.clear_radius_m, 121),
            np.linspace(cfg.clear_radius_m, cfg.radius_m, 61),
        ]
    )
    weights = np.where(r <= cfg.clear_radius_m, 1.0, inputs.get("annulus_weight", 0.2))
    observe = block_diag(space.basis(r), space.basis(r)) @ np.linalg.inv(stiffness)
    matrices = []
    face_ids = []
    for j in range(2):
        mask = response.radial_m[j] <= cfg.clear_radius_m
        matrices.append(
            response.velocity[j][mask]
            / (2 * np.pi * cfg.frequency_hz * abs(response.normals_z[j][mask, None]))
        )
        face_ids.extend([j] * np.count_nonzero(mask))
    cache = {
        "force_kernels": response.force_kernels,
        "required_force_n": required,
        "compliance_observation_m_n": observe,
        "observation_weights": np.tile(weights, 2),
        "observation_radii_m": np.tile(r, 2),
        "observation_faces": np.repeat([0, 1], len(r)),
        "carrier_height_response_s": np.vstack(matrices),
        "carrier_faces": np.array(face_ids),
        "target_coefficients_m": q,
    }
    np.savez_compressed(output / "frozen-operator.npz", **cache)
    validation["wave_linear_residual"] = response.residual
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return cache


def run(inputs_path, output, cache_from=None):
    inputs = json.loads(inputs_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    cfg = DualConfig(**inputs["apparatus"])
    cfg.validate()
    if (
        inputs.get("pupil_slope_scale_m") is not None
        or inputs.get("joint_spot_tolerance_m") is not None
    ) and inputs.get("observation_sampling") != "spline_quadrature":
        raise ValueError("Slope proxy requires dense observations")
    export_sources(output, cfg)
    (output / "resolved-config.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
    if cache_from is None:
        cache = build_cache(inputs, output)
    else:
        prior = json.loads((cache_from / "config.json").read_text())
        prior_cfg = json.loads((cache_from / "resolved-config.json").read_text())
        if (
            prior_cfg != json.loads(json.dumps(cfg.as_dict()))
            or prior["case"]["faces"] != inputs["case"]["faces"]
        ):
            raise ValueError("Cached geometry/apparatus differs from the requested problem.")
        if (
            prior.get("annulus_weight", 0.2) != inputs.get("annulus_weight", 0.2)
            and inputs.get("observation_sampling") != "spline_quadrature"
        ):
            raise ValueError("Cached observation weights differ.")
        source = cache_from / "frozen-operator.npz"
        shutil.copy2(source, output / source.name)
        shutil.copy2(cache_from / "validation.json", output / "validation.json")
        (output / "cache-provenance.json").write_text(
            json.dumps(
                {
                    "source": str(cache_from.resolve()),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "scope": "Reused frozen field operator; original execution provenance at source.",
                },
                indent=2,
            )
            + "\n"
        )
        cache = dict(np.load(source))
    initial_file = data_path(inputs["initial_drive_file"])
    if inputs.get("observation_sampling") == "spline_quadrature":
        space = DualSurface(cfg)
        required, stiffness, _ = space.mechanics(cache["target_coefficients_m"])
        np.testing.assert_allclose(required, cache["required_force_n"], rtol=1e-10, atol=1e-20)
        radii = np.unique(np.r_[0, space.r, cfg.clear_radius_m, cfg.radius_m])
        cache["compliance_observation_m_n"] = block_diag(
            space.basis(radii), space.basis(radii)
        ) @ np.linalg.inv(stiffness)
        cache["observation_radii_m"] = np.tile(radii, 2)
        cache["observation_faces"] = np.repeat([0, 1], len(radii))
        cache["observation_weights"] = np.tile(
            np.where(radii <= cfg.clear_radius_m, 1, inputs.get("annulus_weight", 0.2)), 2
        )
        if inputs.get("pupil_slope_scale_m") is not None:
            add_pupil_slope_observations(cache, space, stiffness, inputs["pupil_slope_scale_m"])
        if inputs.get("joint_spot_tolerance_m") is not None:
            add_joint_spot_observations(
                cache,
                space,
                stiffness,
                inputs["case"]["faces"],
                inputs["joint_spot_tolerance_m"],
                inputs.get("optical_observation_rays", 501),
            )
        np.savez_compressed(
            output / "observation-operator.npz",
            **{
                k: v
                for k, v in cache.items()
                if k.startswith("observation_") or k == "compliance_observation_m_n"
            },
        )
    if inputs.get("observation_only", False):
        if inputs.get("observation_sampling") != "spline_quadrature":
            raise ValueError("Observation-only mode requires dense observations")
        (output / "observation-validation.json").write_text(
            json.dumps(
                {
                    "observations": len(cache["observation_radii_m"]),
                    "joint_optical_observations": int(
                        np.sum(cache.get("observation_kind", []) == 2)
                    ),
                    "source_optimization_executed": False,
                    "coupled_equilibrium_executed": False,
                    "scope": "Local observation operator at the declared target only",
                },
                indent=2,
            )
            + "\n"
        )
        (output / "report.md").write_text(
            "# Target observation operator\n\nHeight and optional slope/optical derivatives. "
            "No source optimization, coupled equilibrium or physical trajectory executed.\n"
        )
        return
    shutil.copy2(initial_file, output / "initial-source.npz")
    initial = np.load(initial_file)["source_velocity_m_s"]
    initial = np.repeat(initial, inputs.get("initial_source_refinement_factor", 1))
    if len(initial) != cfg.channels:
        raise ValueError("Initial source vector does not match source supports.")
    limit = cfg.max_source_speed_m_s / np.sqrt(2)
    x0 = np.clip(np.r_[initial.real, initial.imag], -limit, limit)
    records = []
    for weight in inputs["carrier_weights"]:
        objective = PrecisionObjective(
            cache,
            weight,
            limit,
            inputs.get("effort_weight", 1e-5),
            peak_power=inputs.get("peak_power"),
            clear_radius_m=cfg.clear_radius_m,
            include_carrier_in_peak=inputs.get("include_carrier_in_peak", True),
        )
        working, seed, bounds = objective, x0, (-limit, limit)
        if inputs.get("coordinates") == "radiation_whitened":
            working = WhitenedObjective(
                objective,
                inputs.get("eigenvalue_cutoff", 1e-12),
                carrier_ceiling_m=inputs.get("carrier_subspace_ceiling_m"),
            )
            seed, bounds = working.encode(initial), (-np.inf, np.inf)
            print(f"Observable complex source coordinates: {working.channels}", flush=True)
        print(
            f"Carrier weight {weight}; initial residual {np.linalg.norm(working.residual(seed))}",
            flush=True,
        )
        if inputs.get("optimizer") == "slsqp":
            if not isinstance(working, WhitenedObjective):
                raise ValueError("SLSQP currently requires radiation_whitened coordinates.")
            result = constrained_optimize(
                working,
                seed,
                inputs.get("max_evaluations", 2000),
                objective_scale=inputs.get("constrained_objective_scale", 1.0),
            )
        else:
            result = least_squares(
                working.residual,
                seed,
                jac=working.jacobian,
                bounds=bounds,
                max_nfev=inputs.get("max_evaluations", 2000),
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-9,
                verbose=0,
            )
        source = working.unpack(result.x)
        scale = min(1.0, cfg.max_source_speed_m_s / max(np.max(abs(source)), 1e-30))
        source *= scale
        h, b = objective.metrics(np.r_[source.real, source.imag])
        audit_space = DualSurface(cfg)
        _, audit_stiffness, _ = audit_space.mechanics(cache["target_coefficients_m"])
        force_error = (
            np.einsum("a,iab,b->i", source.conj(), cache["force_kernels"], source).real
            - cache["required_force_n"]
        )
        perturbation = np.linalg.solve(audit_stiffness, force_error).reshape(2, -1)
        faces = cache["observation_faces"]
        pupil = cache["observation_radii_m"] <= cfg.clear_radius_m
        pupil &= cache.get("observation_kind", np.zeros(len(pupil), dtype=int)) == 0
        record = {
            "carrier_weight": weight,
            "peak_power": inputs.get("peak_power"),
            "include_carrier_in_peak": inputs.get("include_carrier_in_peak", True),
            "declared_accuracy_target": inputs.get("accuracy_target", "instantaneous"),
            "observation_sampling": inputs.get("observation_sampling", "legacy_uniform"),
            "pupil_slope_scale_m": inputs.get("pupil_slope_scale_m"),
            "joint_spot_tolerance_m": inputs.get("joint_spot_tolerance_m"),
            "frozen_linearized_spot_radius_m": (
                float(
                    np.max(abs(h[cache["observation_kind"] == 2]))
                    * 1e-8
                    / cache["observation_optical_scale"]
                )
                if "observation_optical_scale" in cache
                else None
            ),
            "frozen_sampled_pupil_slope_error_rad": [
                float(
                    np.max(
                        abs(audit_space.evaluate(q, np.linspace(0, cfg.clear_radius_m, 4001), 1))
                    )
                )
                for q in perturbation
            ],
            "frozen_polynomial_pupil_audit": [
                audit_space.polynomial_maximum(q, upper_m=cfg.clear_radius_m) for q in perturbation
            ],
            "frozen_polynomial_annulus_audit": [
                audit_space.polynomial_maximum(q, lower_m=cfg.clear_radius_m) for q in perturbation
            ],
            "optimizer_success": bool(result.success),
            "first_order_stationarity_test": bool(result.optimality <= 1e-6),
            "message": result.message,
            "evaluations": result.nfev,
            "cost": float(result.cost),
            "source_coordinates": inputs.get("coordinates", "physical_components"),
            "optimizer": inputs.get("optimizer", "least_squares"),
            "constrained_objective_scale": getattr(result, "objective_scale", None),
            "initial_projected_source_peak_m_s": getattr(
                result, "initial_projected_source_peak_m_s", None
            ),
            "carrier_subspace_ceiling_m": inputs.get("carrier_subspace_ceiling_m"),
            "carrier_subspace_operator_bound_m": getattr(working, "carrier_subspace_bound_m", None),
            "amplitude_feasibility_rescaling": float(scale),
            "optimality": float(result.optimality),
            "frozen_compliance_pupil_max_m": [
                float(np.max(abs(h[pupil & (faces == j)])) * 1e-8) for j in range(2)
            ],
            "frozen_carrier_pupil_max_m": [
                float(np.max(abs(b[cache["carrier_faces"] == j])) * 1e-8) for j in range(2)
            ],
            "source_component_peak_m_s": float(np.max(abs(source))),
            "sampled_conservative_combined_proxy_m": [
                float(
                    (
                        np.max(abs(h[pupil & (faces == j)]))
                        + np.max(abs(b[cache["carrier_faces"] == j]))
                    )
                    * 1e-8
                )
                for j in range(2)
            ],
            "coupled_accuracy_established": False,
        }
        name = f"weight-{weight:g}"
        np.savez_compressed(
            output / f"{name}.npz",
            source_velocity_m_s=source,
            target_coefficients_m=cache["target_coefficients_m"],
        )
        records.append(record)
        (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        print(record, flush=True)
    lines = [
        "# Carrier-aware frozen-target screening",
        "",
        "Not a coupled forward result, formation trajectory, or 10 nm certification.",
        "",
        "| Carrier weight | Linearized front/back (nm) | Carrier front/back (nm) | Solver termination flag |",
        "|---:|---:|---:|---|",
    ]
    for r in records:
        h, b = [
            np.array(r[k]) * 1e9
            for k in ("frozen_compliance_pupil_max_m", "frozen_carrier_pupil_max_m")
        ]
        lines.append(
            f"| {r['carrier_weight']} | {h[0]:.3f} / {h[1]:.3f} | {b[0]:.3f} / {b[1]:.3f} | {r['optimizer_success']} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def add_arguments(parser):
    parser.add_argument("config", type=Path)
    parser.add_argument("--cache-from", type=Path)


def cli(args):
    run(args.config, args.out, args.cache_from)
