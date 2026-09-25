"""Full-interface source-force synthesis followed by independent coupled checks.

Uses the declared longitudinal harmonic model, not an ideal imposed traction.
No claim of calibrated hardware, streaming or curing validation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from ..apparatus.config import DualConfig
from ..apparatus.sources import export_sources, source_regions
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..mechanics.surface import CartesianPatch, DualSurface
from ..optics.stigmatic import pair_prescription
from ..verify.metrics import maximum_error
from .screening import (
    PrecisionObjective,
    WhitenedObjective,
    add_joint_spot_observations,
    build_cache,
    constrained_optimize,
)


def checked_cache(inputs, directory, output):
    """Reuse only identical physics/geometry; the command bound is not physics."""
    inputs = json.loads(json.dumps(inputs))  # Canonicalize tuple-valued config fields.
    previous = json.loads((directory / "config.json").read_text())
    for key in ("case", "material_reference_sha256", "wave_model"):
        if previous[key] != inputs[key]:
            raise ValueError(f"Cache mismatch: {key}")
    before, after = dict(previous["apparatus"]), dict(inputs["apparatus"])
    before.pop("max_source_speed_m_s")
    after.pop("max_source_speed_m_s")
    if before != after:
        raise ValueError("Cache mismatch: apparatus")
    path = directory / "frozen-operator.npz"
    (output / "cache-provenance.json").write_text(
        json.dumps(
            {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2
        )
        + "\n"
    )
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def run(configuration, output):
    settings = json.loads(configuration.read_text())
    material_path = Path(settings["material_reference"])
    material = json.loads(material_path.read_text())
    cfg = DualConfig(**{**material["apparatus"], **settings["apparatus"]})
    faces = pair_prescription(
        cfg, material["indices"], material["stigmatic_z_m"], material["vertex_displacement_m"]
    )
    for face in faces:
        face["annulus_match_order"] = settings.get("annulus_match_order", 2)
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    inputs = {
        **settings,
        "apparatus": cfg.as_dict(),
        "annulus_weight": 1,
        "case": {"name": "noa61-water-shared-conjugate", "faces": faces},
        "material_reference_sha256": hashlib.sha256(material_path.read_bytes()).hexdigest(),
        "material_assumptions": material["parameter_status"],
        "wave_model": "Existing lossless longitudinal harmonic transmission with passive ports",
        "scope": "Nominal emitter reachability study with explicit uncertain material inputs; not full viscous/streaming wave physics",
    }
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    (output / "resolved-config.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
    export_sources(output, cfg)
    cache = (
        checked_cache(inputs, data_path(settings["cache_from"]), output)
        if settings.get("cache_from")
        else build_cache(inputs, output)
    )
    space = DualSurface(cfg)
    q = cache["target_coefficients_m"]
    _, stiffness, _ = space.mechanics(q)
    # Match every generalized force, not just a selected pupil observation.
    cache["compliance_observation_m_n"] = np.linalg.inv(stiffness)
    cache["observation_weights"] = np.ones(2 * space.count)
    cache["observation_radii_m"] = np.zeros(2 * space.count)
    cache["observation_faces"] = np.repeat([0, 1], space.count)
    if settings.get("spot_tolerance_m"):
        add_joint_spot_observations(
            cache, space, stiffness, faces, settings["spot_tolerance_m"], rays=501
        )
    if settings.get("front_rim_upper_bound_m") is not None:
        rim = np.r_[space.basis(np.array([cfg.clear_radius_m]))[0], np.zeros(space.count)]
        existing = len(cache["observation_weights"])
        cache["compliance_observation_m_n"] = np.vstack(
            [cache["compliance_observation_m_n"], np.linalg.solve(stiffness.T, rim)]
        )
        for key, value in (
            ("observation_weights", 1),
            ("observation_radii_m", cfg.clear_radius_m),
            ("observation_faces", 0),
            ("observation_kind", 0),
            ("observation_offset_m", 0),
        ):
            cache[key] = np.r_[cache.get(key, np.zeros(existing)), value]
    if settings.get("coupled_linearization_file"):
        path = data_path(settings["coupled_linearization_file"])
        audit = json.loads((path.parent / "config.json").read_text())
        origin = json.loads((data_path(audit["cache_directory"]) / "config.json").read_text())
        current = json.loads(json.dumps(inputs))
        for key in ("case", "material_reference_sha256", "wave_model"):
            if origin[key] != current[key]:
                raise ValueError(f"Coupled-response physics mismatch: {key}")
        before, after = dict(origin["apparatus"]), dict(current["apparatus"])
        before.pop("max_source_speed_m_s")
        after.pop("max_source_speed_m_s")
        if before != after:
            raise ValueError("Coupled-response apparatus mismatch")
        validation = json.loads((path.parent / "validation.json").read_text())
        checks = validation["independent_directional_relative_errors"]
        if validation["acoustic_shape_directions_truncated"] or not checks or max(checks) > 1e-5:
            raise ValueError(
                "Full acoustic feedback with independent derivative checks is required"
            )
        with np.load(path) as response:
            np.testing.assert_array_equal(response["coefficients_m"], q)
            np.testing.assert_allclose(
                response["mechanical_stiffness_n_m"], stiffness, rtol=1e-12, atol=1e-15
            )
            cache["compliance_observation_m_n"] = (
                cache["compliance_observation_m_n"] @ stiffness @ response["compliance_m_n"]
            )
        cache["observation_response_kind"] = np.array("full_coupled_static_at_reference_command")
        (output / "coupled-response-provenance.json").write_text(
            json.dumps(
                {
                    "file": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "independent_directional_relative_errors": checks,
                },
                indent=2,
            )
            + "\n"
        )
    inner = settings.get("clear_endcap_radius_m", 0)
    active = np.array(
        [
            region["boundary"] == "side" or region["r_min_m"] >= inner
            for region in source_regions(cfg)
        ]
    )
    restricted_cache = dict(cache)
    restricted_cache["force_kernels"] = cache["force_kernels"][:, active][:, :, active]
    restricted_cache["carrier_height_response_s"] = cache["carrier_height_response_s"][:, active]
    base = PrecisionObjective(
        restricted_cache, 0, cfg.max_source_speed_m_s / np.sqrt(2), settings.get("effort_weight", 0)
    )
    working = WhitenedObjective(
        base,
        cutoff=settings["source_gram_cutoff"],
        spectral_method=settings.get("source_spectral_method", "gram"),
    )
    print(
        f"Retained {working.channels} complex coordinates; {2 * space.count} force equations",
        flush=True,
    )
    rng = np.random.default_rng(settings["seed"])
    history = []
    best, best_peak, best_score = None, np.inf, np.inf
    for trial in range(settings["starts"]):
        source = rng.normal(size=cfg.channels) + 1j * rng.normal(size=cfg.channels)
        source *= settings["initial_peak_m_s"] / max(abs(source))
        if trial == 0 and settings.get("initial_source_file"):
            initial_path = data_path(settings["initial_source_file"])
            source = np.repeat(
                np.load(initial_path)["source_velocity_m_s"],
                settings.get("initial_source_refinement_factor", 1),
            )
            if source.shape != (cfg.channels,):
                raise ValueError("Initial command does not match source count")
            (output / "initial-source-provenance.json").write_text(
                json.dumps(
                    {
                        "file": str(initial_path),
                        "sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
                    },
                    indent=2,
                )
                + "\n"
            )
        if best is not None and not settings.get("independent_starts", False):
            source = best + source * 0.15
        source[~active] = 0
        x = working.encode(source[active])
        calls = [0]

        def residual(point, calls=calls, trial=trial):
            calls[0] += 1
            value = working.residual(point)
            if calls[0] % 25 == 0:
                heights, _ = working.metrics(point)
                drive = working.unpack(point)
                record = {
                    "start": trial,
                    "evaluations": calls[0],
                    "full_coefficient_defect_m": float(max(abs(heights[: 2 * space.count])) * 1e-8),
                    "source_peak_m_s": float(max(abs(drive))),
                }
                if settings.get("spot_tolerance_m"):
                    record["linearized_spot_radius_m"] = float(
                        max(abs(heights[cache["observation_kind"] == 2]))
                        * settings["spot_tolerance_m"]
                    )
                print(json.dumps(record), flush=True)
                (output / "progress.json").write_text(json.dumps(record) + "\n")
            return value

        if settings.get("optimizer") == "conic":
            from .conic import optimize

            def checkpoint(command, records, trial=trial):
                (output / f"conic-history-{trial}.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )

            command, _records = optimize(
                restricted_cache,
                source[active],
                cfg.max_source_speed_m_s,
                steps=settings["max_evaluations"],
                cutoff=settings["source_gram_cutoff"],
                annulus_weight=1,
                checkpoint=checkpoint,
                precondition=True,
                include_carrier=False,
                clear_radius_m=cfg.clear_radius_m,
                front_rim_upper_bound_m=settings.get("front_rim_upper_bound_m"),
                source_spectral_method=settings.get("source_spectral_method", "gram"),
                second_order_corrections=settings.get("second_order_corrections", 0),
                nonlinear_rim_projection=settings.get("nonlinear_rim_projection", False),
            )
            solution = OptimizeResult(
                x=working.encode(command),
                success=False,
                message="Sequential conic search finished; inspect actual residual and history",
            )
        elif settings.get("optimizer") == "constrained":
            print(f"Start {trial}: exact amplitude-disk constrained optimization", flush=True)
            solution = constrained_optimize(
                working, x, settings["max_evaluations"], settings.get("objective_scale", 1)
            )
        else:
            solution = least_squares(
                residual,
                x,
                jac=working.jacobian,
                x_scale="jac",
                max_nfev=settings["max_evaluations"],
                ftol=1e-13,
                xtol=1e-13,
                gtol=1e-10,
            )
        drive = np.zeros(cfg.channels, complex)
        drive[active] = working.unpack(solution.x)
        # Enforce the physical command bound, then evaluate the changed force.
        drive *= min(1.0, cfg.max_source_speed_m_s / max(abs(drive)))
        defect = np.linalg.solve(
            stiffness,
            np.einsum("a,iab,b->i", drive.conj(), cache["force_kernels"], drive).real
            - cache["required_force_n"],
        )
        peaks = [space.polynomial_maximum(v)["max_abs_m"] for v in defect.reshape(2, -1)]
        score = max(peaks)
        predicted_spot = None
        if settings.get("spot_tolerance_m"):
            observations, _ = base.metrics(np.r_[drive[active].real, drive[active].imag])
            predicted_spot = float(
                max(abs(observations[cache["observation_kind"] == 2]))
                * settings["spot_tolerance_m"]
            )
            score = max(max(peaks) / 1e-8, predicted_spot / settings["spot_tolerance_m"])
        history.append(
            {
                "start": trial,
                "optimizer_success": bool(solution.success),
                "message": solution.message,
                "full_surface_frozen_defect_m": peaks,
                "source_peak_m_s": float(max(abs(drive))),
                "linearized_spot_radius_m": predicted_spot,
            }
        )
        np.savez_compressed(
            output / f"source-start-{trial}.npz", source_velocity_m_s=drive, target_coefficients_m=q
        )
        if score < best_score:
            best, best_peak, best_score = drive.copy(), max(peaks), score
            np.savez_compressed(
                output / "best-source.npz", source_velocity_m_s=best, target_coefficients_m=q
            )
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if best_peak <= settings["full_surface_defect_goal_m"]:
            break
    export_sources(output, cfg, best)
    targets = [CartesianPatch(cfg, j, **face) for j, face in enumerate(faces)]
    report = {
        "history": history,
        "best_full_surface_frozen_defect_m": best_peak,
        "target_projection": [maximum_error(space, v, t) for v, t in zip(q, targets)],
        "source_channels": cfg.channels,
        "active_source_channels": int(np.sum(active)),
        "inactive_optical_aperture_channels": np.flatnonzero(~active).tolist(),
        "source_coordinates": working.channels,
        "observation_response_kind": str(cache.get("observation_response_kind", "mechanical_only")),
        "coupled_forward_verified": False,
        "spot_criterion_achieved": False,
        "source_model": "Ideal axisymmetric ports; g in V.n=Y P+g, not calibrated voltage",
        "scope": inputs["scope"],
    }
    (output / "synthesis-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Full-interface emitter synthesis\n\n"
        "Frozen load matching only; a held-command coupled check is required.\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)
