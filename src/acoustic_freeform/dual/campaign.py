"""Versioned numerical experiment output for simultaneous front/back shaping."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..provenance import capture_execution
from .acoustics import DualAcoustics
from .config import DualConfig
from .design import coupled_equilibrium, synthesize, synthesize_with_annulus
from .sources import export_sources
from .surface import CartesianPatch, DualSurface
from .verification import verify_capillary, verify_slab


def maximum_error(space, q, target):
    """Independent dense sampling and batched derivative-root refinement.

    This is a numerical extremum search, not an interval-arithmetic certificate.
    The mesh is independent of acoustic quadrature and surface collocation.
    """
    r = np.linspace(0, space.config.clear_radius_m, 2001)
    target_h = target.evaluate(r)
    error = space.evaluate(q, r) - target_h
    values = abs(error)
    best = float(np.max(values))
    derivative = space.evaluate(q, r, 1) - target.evaluate(r, 1)
    crossings = np.flatnonzero(derivative[:-1] * derivative[1:] < 0)
    low, high = r[crossings].copy(), r[crossings + 1].copy()
    left = derivative[crossings].copy()
    if len(crossings):
        # Batch the independent derivative-root searches. Scalar minimizers
        # spend minutes resolving roundoff peaks of near-exact projections.
        for _ in range(32):
            middle = (low + high) / 2
            value = space.evaluate(q, middle, 1) - target.evaluate(middle, 1)
            same_sign = np.signbit(value) == np.signbit(left)
            low = np.where(same_sign, middle, low)
            high = np.where(same_sign, high, middle)
            left = np.where(same_sign, value, left)
        roots = (low + high) / 2
        best = max(best, float(np.max(abs(space.evaluate(q, roots) - target.evaluate(roots)))))
    rms = np.sqrt(np.trapezoid(r * error**2, r) * 2 / space.config.clear_radius_m**2)
    return {
        "max_error_m": best,
        "rms_error_m": float(rms),
        "metric": "Absolute lab-coordinate height; no piston or tilt removal.",
        "search": "2001 independent radii and batched bisection of derivative sign changes.",
        "rigorous_interval_bound": False,
    }


def geometric_acceptance(mean_errors, phase_errors, target):
    """Numerical geometry gates only; never a physical or convergence certificate."""
    if target not in ("cycle_mean", "instantaneous"):
        raise ValueError("Accuracy target must be cycle_mean or instantaneous.")
    mean = np.asarray(mean_errors)
    phase = np.asarray(phase_errors)
    if mean.shape != (2,) or phase.shape != (2,) or not np.all(np.isfinite([mean, phase])):
        raise ValueError("Finite errors for both faces are required.")
    return {
        "declared_accuracy_target": target,
        "mean_10nm_sampled_test": bool(np.all(mean <= 1e-8)),
        "instantaneous_10nm_rejected_by_samples": bool(np.any(phase > 1e-8)),
        "declared_10nm_sampled_test": bool(
            np.all((mean if target == "cycle_mean" else phase) <= 1e-8)
        ),
        "physical_accuracy_certified": False,
    }


def run(configuration: Path, output: Path):
    inputs = json.loads(configuration.read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    capture_execution(output)
    cfg = DualConfig(**inputs["apparatus"])
    cfg.validate()
    export_sources(output, cfg)
    (output / "resolved-config.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
    verification = {
        "three_layer_slab": verify_slab(cfg, levels=(8, 16, 32)),
        "bessel_compliance": verify_capillary(cfg),
    }
    (output / "validation.json").write_text(json.dumps(verification, indent=2) + "\n")
    records = []
    for case in inputs["cases"]:
        case_dir = output / case["name"]
        case_dir.mkdir()
        targets = [CartesianPatch(cfg, j, **face) for j, face in enumerate(case["faces"])]
        space = DualSurface(cfg)
        q_target = space.project(targets)
        print(f"{case['name']}: {cfg.channels} sources, {space.count} modes per face", flush=True)
        print(
            "target projection",
            [maximum_error(space, q, t)["max_error_m"] for q, t in zip(q_target, targets)],
            flush=True,
        )
        wave = DualAcoustics(cfg, space)
        starting_drive = None
        initial_file = case.get("fixed_drive_file", case.get("initial_drive_file"))
        if initial_file is not None:
            starting_drive = np.load(initial_file)["source_velocity_m_s"]
            shutil.copy2(initial_file, case_dir / "initial-source.npz")
            starting_drive = np.repeat(
                starting_drive, case.get("initial_source_refinement_factor", 1)
            )
            if len(starting_drive) != cfg.channels:
                raise ValueError("Initial source vector is incompatible with the port layout.")
        if "fixed_drive_file" in case:
            drive = starting_drive
            design = {
                "success": None,
                "message": "Verification of supplied fixed commands; no inverse executed.",
            }
        elif inputs.get("joint_annulus", False):

            def checkpoint(q, drive, calls, residual, case_dir=case_dir):
                np.savez_compressed(
                    case_dir / "latest-iterate.npz", coefficients_m=q, source_velocity_m_s=drive
                )
                (case_dir / "progress.json").write_text(
                    json.dumps(
                        {
                            "status": "running",
                            "evaluations": calls,
                            "displacement_residual_m": residual,
                        }
                    )
                    + "\n"
                )

            drive, q_target, design = synthesize_with_annulus(
                wave,
                space,
                q_target,
                initial=starting_drive,
                max_evaluations=inputs.get("max_evaluations", 100),
                annulus_modes=inputs.get("annulus_modes", 6),
                seed=case.get("seed", 20260922),
                checkpoint=checkpoint,
            )
        else:
            drive, _response, design = synthesize(
                wave,
                space,
                q_target,
                initial=starting_drive,
                seed=case.get("seed", 20260922),
                max_evaluations=inputs.get("max_evaluations", 600),
                effort_weight=inputs.get("effort_weight", 1e-6),
                starts=inputs.get("starts", 1),
            )
        print("source inverse", design, flush=True)
        export_sources(case_dir, cfg, drive)
        np.savez_compressed(
            case_dir / "design.npz",
            source_velocity_m_s=drive,
            target_coefficients_m=q_target,
            annulus_coefficients_m=np.stack([t.annulus.coef for t in targets]),
        )
        (case_dir / "source-inverse.json").write_text(json.dumps(design, indent=2) + "\n")
        (case_dir / "progress.json").write_text(
            json.dumps({"status": "forward_verification"}) + "\n"
        )
        grids = [("design", cfg)] + [
            (f"refined-{j + 1}", replace(cfg, **refinement))
            for j, refinement in enumerate(inputs.get("refinements", []))
        ]
        grid_records = []
        # Headless report rendering must not change a caller's notebook backend.
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2)
        for label, numerical in grids:
            forward_space = DualSurface(numerical)
            forward_wave = DualAcoustics(
                numerical, forward_space, linear_solver=inputs.get("wave_linear_solver", "full")
            )
            initial = np.stack(
                [
                    np.linalg.solve(
                        forward_space.mass,
                        forward_space.b.T
                        @ (forward_space.weights * space.evaluate(q, forward_space.r)),
                    )
                    for q in q_target
                ]
            )

            def root_progress(data, path=case_dir / f"{label}-root-progress.json"):
                path.write_text(json.dumps(data, indent=2) + "\n")

            solved, stationary = coupled_equilibrium(
                forward_wave,
                forward_space,
                drive,
                initial,
                method=inputs.get("stationary_solver", "hybr"),
                progress=root_progress,
            )
            achieved_wave = forward_wave.solve(solved, drive)
            diagnostics = achieved_wave.diagnostics(np.ones(1, complex))
            diagnostics["source_component_peak_m_s"] = float(np.max(abs(drive)))
            errors = [maximum_error(forward_space, q, target) for q, target in zip(solved, targets)]
            carrier_sum_sampled = []
            for j in range(2):
                rr = achieved_wave.radial_m[j]
                mask = rr <= cfg.clear_radius_m
                mean = abs(
                    forward_space.evaluate(solved[j], rr[mask]) - targets[j].evaluate(rr[mask])
                )
                carrier = abs(achieved_wave.velocity[j][mask, 0]) / (
                    2 * np.pi * cfg.frequency_hz * abs(achieved_wave.normals_z[j][mask])
                )
                carrier_sum_sampled.append(float(np.max(mean + carrier)))
            record = {
                "grid": label,
                "numerics": numerical.as_dict(),
                "wave_linear_solver": inputs.get("wave_linear_solver", "full"),
                "faces": errors,
                "equilibrium": stationary,
                "acoustics": diagnostics,
                "power_limit_w": inputs.get("max_power_w"),
                "power_limit_met": diagnostics["source_power_w"]
                <= inputs.get("max_power_w", np.inf),
                "sampled_phase_max_error_m": carrier_sum_sampled,
                **geometric_acceptance(
                    [x["max_error_m"] for x in errors],
                    carrier_sum_sampled,
                    inputs.get("accuracy_target", "instantaneous"),
                ),
            }
            grid_records.append(record)
            partial_results = records + [
                {"case": case["name"], "design": design, "grids": grid_records, "status": "running"}
            ]
            (output / "results.json").write_text(json.dumps(partial_results, indent=2) + "\n")
            write_report(output, partial_results)
            np.savez_compressed(
                case_dir / f"{label}-state.npz",
                coefficients_m=solved,
                source_velocity_m_s=drive,
                pressure_pa=achieved_wave.solution[:, 0],
                pressure_nodes_m=achieved_wave.basis.doflocs * cfg.radius_m,
            )
            print(label, errors, diagnostics, flush=True)
            r = np.linspace(0, cfg.radius_m, 2001)
            for j in range(2):
                height = forward_space.evaluate(solved[j], r)
                axes[j, 0].plot(r * 1e3, height * 1e6, label=label)
                pupil = r <= cfg.clear_radius_m
                axes[j, 1].plot(
                    r[pupil] * 1e3,
                    (height[pupil] - targets[j].evaluate(r[pupil])) * 1e9,
                    label=label,
                )
        for j in range(2):
            axes[j, 0].set(
                ylabel=f"{'Front' if j == 0 else 'Back'} displacement (µm)", xlabel="Radius (mm)"
            )
            axes[j, 1].axhline(10, color="black", linestyle=":")
            axes[j, 1].axhline(-10, color="black", linestyle=":")
            axes[j, 1].set(ylabel="Target height error (nm)", xlabel="Radius (mm)")
            axes[j, 0].legend()
            axes[j, 1].legend()
        fig.suptitle(f"{case['name']}: same physical source commands on every grid")
        fig.savefig(case_dir / "surfaces-and-errors.png", dpi=180)
        fig.clear()
        records.append({"case": case["name"], "design": design, "grids": grid_records})
        (case_dir / "progress.json").write_text(json.dumps({"status": "complete"}) + "\n")
        (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    write_report(output, records)


def write_report(output, records):
    lines = [
        "# Simultaneous Cartesian front/back stationary study",
        "",
        "Model: sealed three-layer cylinder; inviscid harmonic transmission; nonlinear",
        "capillarity and gravity; passive source ports. All material constants are",
        "hypothetical. The commands and physical supports are fixed under refinement.",
        "",
        "| Case | Grid | Front max (nm) | Back max (nm) | Front/back carrier (nm) |",
        "|---|---|---:|---:|---|",
    ]
    for record in records:
        for row in record["grids"]:
            errors = [x["max_error_m"] * 1e9 for x in row["faces"]]
            carrier = [x * 1e9 for x in row["acoustics"]["pupil_sampled_carrier_height_peak_m"]]
            lines.append(
                f"| {record['case']} | {row['grid']} | {errors[0]:.4f} | "
                f"{errors[1]:.4f} | {carrier[0]:.3f} / {carrier[1]:.3f} |"
            )
    lines += [
        "",
        "## Optimization and completion status",
        "",
    ]
    for record in records:
        design = record["design"]
        lines.append(
            f"- {record['case']}: campaign {record.get('status', 'complete')}; "
            f"optimizer success={design['success']}. {design.get('message', '')}"
        )
    lines += [
        "",
        "Unconverged candidates are forward-tested iterates, not certified optima.",
        "Failure to attain the target is not proof of physical unreachability.",
        "",
        "These are stationary mean surfaces. No formation trajectory, coupled",
        "stability, thermoviscous streaming, thermal feedback, material uncertainty,",
        "or experimental validation is supplied by this campaign. A carrier amplitude",
        "above 10 nm rejects strict instantaneous accuracy even if mean fitting passes.",
        "It does not reject cycle-mean accuracy. The declared metric in each new grid",
        "record determines its geometric gate; historical runs retain their original scope.",
        "The source inverse uses a finite surface space; field AND surface refinements",
        "must converge before interpreting source refinement. The independent maximum",
        "search is numerical, not an interval-arithmetic bound.",
        "",
        "`validation.json` contains the analytic three-layer scattering and Bessel",
        "capillary-compliance checks. `provenance/` captures executed code and packages.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
