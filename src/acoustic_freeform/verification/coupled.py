"""Fixed physical drive, independently re-solved surface at each resolution."""

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from acoustic_freeform.lens.config import LensConfig
from acoustic_freeform.lens.optics import trace_surface
from acoustic_freeform.lens.surface import SurfaceSpace
from acoustic_freeform.lens.validation import fixed_drive_equilibrium
from acoustic_freeform.provenance import capture_execution


def coupled_study(
    source,
    destination,
    levels=((64, 96, 32), (64, 96, 48), (80, 128, 48), (80, 128, 64)),
    method="hybr",
):
    source, out = Path(source), Path(destination)
    if (out / "report.json").exists():
        raise FileExistsError(f"Completed study exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    execution = capture_execution(out)
    original = LensConfig(**json.loads((source / "configuration.json").read_text()))
    data = np.load(source / "stationary.npz")
    source_space = SurfaceSpace(original)
    source_volume = float(source_space.volume_vector @ data["coefficients"])
    drive = data["drive_m_s"].copy()
    previous = data["coefficients"].copy()
    records = []
    for nr, nz, modes in levels:
        cfg = replace(
            original,
            geometry_mapping="exact_graph",
            mesh_radial_distribution="rim_clustered",
            align_array_mesh=True,
            acoustic_quadrature_order=10,
            acoustic_order=4,
            mesh_radial=nr,
            mesh_vertical=nz,
            surface_modes=modes,
        )
        space = SurfaceSpace(cfg)
        seed = np.zeros(modes)
        seed[: min(modes, len(previous))] = previous[:modes]
        seed += (
            space.volume_vector
            * (source_volume - space.volume_vector @ seed)
            / np.dot(space.volume_vector, space.volume_vector)
        )
        name = f"P4-{nr}-{nz}-{modes}"
        print(f"coupled resolution: {name}", flush=True)
        start = time.perf_counter()
        c, residual = fixed_drive_equilibrium(space, drive, seed, method=method)
        optics = trace_surface(space, c)
        pupil = optics["pupil_r_m"] / cfg.radius_m
        row = {
            "case": name,
            "configuration": cfg.as_dict(),
            "seconds": time.perf_counter() - start,
            "equilibrium_residual": residual,
            "equilibrium_method": method,
            "dimensionless_volume_error": float(space.volume_vector @ c - source_volume),
            "correction_tolerance_m": 1e-10,
            "ray_rms_at_target_m": optics["rms_spot_at_target_m"],
            "ray_rms_at_best_focus_m": optics["rms_spot_at_best_focus_m"],
            "opd_rms_m": optics["opd_rms_m"],
            "height_change_from_source_pupil_rms_m": float(
                cfg.radius_m
                * np.sqrt(
                    np.mean(
                        (
                            space.evaluate(c, pupil)
                            - source_space.evaluate(data["coefficients"], pupil)
                        )
                        ** 2
                    )
                )
            ),
            "height_change_from_previous_pupil_rms_m": float(
                cfg.radius_m * np.sqrt(np.mean(space.evaluate(c - seed, pupil) ** 2))
            ),
        }
        np.savez_compressed(out / f"{name}.npz", coefficients=c, drive_m_s=drive)
        records.append(row)
        (out / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        print({k: v for k, v in row.items() if k != "configuration"}, flush=True)
        previous = c
    report = {
        "kind": "fixed_drive_coupled_resolution_study",
        "scope": "Identical physical wall velocities and optical target; geometry and field re-solved without re-optimization at each resolution. No dynamical stability claim.",
        "source": str(source.resolve()),
        "source_state_sha256": hashlib.sha256((source / "stationary.npz").read_bytes()).hexdigest(),
        "execution": execution,
        "completion_source_sha256": {
            str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for directory in [Path(__file__).parents[1] / "lens", Path(__file__).parent]
            for p in sorted(directory.glob("*.py"))
        },
        "cases": records,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
