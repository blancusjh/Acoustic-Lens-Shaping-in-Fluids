"""Frozen-target frequency screening; every selected drive needs a coupled solve."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from acoustic_freeform.lens.acoustics import CavityAcoustics
from acoustic_freeform.lens.config import LensConfig
from acoustic_freeform.lens.design import fit_array
from acoustic_freeform.lens.surface import SurfaceSpace
from acoustic_freeform.provenance import capture_execution


def frequency_study(source, destination, frequencies=(4e5, 6e5, 8e5, 1e6, 1.4e6, 1.8e6)):
    source, out = Path(source), Path(destination)
    if (out / "report.json").exists():
        raise FileExistsError(f"Completed study exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    execution = capture_execution(out)
    original = LensConfig(**json.loads((source / "configuration.json").read_text()))
    seed = np.load(source / "stationary.npz")["drive_m_s"]
    records = []
    for frequency in frequencies:
        cfg = replace(
            original, frequency_hz=frequency, viscous_wall_acoustics=True, bulk_streaming=False
        )
        space = SurfaceSpace(cfg)
        target, _ = space.target()
        field = CavityAcoustics(cfg, space).solve_basis(target)
        fit = fit_array(space, field, target, starts=3, initial_drive=seed)
        p = (
            cfg.density_kg_m3
            * 2
            * np.pi
            * frequency
            * cfg.radius_m
            * (field.solutions @ fit.drive_m_s)
        )
        row = {
            "frequency_hz": frequency,
            "configuration": cfg.as_dict(),
            "local_linearized_ray_error_um": fit.predicted_ray_error_m * 1e6,
            "local_linearized_height_error_m": fit.predicted_shape_error_m,
            "peak_pressure_at_dofs_pa": float(abs(p).max()),
            "max_wall_velocity_m_s": float(abs(fit.drive_m_s).max()),
            "attempts": fit.attempts,
            "objective": fit.objective,
        }
        records.append(row)
        case = out / f"f-{int(frequency)}"
        case.mkdir(exist_ok=True)
        (case / "configuration.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
        np.savez_compressed(case / "fit.npz", coefficients=target, drive_m_s=fit.drive_m_s)
        (case / "drive.json").write_text(
            json.dumps(
                {
                    "velocity_real_m_s": fit.drive_m_s.real.tolist(),
                    "velocity_imag_m_s": fit.drive_m_s.imag.tolist(),
                },
                indent=2,
            )
            + "\n"
        )
        (out / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        print({k: v for k, v in row.items() if k not in ("configuration", "attempts")}, flush=True)
    report = {
        "scope": "Frozen Cartesian target, quadratic array fitting with leading viscous wall acoustics. These local linearized error diagnostics are not coupled equilibria or stability results. Bulk streaming is omitted only from this screening stage. Attenuation is the declared constant 5 Np/m scenario at each frequency, not measured frequency-dependent NOA61 data.",
        "cases": records,
        "execution": execution,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
