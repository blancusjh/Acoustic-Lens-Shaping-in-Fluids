"""Nonlinear check of the whole atlas on the common machine.

For every lens that fits the common cell: the convex capability (power and
per-band peak-pressure caps), refinement to one tone per band, then the full
coupled quasi-static start-up branch from the acoustics-off meniscus at the
lens's fill volumes, with all bands applied together. The bands per lens are
the atlas classification (one tone at 1.5 MHz, or 0.75 + 1.5 MHz).
Usage: atlas_nonlinear.py OUT.jsonl SHARD NSHARDS
"""

import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import lens_cell

from acoustic_freeform.acoustics.multiband import wave_for
from acoustic_freeform.apparatus.sources import modal_source_matrix
from acoustic_freeform.forward.branch import startup_branch
from acoustic_freeform.inverse.capability import (
    Tone,
    best_tolerance,
    tone_per_band_refinement,
)
from acoustic_freeform.mechanics.demand import demanded_force
from acoustic_freeform.mechanics.surface import DualSurface

out, shard, shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
CAP_W, CAP_PA = 1.0, 2e6
TWO_BANDS = {6, 7, 30, 31, 33, 34, 42, 58, 60, 61}
grid = list(itertools.product(("-inf", -300.0, -150.0), (60.0, 90.0, 120.0), (10.0, 15.0, 20.0), (-0.5, 0.0, 0.5)))
sigmas = [0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
for index, (d0, d2, t, shape) in enumerate(grid):
    if index % shards != shard:
        continue
    singlet = {"indices": [1.0, 1.7, 1.0], "object_m": d0 if isinstance(d0, str) else d0 * 1e-3,
               "image_m": d2 * 1e-3, "thickness_m": t * 1e-3, "shape_factor": shape, "scale": 1 / 14,
               "pin_radius_m": 0.00202, "clear_radius_m": 0.002, "windows_m": [-0.0008, 0.0022286]}
    row = {"index": index, "object_mm": d0, "image_mm": d2, "thickness_mm": t, "shape_factor": shape}
    started = time.time()
    try:
        cfg, faces, _ = lens_cell("fig9b", singlet=singlet)
        levels = cfg.levels_m
        r = np.linspace(0, cfg.radius_m, 801)
        h = [levels[j + 1] + faces[j].evaluate(r) for j in range(2)]
        if not (levels[0] + 1e-4 < levels[1] < levels[2] < levels[3] - 1e-4) or np.min(h[1] - h[0]) < 5e-5 \
                or np.min(h[0]) < levels[0] + 1e-4 or np.max(h[1]) > levels[3] - 1e-4:
            raise ValueError("lens does not fit the common cell")
        bands = [0.75e6, 1.5e6] if index in TWO_BANDS else [1.5e6]
        space = DualSurface(cfg, allow_volume_offsets=True)
        q, demand, stiffness = demanded_force(space, faces)
        pupil = np.linspace(0, cfg.clear_radius_m, 41)
        observe = np.kron(np.eye(2), space.basis(pupil)) @ np.linalg.inv(stiffness)
        tones, modes = [], None
        for f in bands:
            band = replace(cfg, frequency_hz=f)
            modes, _ = modal_source_matrix(band, {"bottom": 8, "top": 8}, 10)
            tones.append(Tone(band, space, q, modes))
        convex = best_tolerance(tones, observe, demand, CAP_W, pressure_cap_pa=CAP_PA)
        starts = [c["amplitudes"][:, 0] for c in convex["tones"]]
        refined = tone_per_band_refinement(tones, observe, demand, starts, CAP_W, CAP_PA)
        drive = np.array([modes @ c for c in refined["amplitudes"]])
        wave = wave_for(cfg, space, bands if len(bands) > 1 else None, linear_solver="static_condensed")
        start = space.unforced_equilibrium()
        branch = startup_branch(wave, space, drive if len(bands) > 1 else drive[0], sigmas, initial=start)
        end = branch["points"][-1]["coefficients_m"]
        fine = np.linspace(0, cfg.clear_radius_m, 401)
        error = [float(np.max(abs(space.height(j, end[j], fine) - faces[j].evaluate(fine)))) for j in range(2)]
        meniscus = [float(np.max(abs(space.height(j, start[j], fine) - faces[j].evaluate(fine)))) for j in range(2)]
        row.update({
            "status": "checked", "bands_hz": bands,
            "convex_pupil_error_m": convex["pupil_max_error_m"],
            "refined_frozen_pupil_error_m": refined["pupil_max_error_m"],
            "power_w": refined["power_w"], "band_peak_pressure_pa": refined["peak_pressure_pa"],
            "acoustics_off_pupil_error_m": meniscus,
            "branch_reached_end": branch["reached_end"], "branch_stop_reason": branch["stop_reason"],
            "fold_margin_min": min(p["static_min_real"] for p in branch["points"]),
            "nonlinear_pupil_error_m": error,
        })
    except Exception as error:  # noqa: BLE001 - every failure is recorded as a row
        row.update({"status": "skipped" if "fit" in str(error) else "failed",
                    "reason": f"{type(error).__name__}: {str(error)[:200]}"})
    row["seconds"] = round(time.time() - started, 1)
    print(json.dumps(row), flush=True)
    with open(out, "a") as handle:
        handle.write(json.dumps(row) + "\n")
