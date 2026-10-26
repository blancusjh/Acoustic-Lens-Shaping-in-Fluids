"""Atlas: which stigmatic singlets (SOSLs) one common machine can hold.

Every lens is placed in the same cell (radius 2.02 mm, windows at -0.8 and
+2.2286 mm, water / NOA 61 / water) with the same 26-mode source at one
frequency; only the pin-ring heights follow the lens rim (one mechanical
setting) and the fill pumps follow its volumes. For each lens: the no-field
error, the load demand range and the convex multi-tone capability within a
power cap, plus the single-tone refinement. Lenses are the Silva-Lora and
Torres family n = 1 / 1.7 / 1, given in the paper's units and scaled 1/14.
Usage: atlas.py OUT.jsonl SHARD NSHARDS [FREQ_MHZ] [CAP_W]
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import lens_cell

from acoustic_freeform.apparatus.sources import modal_source_matrix
from acoustic_freeform.inverse.capability import (
    Tone,
    best_tolerance,
    single_tone_refinement,
)
from acoustic_freeform.mechanics.demand import demanded_force, demanded_traction
from acoustic_freeform.mechanics.surface import DualSurface

out, shard, shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
f_mhz = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
cap = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
grid = list(itertools.product(
    ("-inf", -300.0, -150.0),   # object distance (mm, paper units)
    (60.0, 90.0, 120.0),        # image distance from the first vertex (mm)
    (10.0, 15.0, 20.0),         # centre thickness (mm)
    (-0.5, 0.0, 0.5),           # shape factor sigma
))
for index, (d0, d2, t, shape) in enumerate(grid):
    if index % shards != shard:
        continue
    singlet = {"indices": [1.0, 1.7, 1.0], "object_m": d0 if isinstance(d0, str) else d0 * 1e-3,
               "image_m": d2 * 1e-3, "thickness_m": t * 1e-3, "shape_factor": shape, "scale": 1 / 14,
               "pin_radius_m": 0.00202, "clear_radius_m": 0.002, "windows_m": [-0.0008, 0.0022286]}
    row = {"index": index, "object_mm": d0, "image_mm": d2, "thickness_mm": t, "shape_factor": shape}
    try:
        cfg, faces, _ = lens_cell("fig9b", singlet=singlet, overrides={"frequency_hz": f_mhz * 1e6})
        levels = cfg.levels_m
        if not (levels[0] + 1e-4 < levels[1] < levels[2] < levels[3] - 1e-4):
            raise ValueError("pinned rims outside the common windows or crossing")
        r = np.linspace(0, cfg.radius_m, 801)
        h = [levels[j + 1] + faces[j].evaluate(r) for j in range(2)]
        if np.min(h[1] - h[0]) < 5e-5 or np.min(h[0]) < levels[0] + 1e-4 or np.max(h[1]) > levels[3] - 1e-4:
            raise ValueError("lens does not fit the common cell (thickness or window clearance)")
        space = DualSurface(cfg, allow_volume_offsets=True)
        q, demand, stiffness = demanded_force(space, faces)
        pupil = np.linspace(0, cfg.clear_radius_m, 41)
        observe = np.kron(np.eye(2), space.basis(pupil)) @ np.linalg.inv(stiffness)
        demand_range = []
        for j, face in enumerate(faces):
            drho_g = (cfg.density_kg_m3[j] - cfg.density_kg_m3[j + 1]) * cfg.gravity_m_s2
            demand_range.append(float(np.ptp(demanded_traction(face, cfg.surface_tension_n_m[j], drho_g, r[1:]))))
        modes, _ = modal_source_matrix(cfg, {"bottom": 8, "top": 8}, 10)
        tone = Tone(cfg, space, q, modes)
        convex = best_tolerance([tone], observe, demand, cap)
        single = single_tone_refinement(tone, observe, demand, convex["tones"][0]["amplitudes"][:, 0], cap)
        sags = [float(faces[j].evaluate(np.array([0.0]))[0]) for j in range(2)]
        row.update({
            "status": convex["status"], "rims_m": [levels[1], levels[2]], "vertex_sag_from_rim_m": sags,
            "no_field_pupil_error_m": float(np.max(abs(observe @ demand))), "demand_range_pa": demand_range,
            "convex_pupil_error_m": convex["pupil_max_error_m"], "convex_power_w": convex["total_power_w"],
            "convex_rank": convex["tones"][0]["rank"],
            "single_tone_pupil_error_m": single["pupil_max_error_m"], "single_tone_power_w": single["power_w"],
            "single_tone_peak_pressure_pa": single["peak_pressure_pa"],
        })
    except Exception as error:  # noqa: BLE001 - every failure is recorded as a row
        row.update({"status": "skipped", "reason": f"{type(error).__name__}: {str(error)[:160]}"})
    print(json.dumps(row), flush=True)
    with open(out, "a") as handle:
        handle.write(json.dumps(row) + "\n")
