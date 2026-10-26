"""Low-rank capability: minimum power at a fixed tolerance (trace heuristic) vs best tolerance.

Minimizing tr(W X) at a pupil tolerance favours low-rank X, whose leading
eigenvectors are better single tones per band than those of the best-tolerance
solution. Usage: low_rank_check.py LENS_INDEX BANDS_MHZ TOLERANCE_NM CAP_W
"""

import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import lens_cell

from acoustic_freeform.apparatus.sources import modal_source_matrix
from acoustic_freeform.inverse.capability import (
    Tone,
    best_tolerance,
    minimum_power,
    tone_per_band_refinement,
)
from acoustic_freeform.mechanics.demand import demanded_force
from acoustic_freeform.mechanics.surface import DualSurface

index, bands = int(sys.argv[1]), [float(f) for f in sys.argv[2].split(",")]
tolerance, cap = float(sys.argv[3]) * 1e-9, float(sys.argv[4])
grid = list(itertools.product(("-inf", -300.0, -150.0), (60.0, 90.0, 120.0), (10.0, 15.0, 20.0), (-0.5, 0.0, 0.5)))
d0, d2, t, shape = grid[index]
singlet = {"indices": [1.0, 1.7, 1.0], "object_m": d0 if isinstance(d0, str) else d0 * 1e-3, "image_m": d2 * 1e-3,
           "thickness_m": t * 1e-3, "shape_factor": shape, "scale": 1 / 14, "pin_radius_m": 0.00202,
           "clear_radius_m": 0.002, "windows_m": [-0.0008, 0.0022286]}
cfg, faces, _ = lens_cell("fig9b", singlet=singlet)
space = DualSurface(cfg, allow_volume_offsets=True)
q, demand, stiffness = demanded_force(space, faces)
observe = np.kron(np.eye(2), space.basis(np.linspace(0, cfg.clear_radius_m, 41))) @ np.linalg.inv(stiffness)
tones = []
for f in bands:
    ratio = max(1.0, f / 1.5)
    band = replace(cfg, frequency_hz=f * 1e6, radial_cells=round(32 * ratio), cells_per_layer=round(12 * ratio))
    modes, _ = modal_source_matrix(band, {"bottom": 8, "top": 8}, 10)
    tones.append(Tone(band, space, q, modes))
for name, solve in (
    ("best_tolerance_pressure_capped", lambda: best_tolerance(tones, observe, demand, cap, pressure_cap_pa=2e6)),
    ("minimum_power_pressure_capped", lambda: minimum_power(tones, observe, demand, tolerance, pressure_cap_pa=2e6)),
):
    convex = solve()
    if "tones" not in convex:
        print(json.dumps({"lens": index, "method": name, "status": convex["status"]}), flush=True)
        continue
    starts = [c["amplitudes"][:, 0] for c in convex["tones"]]
    refined = tone_per_band_refinement(tones, observe, demand, starts, cap, 2e6)
    print(json.dumps({
        "lens": index, "bands_mhz": bands, "method": name, "convex_error_nm": convex["pupil_max_error_m"] * 1e9,
        "convex_power_w": convex["total_power_w"], "ranks": [c["rank"] for c in convex["tones"]],
        "leading_eigen_fraction": [c["eigen_power_fraction"][0] for c in convex["tones"]],
        "refined_error_nm": refined["pupil_max_error_m"] * 1e9, "refined_power_w": refined["power_w"],
        "refined_peak_mpa": refined["peak_pressure_pa"] / 1e6,
    }), flush=True)
