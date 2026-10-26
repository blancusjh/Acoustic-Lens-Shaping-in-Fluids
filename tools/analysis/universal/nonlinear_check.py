"""Nonlinear check of a capability result: the single-tone command found by the
convex program is driven through the quasi-static start-up branch from the
acoustics-off meniscus (full coupled acoustics, no frozen compliance).
Usage: nonlinear_check.py LENS FREQ_MHZ WINDOW_MODES WALL_MODES CAP_W OUT_DIR
"""

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
    single_tone_refinement,
)
from acoustic_freeform.inverse.faithful import FaithfulProblem
from acoustic_freeform.mechanics.demand import demanded_force
from acoustic_freeform.mechanics.surface import DualSurface

lens, f_mhz, window_modes, wall_modes, cap, out = (
    sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), Path(sys.argv[6]),
)
out.mkdir(parents=True, exist_ok=True)
if lens.startswith("atlas"):
    # "atlasN": lens N of the S09 atlas grid (same order as atlas.py).
    import itertools

    grid = list(itertools.product(("-inf", -300.0, -150.0), (60.0, 90.0, 120.0), (10.0, 15.0, 20.0), (-0.5, 0.0, 0.5)))
    d0, d2, t, shape = grid[int(lens[5:])]
    singlet = {"indices": [1.0, 1.7, 1.0], "object_m": d0 if isinstance(d0, str) else d0 * 1e-3,
               "image_m": d2 * 1e-3, "thickness_m": t * 1e-3, "shape_factor": shape, "scale": 1 / 14,
               "pin_radius_m": 0.00202, "clear_radius_m": 0.002, "windows_m": [-0.0008, 0.0022286]}
    cfg, faces, _ = lens_cell("fig9b", singlet=singlet)
else:
    cfg, faces, _ = lens_cell(lens)
ratio = max(1.0, f_mhz / 1.5)
cfg = replace(cfg, frequency_hz=f_mhz * 1e6, radial_cells=round(32 * ratio), cells_per_layer=round(12 * ratio))
space = DualSurface(cfg, allow_volume_offsets=True)
q, demand, stiffness = demanded_force(space, faces)
pupil = np.linspace(0, cfg.clear_radius_m, 41)
observe = np.kron(np.eye(2), space.basis(pupil)) @ np.linalg.inv(stiffness)
modes, labels = modal_source_matrix(cfg, {"bottom": window_modes, "top": window_modes}, wall_modes)
tone = Tone(cfg, space, q, modes)
convex = best_tolerance([tone], observe, demand, cap)
single = single_tone_refinement(tone, observe, demand, convex["tones"][0]["amplitudes"][:, 0], cap, pressure_cap_pa=2e6)
c = single["amplitudes"]
problem = FaithfulProblem(cfg, modes, faces, pupil, 1e-8, [0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0])
branch = problem.reachable(c)
end = branch["points"][-1]["coefficients_m"]
fine = np.linspace(0, cfg.clear_radius_m, 401)
error = [float(np.max(abs(problem.space.height(j, end[j], fine) - faces[j].evaluate(fine)))) for j in range(2)]
meniscus = problem.start
start_error = [float(np.max(abs(problem.space.height(j, meniscus[j], fine) - faces[j].evaluate(fine)))) for j in range(2)]
summary = {
    "lens": lens, "frequency_mhz": f_mhz, "modes": modes.shape[1], "power_cap_w": cap,
    "convex_multitone_pupil_error_m": convex["pupil_max_error_m"], "convex_rank": convex["tones"][0]["rank"],
    "single_tone_frozen_pupil_error_m": single["pupil_max_error_m"],
    "single_tone_power_w": single["power_w"], "single_tone_peak_pressure_pa": single["peak_pressure_pa"],
    "acoustics_off_pupil_error_m": start_error,
    "branch_reached_end": branch["reached_end"], "branch_stop_reason": branch["stop_reason"],
    "branch_min_static": min(p["static_min_real"] for p in branch["points"]),
    "nonlinear_end_pupil_error_m": error,
}
print(json.dumps(summary), flush=True)
(out / f"{lens}-{f_mhz}mhz-{modes.shape[1]}modes.json").write_text(json.dumps(summary, indent=2) + "\n")
np.savez_compressed(out / f"{lens}-{f_mhz}mhz-{modes.shape[1]}modes.npz", modal_amplitudes=c, end_coefficients_m=end,
                    channel_velocity_m_s=modes @ c)
