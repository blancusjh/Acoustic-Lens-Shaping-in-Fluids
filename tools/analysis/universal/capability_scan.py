"""Demand versus capability of one common source on the common water cell.

For each lens, the load demand d (mechanics/demand.py) at the exact target state
and the convex multi-tone capability (inverse/capability.py) of a source design:
a set of frequency bands, each driven through the same continuous modal basis.
Usage: capability_scan.py OUT.jsonl LENS "f1,f2,..." WINDOW_MODES WALL_MODES CAP_W [CAP_W ...]
"""

import json
import sys
import time
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
from acoustic_freeform.mechanics.demand import demanded_force
from acoustic_freeform.mechanics.surface import DualSurface

out, lens, frequencies = sys.argv[1], sys.argv[2], [float(f) * 1e6 for f in sys.argv[3].split(",")]
window_modes, wall_modes = int(sys.argv[4]), int(sys.argv[5])
caps = [float(c) for c in sys.argv[6:]]
cfg, faces, _ = lens_cell(lens)
space = DualSurface(cfg, allow_volume_offsets=True)
q, demand, stiffness = demanded_force(space, faces)
pupil = np.linspace(0, cfg.clear_radius_m, 41)
observe = np.kron(np.eye(2), space.basis(pupil)) @ np.linalg.inv(stiffness)
baseline = float(np.max(abs(observe @ demand)))


def pupil_heights(volumes):
    trial = DualSurface(replace(cfg, volume_offsets_m3=tuple(volumes)), allow_volume_offsets=True)
    state = trial.prescribed_equilibrium(demand, initial=q)
    return np.concatenate([trial.height(j, state[j], pupil) for j in range(2)])


# Fill pumps: d(pupil heights)/d(volume) at fixed demanded force, one column per face.
reference = pupil_heights(cfg.volume_offsets_m3)
columns = []
for j in range(2):
    step = 1e-4 * abs(cfg.volume_offsets_m3[j])
    volumes = list(cfg.volume_offsets_m3)
    volumes[j] += step
    columns.append((pupil_heights(volumes) - reference) / step)
volume_columns = np.column_stack(columns)
tones = []
start = time.time()
for f in frequencies:
    ratio = max(1.0, f / 1.5e6)
    tone_cfg = replace(cfg, frequency_hz=f, radial_cells=round(32 * ratio),
                       cells_per_layer=round(12 * ratio))
    modes, labels = modal_source_matrix(tone_cfg, {"bottom": window_modes, "top": window_modes}, wall_modes)
    tones.append(Tone(tone_cfg, space, q, modes))
setup_s = time.time() - start
for cap in caps:
    start = time.time()
    result = best_tolerance(tones, observe, demand, cap, volume_columns=volume_columns)
    row = {
        "lens": lens, "frequencies_mhz": [f / 1e6 for f in frequencies], "window_modes": window_modes,
        "wall_modes": wall_modes, "modes_per_band": tones[0].kernels.shape[1], "power_cap_w": cap,
        "no_field_pupil_error_m": baseline, "status": result["status"],
    }
    if "tones" in result:
        row.update({
            "multitone_pupil_error_m": result["pupil_max_error_m"],
            "total_power_w": result["total_power_w"],
            "volume_trims_m3": result["volume_trims_m3"],
            "bands": [{k: v for k, v in t.items() if k != "amplitudes"} for t in result["tones"]],
        })
        if len(tones) == 1:
            lead = result["tones"][0]["amplitudes"][:, 0]
            single = single_tone_refinement(tones[0], observe, demand, lead, cap)
            row["single_tone"] = {k: v for k, v in single.items() if k != "amplitudes"}
    row["solve_s"] = time.time() - start
    row["operator_setup_s"] = setup_s
    print(json.dumps(row), flush=True)
    with open(out, "a") as handle:
        handle.write(json.dumps(row) + "\n")
