"""Minimum source hardware: capability versus channel count on the common machine.

The model's 32 rings per window and 16 wall bands per layer (112 channels) are
a discretization of the continuous source, not a requirement. For a set of
lenses, this repeats the capability program and one-tone-per-band refinement
with fewer, wider source regions (and no more modes than regions), at the
common caps. Usage: channel_study.py OUT.jsonl
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
    tone_per_band_refinement,
)
from acoustic_freeform.mechanics.demand import demanded_force
from acoustic_freeform.mechanics.surface import DualSurface

out = sys.argv[1]
grid = list(itertools.product(("-inf", -300.0, -150.0), (60.0, 90.0, 120.0), (10.0, 15.0, 20.0), (-0.5, 0.0, 0.5)))
# (name, atlas index or None, bands): published lenses plus atlas lenses of each kind.
LENSES = [("fig9b", None, [1.5e6]), ("fig7a", None, [1.5e6]), ("atlas22", 22, [1.5e6]),
          ("atlas45", 45, [1.5e6]), ("atlas7", 7, [0.75e6, 1.5e6])]
# (rings per window, wall bands per layer)
LAYOUTS = [(32, 16), (16, 8), (12, 6), (8, 4), (6, 3), (4, 2)]
for name, index, bands in LENSES:
    if index is None:
        base_cfg, faces, _ = lens_cell(name)
    else:
        d0, d2, t, shape = grid[index]
        singlet = {"indices": [1.0, 1.7, 1.0], "object_m": d0 if isinstance(d0, str) else d0 * 1e-3,
                   "image_m": d2 * 1e-3, "thickness_m": t * 1e-3, "shape_factor": shape, "scale": 1 / 14,
                   "pin_radius_m": 0.00202, "clear_radius_m": 0.002, "windows_m": [-0.0008, 0.0022286]}
        base_cfg, faces, _ = lens_cell("fig9b", singlet=singlet)
    for rings, bands_per_layer in LAYOUTS:
        cfg = replace(base_cfg, radial_ports=rings, side_ports_per_layer=bands_per_layer)
        space = DualSurface(cfg, allow_volume_offsets=True)
        q, demand, stiffness = demanded_force(space, faces)
        observe = np.kron(np.eye(2), space.basis(np.linspace(0, cfg.clear_radius_m, 41))) @ np.linalg.inv(stiffness)
        window_modes, wall_modes = min(8, rings), min(10, 3 * bands_per_layer)
        tones = []
        for f in bands:
            band = replace(cfg, frequency_hz=f)
            modes, _ = modal_source_matrix(band, {"bottom": window_modes, "top": window_modes}, wall_modes)
            tones.append(Tone(band, space, q, modes))
        convex = best_tolerance(tones, observe, demand, 1.0, pressure_cap_pa=2e6)
        starts = [c["amplitudes"][:, 0] for c in convex["tones"]]
        refined = tone_per_band_refinement(tones, observe, demand, starts, 1.0, 2e6)
        row = {"lens": name, "bands_hz": bands, "rings_per_window": rings, "wall_bands_per_layer": bands_per_layer,
               "channels": 2 * rings + 3 * bands_per_layer, "modes": tones[0].kernels.shape[1],
               "convex_pupil_error_m": convex["pupil_max_error_m"],
               "single_tone_pupil_error_m": refined["pupil_max_error_m"], "power_w": refined["power_w"],
               "peak_pressure_pa": refined["peak_pressure_pa"]}
        print(json.dumps(row), flush=True)
        with open(out, "a") as handle:
            handle.write(json.dumps(row) + "\n")
