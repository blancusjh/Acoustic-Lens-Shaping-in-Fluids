"""Budget source layouts: one window driven by opaque piezo rings, the other plain glass.

Opaque transducers are acceptable when the lens is only measured after curing,
but one face must stay UV-transparent for curing. A plain glass window is
undriven and nearly rigid (admittance factor about 0.1). For each layout and
lens: the convex capability (1 W, 2 MPa) and the one-tone refinement at 1.5 MHz.
Usage: budget_layouts.py OUT.jsonl
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
GLASS = 0.1
grid = list(itertools.product(("-inf", -300.0, -150.0), (60.0, 90.0, 120.0), (10.0, 15.0, 20.0), (-0.5, 0.0, 0.5)))
SAMPLE = [9, 12, 15, 22, 36, 40, 45, 48, 66, 72, 77, 79]  # single-band atlas lenses across the grid
LENSES = [("fig9b", None), ("fig7a", None)] + [(f"atlas{i}", i) for i in SAMPLE]
# (name, driven windows, rings per window, wall bands per layer, glass windows)
LAYOUTS = [
    ("reference both driven", ("bottom", "top"), 8, 4, ()),
    ("bottom driven, top glass", ("bottom",), 8, 4, ("top",)),
    ("top driven, bottom glass", ("top",), 8, 4, ("bottom",)),
    ("bottom driven, top glass, 4 rings 2 bands", ("bottom",), 4, 2, ("top",)),
    ("top driven, bottom glass, 4 rings 2 bands", ("top",), 4, 2, ("bottom",)),
]
for name, index in LENSES:
    if index is None:
        base, faces, _ = lens_cell(name)
    else:
        d0, d2, t, shape = grid[index]
        singlet = {"indices": [1.0, 1.7, 1.0], "object_m": d0 if isinstance(d0, str) else d0 * 1e-3,
                   "image_m": d2 * 1e-3, "thickness_m": t * 1e-3, "shape_factor": shape, "scale": 1 / 14,
                   "pin_radius_m": 0.00202, "clear_radius_m": 0.002, "windows_m": [-0.0008, 0.0022286]}
        base, faces, _ = lens_cell("fig9b", singlet=singlet)
    for label, driven, rings, bands, glass in LAYOUTS:
        overrides = {"radial_ports": rings, "side_ports_per_layer": bands, "frequency_hz": 1.5e6}
        for window in glass:
            overrides[f"{window}_admittance_factor"] = GLASS
        cfg = replace(base, **overrides)
        space = DualSurface(cfg, allow_volume_offsets=True)
        q, demand, stiffness = demanded_force(space, faces)
        observe = np.kron(np.eye(2), space.basis(np.linspace(0, cfg.clear_radius_m, 41))) @ np.linalg.inv(stiffness)
        window_modes = {w: (min(8, rings) if w in driven else 0) for w in ("bottom", "top")}
        modes, _ = modal_source_matrix(cfg, window_modes, min(10, 3 * bands))
        tone = Tone(cfg, space, q, modes)
        convex = best_tolerance([tone], observe, demand, 1.0, pressure_cap_pa=2e6)
        refined = tone_per_band_refinement([tone], observe, demand, [convex["tones"][0]["amplitudes"][:, 0]], 1.0, 2e6)
        row = {"lens": name, "layout": label, "driven_channels": len(driven) * rings + 3 * bands,
               "modes": modes.shape[1], "convex_pupil_error_m": convex["pupil_max_error_m"],
               "single_tone_pupil_error_m": refined["pupil_max_error_m"], "power_w": refined["power_w"],
               "peak_pressure_pa": refined["peak_pressure_pa"]}
        print(json.dumps(row), flush=True)
        with open(out, "a") as handle:
            handle.write(json.dumps(row) + "\n")
