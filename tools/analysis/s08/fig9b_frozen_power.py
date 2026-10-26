"""Frozen-start power needed to close the Fig. 9b meniscus residual, versus frequency (diagnostic only)."""
# ruff: noqa: SIM115, RUF046  (exploratory scan scripts; closures are evaluated in-loop)
import json
import sys
from dataclasses import replace

import numpy as np

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.apparatus.sources import modal_source_matrix
from acoustic_freeform.inverse.faithful import FaithfulProblem, ModalOperators, frozen_start
from acoustic_freeform.mechanics.surface import targets_from_record

run = "artifacts/studies/S08-published-singlet/fig9b-2026-10-26/caps-1w-2mpa"
record = json.load(open(f"{run}/config.json"))
base = DualConfig(**record["apparatus"])
f = float(sys.argv[1]) * 1e6
ratio = max(1.0, f / 1.5e6)
cfg = replace(base, frequency_hz=f, radial_cells=int(round(32 * ratio)), cells_per_layer=int(round(12 * ratio)))
modes, labels = modal_source_matrix(cfg, record["modes"]["window_modes"], record["modes"]["wall_modes"])
targets = targets_from_record(cfg, record)
pupil = np.linspace(0, cfg.clear_radius_m, 41)
problem = FaithfulProblem(cfg, modes, targets, pupil, 1e-8, [0, 1.0])
rng = np.random.default_rng(1)
starts = frozen_start(problem, {"source_power_w": 1e12, "peak_pressure_pa": 1e15}, 3, rng)
ops = ModalOperators(problem.wave, problem.start, modes, problem.pressure_samples)
best = starts[0]
c = best["coefficients"]
meniscus = np.max(abs(problem.target_heights - problem.pupil_heights(problem.start)))
print(json.dumps({"frequency_mhz": f / 1e6, "mesh": [cfg.radial_cells, cfg.cells_per_layer],
                  "meniscus_pupil_error_m": float(meniscus),
                  "frozen_residual_m": best["frozen_pupil_defect_scaled"] * 1e-8,
                  "power_w": ops.source_power(c), "peak_pressure_pa": ops.peak_pressure(c)}), flush=True)
