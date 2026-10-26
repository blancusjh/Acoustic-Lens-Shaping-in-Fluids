"""Reachability of the Fig. 9b meniscus correction by a rank-r (multi-tone) drive, frozen compliance.

Forces of tones separated by far more than the creeping-flow bandwidth add in the
time average, so a rank-r drive realizes F = sum_i F(c_i). Uncapped: this asks
whether the target is in the reachable cone at all, and at what power.
"""
# ruff: noqa: B023, SIM115, RUF046  (exploratory scan scripts; closures are evaluated in-loop)
import json
import sys
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.apparatus.sources import modal_source_matrix
from acoustic_freeform.inverse.faithful import (
    FaithfulProblem,
    ModalOperators,
    complex_coefficients,
)
from acoustic_freeform.mechanics.surface import targets_from_record

run = "artifacts/studies/S08-published-singlet/fig9b-2026-10-26/caps-1w-2mpa"
record = json.load(open(f"{run}/config.json"))
cfg = DualConfig(**record["apparatus"])
if len(sys.argv) > 1:
    f = float(sys.argv[1]) * 1e6; ratio = max(1.0, f / 1.5e6)
    cfg = replace(cfg, frequency_hz=f, radial_cells=int(round(32 * ratio)), cells_per_layer=int(round(12 * ratio)))
modes, labels = modal_source_matrix(cfg, record["modes"]["window_modes"], record["modes"]["wall_modes"])
problem = FaithfulProblem(cfg, modes, targets_from_record(cfg, record), np.linspace(0, cfg.clear_radius_m, 41), 1e-8, [0, 1])
space, q = problem.space, problem.start
_, K, _ = space.mechanics(q)
ops = ModalOperators(problem.wave, q, modes, problem.pressure_samples)
C = np.kron(np.eye(2), problem.pupil_basis) @ np.linalg.inv(K)
target = (problem.target_heights - problem.pupil_heights(q)).ravel()
m = modes.shape[1]
rng = np.random.default_rng(3)
# scale: power of a unit-norm command
unit = ops.source_power(np.ones(m) / np.sqrt(m))
for rank in (1, 2, 4, 8, 16):
    best = None
    for seed in range(3):
        x0 = rng.normal(size=(rank, 2 * m)) * 1e-3
        def fun(x):
            X = x.reshape(rank, 2 * m)
            return (C @ sum(ops.force(complex_coefficients(v)) for v in X) - target) * 1e9
        def jac(x):
            X = x.reshape(rank, 2 * m)
            return np.hstack([C @ ops.force_jacobian(complex_coefficients(v)) for v in X]) * 1e9
        fit = least_squares(fun, x0.ravel(), jac=jac, max_nfev=3000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        X = fit.x.reshape(rank, 2 * m)
        res = float(np.max(abs(fun(fit.x))))
        power = sum(ops.source_power(complex_coefficients(v)) for v in X)
        peak = max(ops.peak_pressure(complex_coefficients(v)) for v in X)
        if best is None or res < best[0]:
            best = (res, power, peak)
    print(json.dumps({"frequency_mhz": cfg.frequency_hz / 1e6, "rank": rank, "max_pupil_residual_nm": best[0],
                      "total_power_w": best[1], "max_tone_peak_pressure_pa": best[2]}), flush=True)
