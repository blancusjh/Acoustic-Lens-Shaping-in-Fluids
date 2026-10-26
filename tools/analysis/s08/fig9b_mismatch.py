"""Fig. 9b: profile of the acoustics-off meniscus mismatch and the Laplace pressure that would correct it."""
# ruff: noqa: SIM115  (exploratory scan scripts; closures are evaluated in-loop)
import json

import numpy as np

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface, targets_from_record

run = "artifacts/studies/S08-published-singlet/fig9b-2026-10-26/caps-1w-2mpa"
record = json.load(open(f"{run}/config.json"))
cfg = DualConfig(**record["apparatus"])
S = DualSurface(cfg, allow_volume_offsets=True)
targets = targets_from_record(cfg, record)
q0 = S.unforced_equilibrium()
r = np.linspace(0, cfg.radius_m, 21)
for j in range(2):
    d = targets[j].evaluate(r) - S.height(j, q0[j], r)
    print(f"face {j} mismatch (nm) vs r/R:", " ".join(f"{x:+.0f}" for x in d * 1e9))
# Required generalized pressure: Laplace pressure of the mismatch, gamma * curvature difference
gamma = cfg.surface_tension_n_m
rr = np.linspace(1e-6, cfg.radius_m, 2001)
for j in range(2):
    d = targets[j].evaluate(rr) - S.height(j, q0[j], rr)
    lap = np.gradient(rr * np.gradient(d, rr), rr) / rr
    p = gamma[j] * lap
    print(f"face {j}: required Laplace pressure jump (Pa) at r/R=0,.25,.5,.75,.9,.95,1:",
          [f"{p[int(x*2000)]:+.3f}" for x in (0.0005, .25, .5, .75, .9, .95, 1.0)], " mean", f"{np.average(p, weights=rr):+.3f}")
