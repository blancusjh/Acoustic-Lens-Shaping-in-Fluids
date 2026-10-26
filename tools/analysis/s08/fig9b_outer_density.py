"""Fig. 9b cell: acoustics-off meniscus error versus the density of the liquid below / above the resin.

Each face's capillary-gravity shape depends only on its own density step
drho = rho_below - rho_above, so the outer liquids can be chosen per face.
"""
# ruff: noqa: B023, SIM115  (exploratory scan scripts; closures are evaluated in-loop)
import json
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface, targets_from_record

run = "artifacts/studies/S08-published-singlet/fig9b-2026-10-26/caps-1w-2mpa"
record = json.load(open(f"{run}/config.json"))
cfg0 = DualConfig(**record["apparatus"])
targets = targets_from_record(cfg0, record)
r = np.linspace(0, cfg0.clear_radius_m, 201)
rho = list(cfg0.density_kg_m3)
print("densities", rho, "tension", cfg0.surface_tension_n_m)
def face_error(j, outer, volume):
    dens = list(rho); dens[0 if j == 0 else 2] = outer
    off = [0.0, 0.0]; off[j] = volume
    S = DualSurface(replace(cfg0, density_kg_m3=tuple(dens), volume_offsets_m3=tuple(off)), allow_volume_offsets=True)
    q = S.unforced_equilibrium()
    return float(np.max(abs(S.height(j, q[j], r) - targets[j].evaluate(r))))
for j, name in enumerate(("front (liquid below)", "back (liquid above)")):
    v0 = targets[j].volume_offset()
    rows = []
    for outer in (700, 800, 900, 997, 1100, 1231, 1400, 1600, 1800, 2000, 2400, 2800):
        fit = minimize_scalar(lambda v: face_error(j, outer, v), bracket=(0.97 * v0, v0), tol=1e-14)
        rows.append((outer, fit.fun))
        print(f"{name}: rho = {outer:5d} kg/m3 (drho = {(outer - rho[1]) if j == 0 else (rho[1] - outer):+5.0f}): best meniscus error {fit.fun*1e9:7.1f} nm", flush=True)
