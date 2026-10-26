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
def face_error(j, outer, volume, tension):
    dens = list(rho); dens[0 if j == 0 else 2] = outer
    off = [0.0, 0.0]; off[j] = volume
    S = DualSurface(replace(cfg0, density_kg_m3=tuple(dens), volume_offsets_m3=tuple(off), surface_tension_n_m=(tension, tension)), allow_volume_offsets=True)
    q = S.unforced_equilibrium()
    return float(np.max(abs(S.height(j, q[j], r) - targets[j].evaluate(r))))


def scan(j, tensions, densities):
    v0 = targets[j].volume_offset()
    for tension in tensions:
        best = None
        for outer in densities:
            def err(v):
                try:
                    return face_error(j, outer, v, tension)
                except (ValueError, RuntimeError):
                    return 1.0
            fit = minimize_scalar(err, bounds=sorted((0.8 * v0, 1.2 * v0)), method="bounded", options={"xatol": 1e-16})
            if best is None or fit.fun < best[1]:
                best = (outer, fit.fun, fit.x)
        print(f"{('front', 'back')[j]}: tension {tension*1e3:5.1f} mN/m, best outer {best[0]} kg/m3 -> {best[1]*1e9:7.1f} nm (volume {best[2]*1e9:+.4f} uL)", flush=True)
scan(0, (0.015, 0.025, 0.035, 0.05), (900, 950, 997, 1030, 1060))
scan(1, (0.003, 0.004, 0.005, 0.006, 0.008), (760, 800, 850, 900, 950, 997))
