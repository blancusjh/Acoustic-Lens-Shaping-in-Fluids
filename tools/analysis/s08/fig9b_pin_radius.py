"""Apparatus for the scaled Silva-Lora & Torres Fig. 9b singlet (collimated input): pinning radius scan."""
# ruff: noqa: B023, SIM115, BLE001  (exploratory scan scripts; closures are evaluated in-loop)
import json
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface, pinned_levels, pinned_singlet

mm = 1e-3; k = 2.0 / 28.0
n = [1.0, 1.7, 1.0]
d0, d2, t = -np.inf, 90 * mm * k, 10 * mm * k
d1 = 137.95 * mm * k
mat = json.load(open("configs/shared/noa61-water-pressure-reference.json"))
def cell(R, gravity=9.81, margin=0.8 * mm):
    pins = pinned_levels(R, n, [d0, d1, d2], 0.0, t)
    levels = (-margin, pins[0], pins[1], t + margin)
    cfg = DualConfig(**{**mat["apparatus"], "gravity_m_s2": gravity, "radius_m": R, "clear_radius_m": 2 * mm,
                        "levels_m": levels, "surface_elements": 36})
    faces = pinned_singlet(cfg, n, [d0, d1, d2], 0.0, t)
    return cfg, faces
r = np.linspace(0, 2 * mm, 401)
def meniscus_error(cfg, faces, volumes):
    S = DualSurface(replace(cfg, volume_offsets_m3=tuple(volumes)), allow_volume_offsets=True)
    q = S.unforced_equilibrium()
    return [float(np.max(abs(S.height(j, q[j], r) - faces[j].evaluate(r)))) for j in range(2)]
print(f"sags at 2 mm: front {faces0[0].diopter.sag_superconic(np.array([2*mm]))[0]/mm:.4f}, back {faces0[1].diopter.sag_superconic(np.array([2*mm]))[0]/mm:.4f} mm") if (faces0 := cell(2.02*mm)[1]) else None
print(f"scaled Fig. 9b: d0 {d0/mm:.4f}, d1 {d1/mm:.4f}, d2 {d2/mm:.4f}, t {t/mm:.4f} mm")
for gravity in (9.81, 0.0):
    for R in (2.02, 2.05, 2.1, 2.2, 2.3):
        try:
            cfg, faces = cell(R * mm, gravity)
            target = [f.volume_offset() for f in faces]
            e_target = meniscus_error(cfg, faces, target)
            best = []
            for j in range(2):
                def err(v, j=j):
                    vols = list(target); vols[j] = v
                    return meniscus_error(cfg, faces, vols)[j]
                fit = minimize_scalar(err, bracket=(0.98 * target[j], target[j]), tol=1e-12)
                best.append(fit.fun)
            print(f"g={gravity:4.2f} R={R:.2f} mm: pins {cfg.levels_m[1]/mm:+.3f}/{cfg.levels_m[2]/mm:+.3f} mm, target volumes {target[0]*1e9:+.3f}/{target[1]*1e9:+.3f} uL -> meniscus error {e_target[0]*1e9:.0f}/{e_target[1]*1e9:.0f} nm; best-fit volumes {best[0]*1e9:.0f}/{best[1]*1e9:.0f} nm")
        except Exception as ex:
            print(f"g={gravity} R={R}: {type(ex).__name__}: {str(ex)[:100]}")
