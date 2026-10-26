"""Aplanatic stigmatic NOA 61 / water pair versus object distance (numerical aperture)."""
# ruff: noqa: SIM115  (exploratory scan scripts; closures are evaluated in-loop)
import json
import sys
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.optics.aplanatism import abbe_sine_condition, aplanatism_error
from acoustic_freeform.optics.cartesian import CartesianDiopter
from acoustic_freeform.optics.stigmatic import pair_prescription

mat = json.load(open("configs/shared/noa61-water-pressure-reference.json"))
# Optical search frame only: a tall cell so the lens is not limited by the S07 geometry.
cfg = replace(DualConfig(**mat["apparatus"]), levels_m=(-0.006, -0.001, 0.001, 0.006))
n = mat["indices"]; L = cfg.levels_m; a = cfg.clear_radius_m
z0 = -float(sys.argv[1]) * 1e-3
m_target = float(sys.argv[2])
def build(p, rays):
    z2 = abs(z0) * (0.3 + 4.7 / (1 + np.exp(-p[0])))          # image 0.3 to 5 times the object distance
    z1 = 1 / p[1] if abs(p[1]) > 1e-6 else 1e9
    v0, v1 = p[2] * 1e-3, p[3] * 1e-3
    lab = [L[1] + v0, L[2] + v1]
    if not (L[0] + 0.3e-3 < lab[0] < lab[1] - 0.2e-3 and lab[1] < L[3] - 0.3e-3): return None
    try:
        faces = pair_prescription(cfg, n, [z0, z1, z2], [v0, v1])
        D = [CartesianDiopter(**f["optical"]) for f in faces]
        disp = [v0, v1]
        if rays < 100:  # search: cubic splines of the exact surfaces (final candidates are traced exactly)
            from scipy.interpolate import CubicSpline
            nodes = np.linspace(0, cfg.radius_m, 801)
            splines = [CubicSpline(nodes, D[j].sag_superconic(nodes) + disp[j]) for j in range(2)]
            def surface(face, rr, d=0):
                return splines[face](np.asarray(rr, float), d)
        else:
            def surface(face, rr, d=0):
                rr = np.asarray(rr, float)
                return D[face].sag_superconic(rr) + disp[face] if d == 0 else D[face].slope_superconic(rr)
        r = np.r_[1e-7, np.linspace(a / rays, a, rays)]
        h = [L[j + 1] + surface(j, r) for j in range(2)]
        if np.any(~np.isfinite(h[0])) or np.any(~np.isfinite(h[1])) or np.min(h[1] - h[0]) < 0.2e-3: return None
        if np.min(h[0]) < L[0] + 0.2e-3 or np.max(h[1]) > L[3] - 0.2e-3: return None
        res = abbe_sine_condition(cfg, surface, n, [z0, z1, z2], lab, r, h[0])
    except (ValueError, RuntimeError, FloatingPointError):
        return None
    if not res["transmitted"].all(): return None
    return res, aplanatism_error(res), (z0, z1, z2, v0, v1), h
def f(p):
    out = build(p, 40)
    # Fixed unequal magnification: penalize the axial Abbe ratio away from m_target.
    return 1.0 if out is None else out[1]["paper_rms"] + 0.1 * (out[1]["axial_abbe_ratio"] - m_target) ** 2
rng = np.random.default_rng(7)
best, best_x = None, None
for trial in range(10):
    p0 = [rng.uniform(-3, 3), rng.uniform(-100, 100), rng.uniform(-3.0, 0.5), rng.uniform(-0.5, 3.0)]
    if f(p0) >= 1: continue
    fit = minimize(f, p0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-16, "maxiter": 800})
    out = build(fit.x, 200)
    if out is None: continue
    if best is None or f(fit.x) < f(best_x):
        best, best_x = out, fit.x
if best is None:
    print(json.dumps({"object_mm": float(sys.argv[1]), "found": False})); sys.exit()
res, e, z, h = best
first = res["launch_radius_m"][-1] / np.hypot(res["launch_radius_m"][-1], h[0][-1] - z[0])
print(json.dumps({"m_target": m_target, "object_mm": -z[0] * 1e3, "image_mm": z[2] * 1e3, "shared_mm": z[1] * 1e3,
                  "front_vertex_mm": (L[1] + z[3]) * 1e3, "back_vertex_mm": (L[2] + z[4]) * 1e3,
                  "center_thickness_mm": (z[4] - z[3] + L[2] - L[1]) * 1e3,
                  "object_NA": float(n[0] * first), "magnification": e["axial_abbe_ratio"],
                  "M_minus_1_rms": e["paper_rms"], "M_minus_1_max": e["paper_max"],
                  "front_sag_um": float((h[0][-1] - h[0][0]) * 1e6), "back_sag_um": float((h[1][-1] - h[1][0]) * 1e6)}))
