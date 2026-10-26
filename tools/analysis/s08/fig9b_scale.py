"""Scaled Fig. 9b: acoustics-off meniscus error versus lens scale (pin just outside the clear aperture)."""
# ruff: noqa: B023, SIM115  (exploratory scan scripts; closures are evaluated in-loop)
import json
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface, pinned_levels, pinned_singlet

mm = 1e-3
n = [1.0, 1.7, 1.0]
mat = json.load(open("configs/shared/noa61-water-pressure-reference.json"))
for aperture in (2.0, 3.0, 4.0, 6.0):  # clear radius, mm
    k = aperture / 28.0
    d0, d1, d2, t = -np.inf, 137.95 * mm * k, 90 * mm * k, 10 * mm * k
    R = aperture * 1.01 * mm
    pins = pinned_levels(R, n, [d0, d1, d2], 0.0, t)
    margin = 0.4 * aperture * mm
    for g in (9.81, 0.0):
        cfg = DualConfig(**{**mat["apparatus"], "gravity_m_s2": g, "radius_m": R, "clear_radius_m": aperture * mm,
                            "levels_m": (-margin, pins[0], pins[1], t + margin), "surface_elements": 36})
        faces = pinned_singlet(cfg, n, [d0, d1, d2], 0.0, t)
        r = np.linspace(0, aperture * mm, 201)
        target = np.array([f.evaluate(r) for f in faces])
        def errors(v):
            S = DualSurface(replace(cfg, volume_offsets_m3=tuple(v)), allow_volume_offsets=True)
            q = S.unforced_equilibrium()
            return np.array([np.max(abs(S.height(j, q[j], r) - target[j])) for j in range(2)])
        v0 = np.array([f.volume_offset() for f in faces])
        fit = minimize(lambda x: np.sum(errors(v0 * (1 + x)) ** 2) * 1e12, np.zeros(2), method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-9})
        e = errors(v0 * (1 + fit.x))
        sag = np.max(abs(target), axis=1)
        print(f"aperture {2*aperture:.0f} mm  g={g:4.2f}: f = {86.25*k:.2f} mm, t = {t/mm:.3f} mm, "
              f"meniscus error {e[0]*1e9:.0f}/{e[1]*1e9:.0f} nm = {e[0]/sag[0]:.2e}/{e[1]/sag[1]:.2e} of sag", flush=True)
