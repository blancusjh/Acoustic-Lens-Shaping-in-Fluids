"""Optics of the tuned-liquid Fig. 9b states with the published indices (cured lens in air)."""
# ruff: noqa: SIM115  (exploratory scan scripts; closures are evaluated in-loop)
import json
import sys

import numpy as np

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface, targets_from_record
from acoustic_freeform.optics.aplanatism import abbe_sine_condition, aplanatism_error
from acoustic_freeform.optics.raytrace import trace_pair

run = sys.argv[1]
record = json.load(open(f"{run}/config.json"))
cfg = DualConfig(**json.load(open(f"{run}/design/config.json")))
space = DualSurface(cfg, allow_volume_offsets=True)
targets = targets_from_record(cfg, record)
n = record["optics"]["indices"]; z = [float(x) for x in record["optics"]["stigmatic_z_m"]]
r = np.r_[1e-9, np.linspace(cfg.clear_radius_m / 400, cfg.clear_radius_m, 400)]
states = {"acoustics-off meniscus": space.unforced_equilibrium(),
          "design": np.load(f"{run}/design/state.npz")["coefficients_m"]}
if len(sys.argv) > 2:
    states["formed (end of start-up)"] = np.load(sys.argv[2])["coefficients_m"][-1]
wavelength = 587.6e-9
f_number = (z[2] - cfg.levels_m[2]) / (2 * cfg.clear_radius_m)
airy = 1.22 * wavelength * f_number
print(f"image at {z[2]*1e3:.3f} mm, working f-number {f_number:.2f}, Airy radius {airy*1e6:.2f} um at 587.6 nm")
def target_surface(face, rr, d=0):
    return targets[face].evaluate(np.asarray(rr, float), d)
res = abbe_sine_condition(cfg, target_surface, n, z, [0.0, record["optics"]["thickness_m"]], r, None)
print("target: spot max", res["max_spot_radius_m"], "aplanatism", aplanatism_error(res)["paper_rms"])
for name, q in states.items():
    tr = trace_pair(space, q, n, z[0], z[2], launch_radius_m=r)
    err = [np.max(abs(space.height(j, q[j], r) - targets[j].evaluate(r))) for j in range(2)]
    # Wavefront from the height error: (n1 - n_outer) dh per face, signs per face
    opd = (n[1] - n[0]) * (space.height(0, q[0], r) - targets[0].evaluate(r)) - (n[1] - n[2]) * (space.height(1, q[1], r) - targets[1].evaluate(r))
    w = r**2 / cfg.clear_radius_m**2
    opd_pv = np.ptp(opd - np.polyval(np.polyfit(w, opd, 1), w))  # remove piston and focus
    print(f"{name}: height error {err[0]*1e9:.1f}/{err[1]*1e9:.1f} nm; spot max {tr['max_radius_m']*1e6:.3f} um, rms {tr['rms_radius_m']*1e6:.3f} um, "
          f"{int(tr['transmitted'].sum())}/{len(r)} rays; surface-error OPD (focus removed) PV {opd_pv*1e9:.1f} nm = lambda/{wavelength/max(opd_pv,1e-30):.0f}")
