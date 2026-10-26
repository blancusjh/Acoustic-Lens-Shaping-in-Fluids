"""Silva-Lora & Torres Fig. 9 singlets (object at infinity): geometry, focus, aplanatism."""
# ruff: noqa: B023  (exploratory scan scripts; closures are evaluated in-loop)
import numpy as np
from scipy.optimize import brentq

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.optics.aplanatism import abbe_sine_condition, aplanatism_error
from acoustic_freeform.optics.cartesian import CartesianDiopter

n0, n1, n2 = 1.0, 1.7, 1.0
d0, d2, t = -np.inf, 90.0, 10.0  # mm, paper units
for name, d1, edge_pub in (("9a", 200.0, 3.4234459735169027), ("9b", 137.95, 3.3429500425595355)):
    s0, s1 = CartesianDiopter(n0, d0, n1, d1), CartesianDiopter(n1, d1 - t, n2, d2 - t)
    edge = lambda r: t + s1.sag_superconic(np.array([r]))[0] - s0.sag_superconic(np.array([r]))[0]
    R = brentq(lambda r: edge(r) - edge_pub, 1, 60)
    print(name, "radius giving published edge:", R, "T/D", t / (2 * R))
    a = R
    cfg = DualConfig(radius_m=a * 1.2, clear_radius_m=a, levels_m=(-5.0, 0.0, t, t + 40))
    surface = lambda face, r, d=0: (s0, s1)[face].sag_superconic(np.asarray(r, float)) if d == 0 else (s0, s1)[face].slope_superconic(np.asarray(r, float))
    r = np.r_[1e-9 * a, np.linspace(a / 200, a * 0.999, 200)]
    res = abbe_sine_condition(cfg, surface, [n0, n1, n2], [d0, d1, d2], [0.0, t], r, None)
    err = aplanatism_error(res)
    print("  spot max (mm):", res["max_spot_radius_m"], "transmitted", res["transmitted"].all())
    print("  ", {k: f"{v:.3e}" for k, v in err.items()})
    ok = res["transmitted"] & (r > 0)
    print("   abbe vs paper M max diff:", np.max(abs(res["abbe_ratio"][ok] / err["axial_abbe_ratio"] - res["paper_m"][ok])))
    print("   axial focal length h/(n sin u):", -err["axial_abbe_ratio"])
