"""Independent checks of the load demand (mechanics/demand.py) on the common water cell.

1. Series Q2 r^2 + Q4 r^4 against the exact demanded traction near the axis.
2. The generalized force of the exact traction against the FE mechanical gradient
   at the projected target (two discretizations of the same functional).
3. The FE equilibrium under the exact traction reproduces the target faces.
"""

import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import lens_cell

from acoustic_freeform.mechanics.demand import (
    demanded_force,
    demanded_traction,
    sag_series,
    series_demand,
)
from acoustic_freeform.mechanics.surface import DualSurface

rows = []
for name in ("fig9b", "fig7a"):
    for elements in (36, 72):
        cfg, faces, _ = lens_cell(name, overrides={"surface_elements": elements})
        space = DualSurface(cfg, allow_volume_offsets=True)
        q, gradient, _ = demanded_force(space, faces)
        traction_force, series = [], []
        for j, face in enumerate(faces):
            sigma = cfg.surface_tension_n_m[j]
            drho_g = (cfg.density_kg_m3[j] - cfg.density_kg_m3[j + 1]) * cfg.gravity_m_s2
            t = demanded_traction(face, sigma, drho_g, space.r)
            traction_force.append(space.b.T @ (space.weights * t))
            s = sag_series(face, cfg.radius_m)
            coefficients = series_demand(s["c"], s["a4"], s["a6"], sigma, drho_g)
            near = np.linspace(1e-6, 0.15 * cfg.radius_m, 200)
            exact = demanded_traction(face, sigma, drho_g, near)
            exact -= demanded_traction(face, sigma, drho_g, np.array([1e-9]))[0]
            approx = coefficients["Q2_pa_m2"] * near**2 + coefficients["Q4_pa_m4"] * near**4
            full = demanded_traction(face, sigma, drho_g, np.linspace(1e-6, cfg.radius_m, 400))
            series.append({
                "face": j,
                **{k: float(v) for k, v in coefficients.items()},
                "Q2_R2_pa": float(coefficients["Q2_pa_m2"] * cfg.radius_m**2),
                "series_rel_error_r_le_0.15R": float(np.max(abs(exact - approx)) / np.max(abs(exact))),
                "demand_range_pa": float(np.ptp(full)),
            })
        traction_force = np.concatenate(traction_force)
        relative = float(np.linalg.norm(traction_force - gradient) / np.linalg.norm(gradient))
        q_eq = space.prescribed_equilibrium(traction_force, initial=q)
        pupil = np.linspace(0, cfg.clear_radius_m, 401)
        error = [float(np.max(abs(space.height(j, q_eq[j], pupil) - faces[j].evaluate(pupil)))) for j in range(2)]
        row = {"lens": name, "surface_elements": elements, "force_relative_difference": relative,
               "equilibrium_under_exact_traction_pupil_error_m": error, "faces": series}
        rows.append(row)
        print(json.dumps(row), flush=True)
