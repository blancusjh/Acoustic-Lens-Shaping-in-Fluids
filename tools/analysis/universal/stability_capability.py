"""Capability with a guaranteed static fold margin (LMI) for one lens.

1. The finite-difference acoustic stiffness of the kernel derivatives is checked
   against the analytic shape Jacobian of the solver at one command.
2. For each margin mu: convex best tolerance within the caps subject to
   sym(K - A(X)) >= mu K, single-tone refinement, and the refined tone's margin.
Usage: stability_capability.py LENS CAP_W OUT.jsonl MU [MU ...]   (MU = none or a number)
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import lens_cell

from acoustic_freeform.acoustics.helmholtz import DualAcoustics
from acoustic_freeform.apparatus.sources import modal_source_matrix
from acoustic_freeform.inverse.capability import (
    Tone,
    best_tolerance,
    single_tone_refinement,
)
from acoustic_freeform.mechanics.demand import demanded_force
from acoustic_freeform.mechanics.surface import DualSurface

lens, cap, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
margins = [None if m == "none" else float(m) for m in sys.argv[4:]]
cfg, faces, _ = lens_cell(lens)
space = DualSurface(cfg, allow_volume_offsets=True)
q, demand, stiffness = demanded_force(space, faces)
pupil = np.linspace(0, cfg.clear_radius_m, 41)
observe = np.kron(np.eye(2), space.basis(pupil)) @ np.linalg.inv(stiffness)
modes, _ = modal_source_matrix(cfg, {"bottom": 8, "top": 8}, 10)
tone = Tone(cfg, space, q, modes)
tone.shape_derivatives()


def acoustic_stiffness(c):
    return np.real(np.einsum("i,jnik,k->nj", c.conj(), tone.kernel_derivatives, c))


def margin(c):
    return float(np.min(np.linalg.eigvals(np.linalg.solve(stiffness, stiffness - acoustic_stiffness(c))).real))


reference = best_tolerance([tone], observe, demand, cap)
c0 = single_tone_refinement(tone, observe, demand, reference["tones"][0]["amplitudes"][:, 0], cap)["amplitudes"]
wave = DualAcoustics(cfg, space, linear_solver="static_condensed")
_, analytic = wave.shape_force_jacobian(q, modes @ c0)
check = float(np.linalg.norm(acoustic_stiffness(c0) - analytic) / np.linalg.norm(analytic))
print(json.dumps({"lens": lens, "fd_vs_analytic_acoustic_stiffness_relative": check}), flush=True)
for mu in margins:
    stability = None if mu is None else (stiffness, mu)
    convex = best_tolerance([tone], observe, demand, cap, stability=stability)
    row = {"lens": lens, "power_cap_w": cap, "required_margin": mu, "status": convex["status"]}
    if "tones" in convex:
        X = sum(np.outer(a, a.conj()) for a in convex["tones"][0]["amplitudes"].T)
        A = np.real(np.einsum("jnik,ki->nj", tone.kernel_derivatives, X))
        lead = convex["tones"][0]["amplitudes"][:, 0]
        single = single_tone_refinement(tone, observe, demand, lead, cap, pressure_cap_pa=5e6)
        row.update({
            "convex_pupil_error_m": convex["pupil_max_error_m"], "convex_power_w": convex["total_power_w"],
            "convex_rank": convex["tones"][0]["rank"],
            "convex_margin": float(np.min(np.linalg.eigvals(np.linalg.solve(stiffness, stiffness - A)).real)),
            "single_pupil_error_m": single["pupil_max_error_m"], "single_power_w": single["power_w"],
            "single_peak_pressure_pa": single["peak_pressure_pa"], "single_margin": margin(single["amplitudes"]),
        })
        np.save(Path(out).with_suffix(f".mu{mu}.npy"), single["amplitudes"])
    print(json.dumps(row), flush=True)
    with open(out, "a") as handle:
        handle.write(json.dumps(row) + "\n")
