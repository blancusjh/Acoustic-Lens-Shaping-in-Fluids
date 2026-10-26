"""Capability of a common source: can its time-averaged force supply a lens's demand?

About a target state q* the generalized acoustic force of a drive with complex
modal amplitudes c at one frequency is F_i = c^H G_i c. Tones at distinct
frequencies (spaced far beyond the creeping-flow bandwidth) add their
time-averaged forces, so a multi-tone drive gives F_i = sum_t tr(G_ti X_t)
with X_t positive semidefinite: a single tone is rank one, r tones near one
frequency are rank r. The set of reachable forces is then a convex cone, and
the following are semidefinite programs:

- ``minimum_power``: least total source power whose force brings the frozen
  pupil heights within a tolerance of the target,
- ``best_tolerance``: smallest pupil tolerance reachable within a power cap.

Heights are first-order, frozen-compliance predictions: delta h = O (F - d),
O = H K^-1, with d the load demand (``mechanics.demand``). They bound what the
source can do; acceptance still needs the nonlinear chain (start-up branch,
coupled stability, formation).
"""

import cvxpy as cp
import numpy as np

from ..acoustics.helmholtz import DualAcoustics
from .faithful import ModalOperators


class Tone:
    """Force kernels, power form and pressure rows of one frequency band at q*."""

    def __init__(self, cfg, space, q, modes, pressure_samples=400):
        self.frequency_hz = cfg.frequency_hz
        wave = DualAcoustics(cfg, space, linear_solver="static_condensed")
        ops = ModalOperators(wave, q, modes, pressure_samples)
        kernels = ops.kernels
        self.kernels = (kernels + np.conj(np.swapaxes(kernels, 1, 2))) / 2
        self.power = ops.power
        self.pressure = ops.all_pressure
        self.operators = ops
        self.wave, self.q, self.modes = wave, np.array(q), modes
        self.kernel_derivatives = None

    def shape_derivatives(self, step_m=1e-9):
        """d kernels / d q_j by forward differences, shape (n_q, n_force, m, m).

        The acoustic shape stiffness of a drive matrix X is then
        A_ij = Re tr(dG_ji X), linear in X like the force itself.
        """
        flat = self.q.ravel()
        derivatives = []
        for j in range(len(flat)):
            moved = flat.copy()
            moved[j] += step_m
            ops = ModalOperators(self.wave, moved.reshape(self.q.shape), self.modes, 1)
            hermitian = (ops.kernels + np.conj(np.swapaxes(ops.kernels, 1, 2))) / 2
            derivatives.append((hermitian - self.kernels) / step_m)
        self.kernel_derivatives = np.array(derivatives)
        return self.kernel_derivatives


def _problem(
    tones, observe, demand, scale_m, volume_columns=None, stability=None, pressure_cap_pa=None
):
    """Scaled SDP pieces. Each band's matrix is X_t = s_t Y_t with s_t chosen so the
    pupil-height rows of Y_t are O(1) in units of scale_m; only the pupil heights of
    the force enter. Pump trims are in nanolitres."""
    variables, heights, power = [], -observe @ demand / scale_m, 0
    for t in tones:
        m = t.kernels.shape[1]
        raw = observe @ np.conj(t.kernels).reshape(len(t.kernels), m * m)
        s = scale_m / np.max(abs(raw))
        Y = cp.Variable((m, m), hermitian=True)
        rows = raw * (s / scale_m)
        heights = heights + cp.real(rows @ cp.vec(Y, order="C"))
        power = power + s * cp.real(cp.trace(t.power @ Y))
        variables.append((Y, s))
    pumps = None
    if volume_columns is not None:
        # Fill-volume trims (m^3) are free machine inputs: linear height columns.
        pumps = cp.Variable(volume_columns.shape[1])
        heights = heights + volume_columns @ pumps * (1e-12 / scale_m)
    constraints = [Y >> 0 for Y, _ in variables]
    if pressure_cap_pa is not None:
        # Per-band peak amplitude at the sampled pressure rows: p_s^H X p_s <= cap^2,
        # linear in X (exact for one tone per band, an RMS bound for several).
        for t, (Y, s) in zip(tones, variables):
            rows = t.operators.pressure_rows
            m = rows.shape[1]
            outer = np.einsum("si,sj->sij", rows.conj(), rows).reshape(len(rows), m * m)
            square = cp.real(outer @ cp.vec(Y, order="C")) * (s / pressure_cap_pa**2)
            constraints.append(square <= 1)
    if stability is not None:
        # Static fold margin as a linear matrix inequality:
        # sym(K - A(X)) >= mu K  implies  Re eig(K^-1 (K - A)) >= mu.
        stiffness, mu = stability
        norm = np.linalg.norm(stiffness, 2)
        n = stiffness.shape[0]
        coupled = (1 - mu) * stiffness / norm
        for t, (Y, s) in zip(tones, variables):
            m = t.kernels.shape[1]
            # rows (i, j) -> A_ij = Re tr(dG_ji Y): transpose the (j, i) layout.
            derivative = np.conj(np.swapaxes(t.kernel_derivatives, 0, 1)).reshape(n * n, m * m)
            acoustic = cp.reshape(
                cp.real(derivative @ cp.vec(Y, order="C")) * (s / norm), (n, n), order="C"
            )
            coupled = coupled - (acoustic + acoustic.T) / 2
        constraints.append(coupled >> 0)
    return variables, heights, power, constraints, pumps


def _report(tones, variables, observe, demand, status, volume_columns=None, pumps=None):
    X = [np.asarray(Y.value) * s for Y, s in variables]
    force = sum(
        np.real(np.einsum("nij,ij->n", np.conj(t.kernels), x)) for t, x in zip(tones, X)
    )
    tones_out = []
    for t, x in zip(tones, X):
        values, vectors = np.linalg.eigh((x + x.conj().T) / 2)
        values = np.clip(values, 0, None)
        keep = values > 1e-6 * max(values.max(), 1e-300)
        amplitudes = vectors[:, keep][:, ::-1] * np.sqrt(values[keep][::-1])
        tones_out.append(
            {
                "frequency_hz": t.frequency_hz,
                "power_w": float(np.real(np.trace(t.power @ x))),
                "rank": int(keep.sum()),
                "eigen_power_fraction": (values[::-1][:6] / max(values.sum(), 1e-300)).tolist(),
                "rms_pressure_peak_pa": float(
                    np.sqrt(np.max(np.real(np.einsum("si,ij,sj->s", t.pressure.conj(), x, t.pressure))))
                ),
                "worst_case_peak_pressure_pa": float(
                    np.max(np.sum(abs(t.pressure @ amplitudes), axis=1)) if keep.any() else 0.0
                ),
                "amplitudes": amplitudes,
            }
        )
    heights = observe @ (force - demand)
    trims = None
    if pumps is not None:
        trims = np.asarray(pumps.value) * 1e-12
        heights = heights + volume_columns @ trims
    return {
        "status": status,
        "pupil_max_error_m": float(np.max(abs(heights))),
        "total_power_w": float(sum(t["power_w"] for t in tones_out)),
        "volume_trims_m3": None if trims is None else trims.tolist(),
        "tones": tones_out,
        "force": force,
    }


def minimum_power(
    tones,
    observe,
    demand,
    tolerance_m,
    volume_columns=None,
    pressure_cap_pa=None,
    solver="CLARABEL",
):
    variables, heights, power, constraints, pumps = _problem(
        tones, observe, demand, tolerance_m, volume_columns, pressure_cap_pa=pressure_cap_pa
    )
    problem = cp.Problem(cp.Minimize(power), [*constraints, cp.abs(heights) <= 1])
    problem.solve(solver=solver)
    if variables[0][0].value is None:
        return {"status": problem.status}
    return _report(tones, variables, observe, demand, problem.status, volume_columns, pumps)


def best_tolerance(
    tones,
    observe,
    demand,
    power_cap_w,
    scale_m=1e-8,
    volume_columns=None,
    stability=None,
    pressure_cap_pa=None,
    solver="CLARABEL",
):
    """``stability=(K, mu)`` adds the fold-margin inequality (needs shape_derivatives);
    ``pressure_cap_pa`` bounds each band's sampled peak amplitude."""
    variables, heights, power, constraints, pumps = _problem(
        tones, observe, demand, scale_m, volume_columns, stability, pressure_cap_pa
    )
    bound = cp.Variable()
    problem = cp.Problem(
        cp.Minimize(bound), [*constraints, cp.abs(heights) <= bound, power / power_cap_w <= 1]
    )
    problem.solve(solver=solver)
    if variables[0][0].value is None:
        return {"status": problem.status}
    return _report(tones, variables, observe, demand, problem.status, volume_columns, pumps)


def single_tone_refinement(
    tone, observe, demand, start, power_cap_w, pressure_cap_pa=None, evaluations=400, stability=None
):
    """Local rank-one fit from a starting amplitude (e.g. the leading eigenvector)."""
    result = tone_per_band_refinement(
        [tone], observe, demand, [start], power_cap_w, pressure_cap_pa, evaluations, stability
    )
    return {**result, "amplitudes": result["amplitudes"][0]}


def fold_margin(tones, amplitudes, stiffness):
    """min Re eig K^-1 (K - A) for one tone per band (needs shape_derivatives)."""
    acoustic = sum(
        np.real(np.einsum("i,jnik,k->nj", c.conj(), t.kernel_derivatives, c))
        for t, c in zip(tones, amplitudes)
    )
    return float(np.min(np.linalg.eigvals(np.linalg.solve(stiffness, stiffness - acoustic)).real))


def tone_per_band_refinement(
    tones,
    observe,
    demand,
    starts,
    power_cap_w,
    pressure_cap_pa=None,
    evaluations=400,
    stability=None,
):
    """Local fit with exactly one tone in each band, from given starting amplitudes.

    Minimizes the pupil height misfit with soft penalties above the total power
    cap, optionally above the per-band sampled peak pressure cap, and, with
    ``stability=(K, mu)``, below the static fold margin mu.
    """
    from scipy.optimize import least_squares

    sizes = [t.kernels.shape[1] for t in tones]
    scale = np.linalg.norm(observe @ demand) + 1e-300

    def unpack(x):
        out, offset = [], 0
        for m in sizes:
            out.append(x[offset : offset + m] + 1j * x[offset + m : offset + 2 * m])
            offset += 2 * m
        return out

    def evaluate(x):
        cs = unpack(x)
        force = sum(
            np.real(np.einsum("i,nij,j->n", c.conj(), t.kernels, c)) for t, c in zip(tones, cs)
        )
        power = sum(float(np.real(c.conj() @ t.power @ c)) for t, c in zip(tones, cs))
        return cs, force, power

    def fun(x):
        cs, force, power = evaluate(x)
        penalties = [10 * max(0.0, power / power_cap_w - 1.0)]
        if pressure_cap_pa is not None:
            for t, c in zip(tones, cs):
                peak = abs(t.operators.pressure_rows @ c) / pressure_cap_pa
                penalties.append(10 * np.sqrt(np.sum(np.maximum(0.0, peak - 1.0) ** 2)))
        if stability is not None:
            stiffness, mu = stability
            penalties.append(10 * max(0.0, mu - fold_margin(tones, cs, stiffness)))
        return np.r_[observe @ (force - demand) / scale, penalties]

    x0 = np.concatenate([np.r_[c.real, c.imag] for c in starts])
    fit = least_squares(fun, x0, max_nfev=evaluations)
    cs, force, power = evaluate(fit.x)
    margin = None
    if stability is not None:
        margin = fold_margin(tones, cs, stability[0])
    return {
        "fold_margin": margin,
        "pupil_max_error_m": float(np.max(abs(observe @ (force - demand)))),
        "power_w": power,
        "peak_pressure_pa": float(max(np.max(abs(t.pressure @ c)) for t, c in zip(tones, cs))),
        "band_power_w": [float(np.real(c.conj() @ t.power @ c)) for t, c in zip(tones, cs)],
        "amplitudes": cs,
    }
