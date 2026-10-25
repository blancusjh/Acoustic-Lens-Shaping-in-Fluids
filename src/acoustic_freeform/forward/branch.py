"""Start-up branch of held-command equilibria under a quasi-static ramp.

With the command s c and quadratic loads, the equilibria of the ramp solve
sigma F_ac(q; c) = grad E(q) with sigma = s^2 rising from 0 to 1, starting from
the unforced state. Each point is a fresh nonlinear force balance; the branch
ends early at a fold, where K - sigma A becomes singular and the physical
start-up would leave it. This is a property of the forward start-up, not
continuation in the target, and it is checked against time integration.
"""

import numpy as np


def branch_point_stability(space, stiffness, acoustic, sigma, mobility=None):
    """Static fold indicator and, with a mobility, the creeping-flow growth rate."""
    static = np.linalg.eigvals(np.linalg.solve(stiffness, stiffness - sigma * acoustic))
    result = {"static_min_real": float(np.min(static.real))}
    if mobility is not None:
        rates = np.linalg.eigvals(mobility @ (sigma * acoustic - stiffness))
        result["max_growth_rate_s"] = float(np.max(rates.real))
    return result


def startup_branch(
    wave,
    space,
    drive,
    sigmas,
    initial=None,
    tolerance_m=1e-13,
    max_corrections=12,
    min_increment=1e-3,
    mobility=None,
    progress=None,
):
    """Follow sigma F_ac(q; drive) = grad E(q) over increasing ``sigmas``.

    Returns the branch points, each with geometry, the pupil-independent
    compliance defect, the fold indicator and, if ``mobility(q)`` is supplied,
    the creeping-flow growth rate. ``reached_end`` is False at a fold or a
    failed correction; the last accepted point is then the end of the branch.
    """
    sigmas = np.asarray(sigmas, float)
    if sigmas[0] != 0 or np.any(np.diff(sigmas) <= 0):
        raise ValueError("Load parameters must start at zero and increase")
    q = space.unforced_equilibrium() if initial is None else np.array(initial, float)
    grad, stiffness, _ = space.mechanics(q)
    # The start is the acoustics-off capillary-gravity equilibrium.
    if np.max(abs(np.linalg.solve(stiffness, grad))) > 1e-12:
        raise ValueError("The start of the branch must be an acoustics-off equilibrium")
    response, acoustic = wave.shape_force_jacobian(q, drive)
    force = response.force(np.ones(1, complex))
    points = [
        {
            "sigma": 0.0,
            "coefficients_m": q.copy(),
            "defect_m": 0.0,
            **branch_point_stability(space, stiffness, acoustic, 0.0),
        }
    ]
    reached_end, reason = True, None
    # Requested load levels are visited in order; a failed or geometrically
    # invalid correction halves the increment, down to ``min_increment``.
    pending = list(sigmas[1:])
    previous = 0.0
    while pending:
        sigma = pending[0]
        # Predictor along the branch tangent (K - sigma A) dq/dsigma = F_ac.
        tangent = np.linalg.solve(stiffness - previous * acoustic, force).reshape(q.shape)
        trial = q + (sigma - previous) * tangent
        defect = np.inf
        try:
            response = wave.solve(trial, drive)
            for _ in range(max_corrections):
                load = response.force(np.ones(1, complex))
                grad, trial_stiffness, _ = space.mechanics(trial)
                residual = sigma * load - grad
                # Chord corrector: A is held from the last accepted point; the
                # load and the mechanical stiffness are fresh at every correction.
                step = np.linalg.solve(trial_stiffness - sigma * acoustic, residual)
                step = step.reshape(q.shape)
                defect = max(space.polynomial_maximum(v)["max_abs_m"] for v in step)
                if defect < tolerance_m:
                    break
                trial = trial + step
                response = wave.solve(trial, drive)
        except ValueError:
            defect = np.inf
        if defect >= tolerance_m:
            if sigma - previous <= min_increment:
                reached_end = False
                reason = f"Corrector failed at sigma={sigma:.4g} with the minimum increment"
                break
            pending.insert(0, previous + (sigma - previous) / 2)
            continue
        pending.pop(0)
        q, previous = trial, sigma
        grad, stiffness, _ = space.mechanics(q)
        response, acoustic = wave.shape_force_jacobian(q, drive)
        force = response.force(np.ones(1, complex))
        point = {
            "sigma": float(sigma),
            "coefficients_m": q.copy(),
            "defect_m": float(defect),
            **branch_point_stability(
                space, stiffness, acoustic, sigma, None if mobility is None else mobility(q)
            ),
        }
        points.append(point)
        if progress:
            progress(point)
        if point["static_min_real"] <= 0:
            reached_end, reason = False, f"Fold: K - sigma A not positive at sigma={sigma:.4g}"
            break
    return {
        "points": points,
        "reached_end": bool(reached_end and points[-1]["sigma"] == sigmas[-1]),
        "stop_reason": reason,
        "final_response": response,
        "final_acoustic_jacobian": acoustic,
        "final_stiffness": stiffness,
    }
