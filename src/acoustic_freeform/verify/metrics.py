"""Acceptance metrics: sampled maximum height error and geometric gates."""

import numpy as np


def maximum_error(space, q, target):
    """Independent dense sampling and batched derivative-root refinement.

    This is a numerical extremum search, not an interval-arithmetic certificate.
    The mesh is independent of acoustic quadrature and surface collocation.
    """
    r = np.linspace(0, space.config.clear_radius_m, 2001)
    target_h = target.evaluate(r)
    error = space.evaluate(q, r) - target_h
    values = abs(error)
    best = float(np.max(values))
    derivative = space.evaluate(q, r, 1) - target.evaluate(r, 1)
    crossings = np.flatnonzero(derivative[:-1] * derivative[1:] < 0)
    low, high = r[crossings].copy(), r[crossings + 1].copy()
    left = derivative[crossings].copy()
    if len(crossings):
        # Batch the independent derivative-root searches. Scalar minimizers
        # spend minutes resolving roundoff peaks of near-exact projections.
        for _ in range(32):
            middle = (low + high) / 2
            value = space.evaluate(q, middle, 1) - target.evaluate(middle, 1)
            same_sign = np.signbit(value) == np.signbit(left)
            low = np.where(same_sign, middle, low)
            high = np.where(same_sign, high, middle)
            left = np.where(same_sign, value, left)
        roots = (low + high) / 2
        best = max(best, float(np.max(abs(space.evaluate(q, roots) - target.evaluate(roots)))))
    rms = np.sqrt(np.trapezoid(r * error**2, r) * 2 / space.config.clear_radius_m**2)
    return {
        "max_error_m": best,
        "rms_error_m": float(rms),
        "metric": "Absolute lab-coordinate height; no piston or tilt removal.",
        "search": "2001 independent radii and batched bisection of derivative sign changes.",
        "rigorous_interval_bound": False,
    }


def geometric_acceptance(mean_errors, phase_errors, target):
    """Numerical geometry gates only; never a physical or convergence certificate."""
    if target not in ("cycle_mean", "instantaneous"):
        raise ValueError("Accuracy target must be cycle_mean or instantaneous.")
    mean = np.asarray(mean_errors)
    phase = np.asarray(phase_errors)
    if mean.shape != (2,) or phase.shape != (2,) or not np.all(np.isfinite([mean, phase])):
        raise ValueError("Finite errors for both faces are required.")
    return {
        "declared_accuracy_target": target,
        "mean_10nm_sampled_test": bool(np.all(mean <= 1e-8)),
        "instantaneous_10nm_rejected_by_samples": bool(np.any(phase > 1e-8)),
        "declared_10nm_sampled_test": bool(
            np.all((mean if target == "cycle_mean" else phase) <= 1e-8)
        ),
        "physical_accuracy_certified": False,
    }
