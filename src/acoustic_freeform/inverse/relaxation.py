"""Carrier-aware semidefinite diagnostic; higher rank is not a coherent drive."""

import numpy as np


def solve_relaxation(kernels, required, carrier, transform, source_limit, iterations=20000):
    import cvxpy as cp

    reduced = np.einsum("ap,iab,bq->ipq", transform.conj(), kernels, transform, optimize=True)
    v = carrier @ transform / 1e-8
    size = transform.shape[1]
    y = cp.Variable((size, size), hermitian=True)
    peak_squared = cp.Variable(nonneg=True)
    force_scale = max(np.linalg.norm(required) / np.sqrt(len(required)), 1e-12)
    vector = cp.vec(y, order="C")
    predicted = cp.real(reduced.transpose(0, 2, 1).reshape(len(reduced), -1) @ vector)
    source_map = np.einsum("sa,sb->sab", transform, transform.conj()).reshape(len(transform), -1)
    carrier_map = np.einsum("sa,sb->sab", v, v.conj()).reshape(len(v), -1)
    constraints = [
        y >> 0,
        predicted / force_scale == required / force_scale,
        cp.real(source_map @ vector) <= source_limit**2,
        cp.real(carrier_map @ vector) <= peak_squared,
    ]
    problem = cp.Problem(cp.Minimize(peak_squared), constraints)
    value = problem.solve(solver="SCS", eps=1e-6, max_iters=iterations, verbose=False)
    report = {
        "solver": "SCS",
        "cvxpy_version": cp.__version__,
        "status": problem.status,
        "objective": None if not np.isfinite(value) else float(value),
        "iterations": problem.solver_stats.num_iters,
        "source_subspace_complex_dimension": size,
        "certified_lower_bound": False,
        "scope": "Numerical relaxation in a declared source subspace; no coherent, coupled, or physical pass.",
    }
    if y.value is None:
        return report, None
    matrix = (y.value + y.value.conj().T) / 2
    eigen = np.linalg.eigvalsh(matrix)
    actual = np.einsum("iab,ba->i", reduced, matrix).real
    carrier_squared = np.einsum("sa,ab,sb->s", v, matrix, v.conj()).real
    source_squared = np.einsum("sa,ab,sb->s", transform, matrix, transform.conj()).real
    report.update(
        {
            "force_residual_n": (actual - required).tolist(),
            "minimum_covariance_eigenvalue": float(eigen[0]),
            "dominant_eigenvalue_fraction": float(
                eigen[-1] / max(np.sum(np.maximum(eigen, 0)), 1e-30)
            ),
            "sampled_relaxed_carrier_peak_m": float(
                np.sqrt(max(np.max(carrier_squared), 0)) * 1e-8
            ),
            "source_second_moment_peak_m2_s2": float(np.max(source_squared)),
            "carrier_inequality_violation_scaled": float(
                np.max(carrier_squared) - peak_squared.value
            ),
        }
    )
    return report, matrix
