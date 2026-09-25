"""Carrier-aware semidefinite diagnostic; higher rank is not a coherent drive."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..provenance import capture_execution
from .precision import PrecisionObjective, WhitenedObjective


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=1e-12)
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    capture_execution(args.out)
    source = args.cache / "frozen-operator.npz"
    cache = dict(np.load(source))
    cfg = json.loads((args.cache / "resolved-config.json").read_text())
    limit = cfg["max_source_speed_m_s"]
    base = PrecisionObjective(cache, 1, limit / np.sqrt(2))
    coordinates = WhitenedObjective(base, args.cutoff)
    config = {
        "cache_directory": str(args.cache.resolve()),
        "cache_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_limit_m_s": limit,
        "cutoff": args.cutoff,
        "iterations": args.iterations,
        "reference": "docs/theory/sections/06_arrays.tex; docs/precision-program.md",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Relaxation in {coordinates.channels} complex source coordinates", flush=True)
    report, matrix = solve_relaxation(
        cache["force_kernels"],
        cache["required_force_n"],
        cache["carrier_height_response_s"],
        coordinates.transform,
        limit,
        args.iterations,
    )
    if matrix is not None:
        np.savez_compressed(
            args.out / "covariance.npz", covariance=matrix, transform=coordinates.transform
        )
        residual = cache["compliance_observation_m_n"] @ np.asarray(report["force_residual_n"])
        report["frozen_compliance_residual_max_m"] = float(np.max(abs(residual)))
    (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "report.md").write_text(
        "# Semidefinite source diagnostic\n\n"
        + json.dumps(report, indent=2)
        + "\n\nNo certified lower bound is claimed without a verified dual certificate. "
        "A higher-rank covariance is not a same-frequency coherent actuator command. "
        "Subspace truncation prevents extrapolating infeasibility to the full source space.\n"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "force_residual_n"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
