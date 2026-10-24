"""Carrier-aware semidefinite diagnostic; higher rank is not a coherent drive."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ..core.provenance import capture_execution
from ..inverse.relaxation import solve_relaxation
from ..inverse.screening import PrecisionObjective, WhitenedObjective


def add_arguments(parser):
    parser.add_argument("cache", type=Path)
    parser.add_argument("--cutoff", type=float, default=1e-12)
    parser.add_argument("--iterations", type=int, default=20000)


def cli(args):
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
        "reference": "docs/theory/sections/06_arrays.tex; studies/S02-independent-two-face/precision-program.md",
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
