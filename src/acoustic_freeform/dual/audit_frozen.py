"""Independent polynomial-extremum audit of a saved frozen compliance fit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..provenance import capture_execution
from .config import DualConfig
from .surface import DualSurface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    capture_execution(args.out)
    operator_path = args.cache / "frozen-operator.npz"
    with operator_path.open("rb") as stream:
        operator_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    cfg = DualConfig(**json.loads((args.cache / "resolved-config.json").read_text()))
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "cache_directory": str(args.cache),
                "source_file": str(args.source),
                "operator_sha256": operator_hash,
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "apparatus": cfg.as_dict(),
                "scope": "Frozen compliance perturbation, not achieved geometry.",
            },
            indent=2,
        )
        + "\n"
    )
    cache = np.load(operator_path)
    source = np.load(args.source)["source_velocity_m_s"]
    space = DualSurface(cfg)
    required, stiffness, _ = space.mechanics(cache["target_coefficients_m"])
    np.testing.assert_allclose(required, cache["required_force_n"], rtol=1e-10, atol=1e-20)
    force_error = (
        np.einsum("a,iab,b->i", source.conj(), cache["force_kernels"], source).real - required
    )
    perturbation = np.linalg.solve(stiffness, force_error).reshape(2, -1)
    sampled = cache["compliance_observation_m_n"] @ force_error
    report = []
    for j, q in enumerate(perturbation):
        face = cache["observation_faces"] == j
        report.append(
            {
                "face": j,
                "legacy_sampled_full_max_m": float(np.max(abs(sampled[face]))),
                "pupil": space.polynomial_maximum(q, upper_m=cfg.clear_radius_m),
                "annulus": space.polynomial_maximum(q, lower_m=cfg.clear_radius_m),
                "coupled_accuracy_established": False,
            }
        )
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "validation.json").write_text(
        json.dumps(
            {
                "required_force_recomputed_and_matched": True,
                "method": "Floating-point derivative roots plus spline knots and interval ends",
                "independent_analytic_test": "tests/dual/test_polynomial_audit.py (not rerun by this audit)",
                "new_acoustic_field_solved": False,
                "rigorous_interval_bound": False,
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Frozen compliance extremum audit",
        "",
        "Not a coupled surface or physical accuracy certificate.",
        "",
        "| Face | Legacy full sampled max (nm) | Polynomial pupil max (nm) | Polynomial annular max (nm) |",
        "|---:|---:|---:|---:|",
    ]
    for r in report:
        lines.append(
            f"| {r['face']} | {r['legacy_sampled_full_max_m'] * 1e9:.6f} "
            f"| {r['pupil']['max_abs_m'] * 1e9:.6f} | {r['annulus']['max_abs_m'] * 1e9:.6f} |"
        )
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
