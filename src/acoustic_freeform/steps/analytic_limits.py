"""Independent analytic slab and Bessel compliance limits."""

import json
from pathlib import Path

from ..apparatus.config import DualConfig
from ..core.provenance import capture_execution
from ..verify.analytic import verify_slab


def add_arguments(parser):
    parser.add_argument("config", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--linear-solver", choices=["full", "static_condensed"], default="full")


def cli(args):
    """Independent analytic resolution audit, preserving earlier run evidence."""
    inputs = json.loads(args.config.read_text())
    cfg = DualConfig(**inputs["apparatus"])
    cfg.validate()
    args.out.mkdir(parents=True, exist_ok=False)
    capture_execution(args.out)
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "apparatus": cfg.as_dict(),
                "analytic_axial_cells": args.levels,
                "wave_linear_solver": args.linear_solver,
                "source_configuration": str(args.config),
                "scope": "Planar one-dimensional analytic limit; not curved-candidate convergence.",
            },
            indent=2,
        )
        + "\n"
    )
    records = verify_slab(cfg, levels=args.levels, linear_solver=args.linear_solver)
    (args.out / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = [
        "# Independent three-layer analytic resolution audit",
        "",
        "This validates a planar limit, not a shaped lens or 10 nm accuracy.",
        "",
        "| Axial cells/layer | Pressure relative error | Front/back velocity relative error |",
        "|---:|---:|---:|",
    ]
    for r in records:
        v = r["interface_velocity_relative_max_error"]
        lines.append(
            f"| {r['cells_per_layer']} | {r['pressure_relative_max_error']:.6g} "
            f"| {v[0]:.6g} / {v[1]:.6g} |"
        )
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
