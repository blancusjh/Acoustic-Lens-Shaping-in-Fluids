"""Local command line entry points; no remote services or expensive agent runs."""

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_experiment
from .simulation import load_result, run, save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser("run", help="Run a forward experiment")
    simulate.add_argument("experiment", type=Path)
    simulate.add_argument("--out", type=Path, required=True)
    simulate.add_argument(
        "--weights", type=Path, help="Optional complex .npy (beam,row,col) weights"
    )
    for name in ("render", "view"):
        command = commands.add_parser(
            name, help="Render output" if name == "render" else "Open time slider"
        )
        command.add_argument("result", type=Path)
        if name == "render":
            command.add_argument("--no-movie", action="store_true")
    validate = commands.add_parser(
        "validate", help="Run convergence studies and independent acoustics check"
    )
    validate.add_argument("--out", type=Path, default=Path("runs/validation"))
    args = parser.parse_args()
    if args.command == "run":
        if (args.out / "result.npz").exists():
            parser.error("Output exists; choose a new --out directory.")
        weights = None if args.weights is None else np.load(args.weights, allow_pickle=False)
        result = run(load_experiment(args.experiment), element_weights=weights)
        save_result(result, args.out, args.experiment)
        print(json.dumps(result.report["diagnostics"], indent=2))
        print(f"Saved: {args.out.resolve()}")
    elif args.command == "validate":
        from .validation import validate_forward_model

        print(json.dumps(validate_forward_model(args.out), indent=2))
    else:
        from .visualization import Dashboard, render

        result = load_result(args.result)
        if args.command == "render":
            render(result, args.result / "visuals", movie=not args.no_movie)
            print(f"Rendered: {(args.result / 'visuals').resolve()}")
        else:
            Dashboard(result, off_screen=False).show()
