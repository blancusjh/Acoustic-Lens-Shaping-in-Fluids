"""One entry point for simulation, independent verification and visualization."""

import argparse
import json
import shutil
import sys
import threading
import webbrowser
from dataclasses import replace
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "planar":
        from .verification.planar.cli import main as planar_main

        sys.argv.pop(1)
        return planar_main()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser(
        "simulate", help="Design array control and integrate the finite liquid lens"
    )
    simulate.add_argument("config", type=Path)
    simulate.add_argument("--out", type=Path, required=True)
    replay = commands.add_parser(
        "replay", help="Replay saved physical drives without shape feedback"
    )
    replay.add_argument("result", type=Path)
    replay.add_argument("--step", type=float, required=True, help="Fluid time step, seconds")
    replay.add_argument("--out", type=Path, required=True)
    for name, help_text in [
        ("render", "Build the offline time viewer, PyVista scene and optical figures"),
        ("view", "Serve the interactive viewer on the local machine"),
        ("verify", "Independent analytic acoustics and optional fixed-drive convergence"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("result", type=Path)
        if name == "view":
            command.add_argument("--port", type=int, default=8765)
            command.add_argument("--no-open", action="store_true")
        if name == "verify":
            command.add_argument("--spatial", action="store_true")
    args = parser.parse_args()
    from .lens.config import LensConfig, load_config

    if args.command == "simulate":
        from .lens.simulation import run_lens

        run_lens(load_config(args.config), args.out)
        shutil.copy2(args.config, args.out / "input.toml")
    elif args.command == "replay":
        from .lens.simulation import run_lens

        cfg = LensConfig(**json.loads((args.result / "configuration.json").read_text()))
        run_lens(
            replace(cfg, step_s=args.step), args.out, replay=np.load(args.result / "trajectory.npz")
        )
    elif args.command == "render":
        from .lens.visualization import export_viewer, render_figures

        print(export_viewer(args.result))
        print(render_figures(args.result))
    elif args.command == "view":
        from .lens.visualization import serve_viewer

        if not args.no_open:
            threading.Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}/")).start()
        serve_viewer(args.result, args.port)
    elif args.command == "verify":
        from .lens.validation import flat_cavity_check, spatial_convergence

        report = flat_cavity_check()
        directory = args.result / "verification"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "analytic-cavity.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if args.spatial:
            spatial_convergence(args.result)
