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
    stationary = commands.add_parser(
        "stationary", help="Joint stationary diopter/array design; no physical-time trajectory"
    )
    stationary.add_argument("config", type=Path)
    stationary.add_argument("--out", type=Path, required=True)
    stationary.add_argument("--seed", type=Path, help="Optional initial complex wall-drive JSON")
    stationary.add_argument("--starts", type=int, default=3)
    stationary.add_argument(
        "--stability", action="store_true", help="Check continuous fixed-drive linear stability"
    )
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
        ("excitation", "Export holding amplitudes/phases, drive schedule and surface perturbation"),
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
    elif args.command == "stationary":
        from .lens.stationary import run_stationary

        cfg = load_config(args.config)
        seed = None
        if args.seed:
            values = json.loads(args.seed.read_text())
            seed = np.asarray(values["velocity_real_m_s"]) + 1j * np.asarray(
                values["velocity_imag_m_s"]
            )
            if seed.shape != (cfg.array_rows,) or not np.all(np.isfinite(seed)):
                raise ValueError("Seed must contain one finite complex velocity per row.")
        if args.starts < 1:
            raise ValueError("At least one optimization start is required.")
        print(
            json.dumps(
                run_stationary(cfg, args.out, args.stability, seed_drive=seed, starts=args.starts),
                indent=2,
            )
        )
        shutil.copy2(args.config, args.out / "input.toml")
        if args.seed:
            shutil.copy2(args.seed, args.out / "initial-drive.json")
    elif args.command == "replay":
        from .lens.simulation import run_lens

        cfg = LensConfig(**json.loads((args.result / "configuration.json").read_text()))
        run_lens(
            replace(cfg, step_s=args.step), args.out, replay=np.load(args.result / "trajectory.npz")
        )
    elif args.command == "render":
        if (args.result / "stationary.npz").exists():
            from .lens.stationary_visualization import render_stationary

            print(render_stationary(args.result))
            return
        from .lens.visualization import export_viewer, render_figures

        print(export_viewer(args.result))
        print(render_figures(args.result))
    elif args.command == "view":
        if (args.result / "stationary.npz").exists():
            from .lens.stationary_visualization import render_stationary

            viewer = args.result.resolve() / "stationary-viewer.html"
            if not viewer.exists():
                render_stationary(args.result)
            if not args.no_open:
                webbrowser.open(viewer.as_uri())
            print(viewer)
            return
        from .lens.visualization import serve_viewer

        if not args.no_open:
            threading.Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}/")).start()
        serve_viewer(args.result, args.port)
    elif args.command == "verify":
        if (args.result / "stationary.npz").exists():
            from .lens.stationary import verify_stationary

            verify_stationary(args.result)
            return
        from .lens.validation import flat_cavity_check, spatial_convergence

        report = flat_cavity_check()
        directory = args.result / "verification"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "analytic-cavity.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if args.spatial:
            spatial_convergence(args.result)
    elif args.command == "excitation":
        if (args.result / "stationary.npz").exists():
            print((args.result / "hold-drive.csv").resolve())
            print((args.result / "excitation-definition.json").read_text())
            return
        from .lens.excitation import export_excitation

        print(json.dumps(export_excitation(args.result), indent=2))
