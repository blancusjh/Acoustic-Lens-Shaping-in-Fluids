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
    regrid = commands.add_parser(
        "regrid-stationary", help="Re-solve the same drive and liquid volume on another mesh"
    )
    regrid.add_argument("result", type=Path)
    regrid.add_argument("--out", type=Path, required=True)
    regrid.add_argument(
        "--resolution", type=int, nargs=3, required=True, metavar=("NR", "NZ", "MODES")
    )
    replay = commands.add_parser(
        "replay", help="Replay saved physical drives without shape feedback"
    )
    replay.add_argument("result", type=Path)
    replay.add_argument("--step", type=float, required=True, help="Fluid time step, seconds")
    replay.add_argument("--out", type=Path, required=True)
    diagnosis = commands.add_parser(
        "diagnose-cavity", help="Separate geometry and wave errors at fixed surface and drive"
    )
    diagnosis.add_argument("result", type=Path)
    diagnosis.add_argument("--out", type=Path, required=True)
    coupled = commands.add_parser(
        "verify-coupled", help="Re-solve a holding drive on consistent, refined curved meshes"
    )
    coupled.add_argument("result", type=Path)
    coupled.add_argument("--out", type=Path, required=True)
    coupled.add_argument(
        "--level", type=int, nargs=3, action="append", metavar=("NR", "NZ", "MODES")
    )
    coupled.add_argument("--method", choices=["hybr", "broyden1"], default="hybr")
    control_model = commands.add_parser(
        "control-model", help="Radiation/fluid control surrogate; omits streaming derivatives"
    )
    control_model.add_argument("result", type=Path)
    control_model.add_argument("--out", type=Path, required=True)
    control_model.add_argument("--difference-step", type=float, default=1e-6)
    frequency = commands.add_parser(
        "screen-frequency", help="Frozen-target frequency study; requires coupled verification"
    )
    frequency.add_argument("result", type=Path)
    frequency.add_argument("--out", type=Path, required=True)
    frequency.add_argument("--frequencies", type=float, nargs="+")
    stability = commands.add_parser(
        "stability", help="Fixed-drive acoustic sensitivity and inertial fluid stability"
    )
    stability.add_argument("result", type=Path)
    stability.add_argument("--out", type=Path, required=True)
    stability.add_argument("--difference-step", type=float, default=1e-6)
    stabilize = commands.add_parser(
        "stabilize", help="Optimize a holding drive with a local inertial stability penalty"
    )
    stabilize.add_argument("result", type=Path)
    stabilize.add_argument("--stability", type=Path, required=True)
    stabilize.add_argument("--out", type=Path, required=True)
    stabilize.add_argument(
        "--margin",
        type=float,
        default=5.0,
        help="Requested continuous decay margin, inverse seconds",
    )
    budget = commands.add_parser(
        "acoustic-budget", help="Energy, bulk absorption flow and fast-interface diagnostics"
    )
    budget.add_argument("result", type=Path)
    budget.add_argument("--out", type=Path, required=True)
    feedback = commands.add_parser(
        "feedback", help="Synthesize surface feedback and check inertia plus sample delay"
    )
    feedback.add_argument("result", type=Path)
    feedback.add_argument("--stability", type=Path, required=True)
    feedback.add_argument("--out", type=Path, required=True)
    feedback.add_argument("--decay", type=float, default=40.0)
    feedback.add_argument("--period", type=float, default=0.005)
    evolve = commands.add_parser(
        "evolve",
        help="Integrate nonlinear ALE flow for formation, holding or perturbation recovery",
    )
    evolve.add_argument("result", type=Path)
    evolve.add_argument("--out", type=Path, required=True)
    evolve.add_argument("--controller", type=Path)
    evolve.add_argument(
        "--initial", choices=["rest", "rest_fixed", "hold", "perturbation"], default="perturbation"
    )
    evolve.add_argument("--step", type=float, default=0.005)
    evolve.add_argument("--duration", type=float, default=0.4)
    evolve.add_argument(
        "--perturbation",
        type=float,
        default=1e-6,
        help="Full-aperture RMS height perturbation, metres",
    )
    evolve.add_argument("--mode", type=int, default=0)
    evolve.add_argument(
        "--noise", type=float, default=0.0, help="Independent height-observation RMS noise, metres"
    )
    evolve.add_argument("--ramp", type=float, default=1.2)
    evolve.add_argument("--resolution", type=int, nargs=3, metavar=("NR", "NZ", "MODES"))
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

    if args.command == "evolve":
        from .lens.transient import run_transient

        run_transient(
            args.result,
            args.out,
            step_s=args.step,
            end_s=args.duration,
            initial=args.initial,
            perturbation_m=args.perturbation,
            perturbation_mode=args.mode,
            controller_directory=args.controller,
            noise_rms_m=args.noise,
            ramp_s=args.ramp,
            resolution=args.resolution,
        )
    elif args.command == "feedback":
        from .lens.feedback import design_feedback

        design_feedback(args.result, args.stability, args.out, args.decay, args.period)
    elif args.command == "acoustic-budget":
        from .verification.acoustic_budget import acoustic_budget

        acoustic_budget(args.result, args.out)
    elif args.command == "stabilize":
        from .lens.stable_design import run_stable_design

        run_stable_design(args.result, args.stability, args.out, args.margin)
    elif args.command in ("stability", "control-model"):
        from .lens.dynamics import linearize_for_control, stability_report
        from .lens.surface import SurfaceSpace

        cfg = LensConfig(**json.loads((args.result / "configuration.json").read_text()))
        state = np.load(args.result / "stationary.npz")
        analyze = stability_report if args.command == "stability" else linearize_for_control
        analyze(
            SurfaceSpace(cfg),
            state["coefficients"],
            state["drive_m_s"],
            args.out,
            args.difference_step,
        )
    elif args.command == "verify-coupled":
        from .verification.coupled import coupled_study

        kwargs = {} if args.level is None else {"levels": args.level}
        coupled_study(args.result, args.out, method=args.method, **kwargs)
    elif args.command == "screen-frequency":
        from .verification.frequency import frequency_study

        kwargs = {} if args.frequencies is None else {"frequencies": args.frequencies}
        frequency_study(args.result, args.out, **kwargs)
    elif args.command == "diagnose-cavity":
        from .verification.curved import fixed_state_study

        fixed_state_study(args.result, args.out)
    elif args.command == "simulate":
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
    elif args.command == "regrid-stationary":
        from .lens.stationary import regrid_stationary

        regrid_stationary(args.result, args.out, args.resolution)
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
