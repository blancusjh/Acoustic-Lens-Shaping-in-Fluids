"""Entry points for the implemented, model-specific research steps.

These commands are not an end-to-end general acoustic-source compiler. Each
configuration declares the approximation and acceptance protocol of its step.
Every step writes one new directory under artifacts/.
"""

import argparse
import os
import sys
from importlib import import_module
from pathlib import Path

from .core.paths import REPOSITORY, data_path

# Command -> (module in acoustic_freeform.steps, help), in research-path order.
STEPS = {
    "target": ("target", "Validate a shared-conjugate optical target"),
    "required-load": ("required_load", "Compute the prescribed mechanical load"),
    "ideal-load-formation": ("ideal_load_formation", "Creeping flow under a fixed ideal load"),
    "screen": ("screen", "Build the frozen operator and screen source commands"),
    "synthesize": ("synthesize", "Fit bounded finite source commands"),
    "conic": ("conic", "Sequential convex frozen-target source steps"),
    "relaxation": ("relaxation", "Semidefinite source diagnostic on a frozen operator"),
    "free-annulus": ("free_annulus", "Coupled inverse with fixed pupils and free annuli"),
    "observations": ("observations", "Compose observations with a static response matrix"),
    "equilibrate": ("equilibrate", "Solve both interfaces at a held command"),
    "independent-pair": ("independent_pair", "Reproduce the earlier independent-pair campaign"),
    "form": ("form", "Run the declared emitter-driven Stokes diagnostic"),
    "inertial-formation": ("inertial_formation", "Flat-domain potential-inertia formation"),
    "verify-wave": ("verify_wave", "Refine the wave at fixed geometry and command"),
    "analytic-limits": ("analytic_limits", "Planar slab and Bessel analytic limits"),
    "kernel-audit": ("kernel_audit", "Driven wave versus cached quadratic traction"),
    "analytic-response": ("analytic_response", "Acoustic shape Jacobian and static response"),
    "coupled-response": ("coupled_response", "Low-mode acoustic shape feedback"),
    "frozen-extremum": ("frozen_extremum", "Polynomial extremum audit of a frozen fit"),
    "stigmatic-state": ("stigmatic_state", "Joint optical gates for saved equilibria"),
}

# Earlier, separately scoped models keep their own command lines.
MODELS = {
    "single-interface": ("models.single_interface.cli", "Single-interface model (S01; `lenslab`)"),
    "planar": ("models.planar.cli", "Planar-interface verification limits"),
}


def _step(name):
    return import_module("acoustic_freeform.steps." + STEPS[name][0])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] and argv[0] in MODELS:
        module = import_module("acoustic_freeform." + MODELS[argv[0]][0])
        return module.main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_module, help_text) in STEPS.items():
        sub = subparsers.add_parser(name, help=help_text)
        # Only the selected step is imported; listing commands stays cheap.
        module = _step(name) if argv[:1] == [name] else None
        if hasattr(module, "add_arguments"):
            module.add_arguments(sub)
        else:
            sub.add_argument("config", type=Path, help="existing SI JSON configuration")
        sub.add_argument(
            "--out", required=True, type=Path, help="new output directory under artifacts/"
        )
    for name, (_module, help_text) in MODELS.items():
        subparsers.add_parser(name, help=help_text, add_help=False)
    subparsers.add_parser("status", help="show the reviewed research status")
    args = parser.parse_args(argv)
    if args.command == "status":
        print((REPOSITORY / "STATUS.md").read_text(), end="")
        return
    for key, value in vars(args).items():
        if isinstance(value, Path) and key != "out":
            resolved = value.resolve() if value.exists() else data_path(value)
            if not resolved.exists():
                parser.error(f"Input does not exist: {value}")
            setattr(args, key, resolved)
    args.out = args.out.resolve()
    if not args.out.is_relative_to(REPOSITORY / "artifacts"):
        parser.error("Research output must be under artifacts/")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    os.chdir(REPOSITORY)
    module = _step(args.command)
    if hasattr(module, "cli"):
        module.cli(args)
    else:
        module.run(args.config, args.out)


if __name__ == "__main__":
    main()
