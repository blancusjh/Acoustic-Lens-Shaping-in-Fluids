"""Entry points for the implemented, model-specific research steps.

These commands are not an end-to-end general acoustic-source compiler. Each
configuration declares the approximation and acceptance protocol of its step.
"""

import argparse
import os
from importlib import import_module
from pathlib import Path

from .core.paths import REPOSITORY

COMMANDS = {
    "target": ("verify.stigmatic_target", "Validate a shared-conjugate optical target"),
    "required-load": ("mechanics.required_load", "Compute the prescribed mechanical load"),
    "synthesize": ("inverse.force_synthesis", "Fit bounded finite source commands"),
    "equilibrate": ("forward.held_command", "Solve both interfaces at a held command"),
    "verify-wave": ("verify.fixed_geometry", "Refine the wave at fixed geometry and command"),
    "form": ("forward.emitter_formation", "Run the declared emitter-driven Stokes diagnostic"),
    "independent-pair": ("forward.campaign", "Reproduce the earlier independent-pair campaign"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_module, help_text) in COMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("config", type=Path, help="existing SI JSON configuration")
        sub.add_argument("--out", required=True, type=Path, help="new output directory under artifacts/")
    subparsers.add_parser("status", help="show the reviewed research status")
    args = parser.parse_args(argv)
    if args.command == "status":
        print((REPOSITORY / "STATUS.md").read_text(), end="")
        return
    config = args.config.resolve()
    if not config.is_file():
        parser.error(f"Configuration does not exist: {config}")
    output = args.out.resolve()
    if not output.is_relative_to(REPOSITORY / "artifacts"):
        parser.error("Research output must be under artifacts/")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    os.chdir(REPOSITORY)
    module = import_module("acoustic_freeform." + COMMANDS[args.command][0])
    module.run(config, output)


if __name__ == "__main__":
    main()
