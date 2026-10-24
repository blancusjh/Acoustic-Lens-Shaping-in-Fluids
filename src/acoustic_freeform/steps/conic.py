"""Sequential convex source steps with true quadratic merit checks.

The target, frequency and apparatus remain fixed throughout optimization.
Iterations are not continuation of targets or physical time evolution.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..inverse.conic import optimize


def add_arguments(parser):
    parser.add_argument("config", type=Path)


def cli(args):
    inputs = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    capture_execution(args.out)
    (args.out / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    cache_dir = data_path(inputs["cache_directory"])
    source_path = cache_dir / "frozen-operator.npz"
    cache = dict(np.load(source_path))
    observation_path = inputs.get("observation_operator_file")
    if observation_path:
        cache.update(dict(np.load(data_path(observation_path))))
    source = np.load(data_path(inputs["initial_drive_file"]))["source_velocity_m_s"]
    cfg = json.loads((cache_dir / "resolved-config.json").read_text())
    (args.out / "validation.json").write_text(
        json.dumps(
            {
                "cache_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "observation_sha256": (
                    hashlib.sha256(Path(observation_path).read_bytes()).hexdigest()
                    if observation_path
                    else None
                ),
                "independent_forward_solve": False,
                "scope": "Frozen-target minimax screening only; refinement and stability untested.",
            },
            indent=2,
        )
        + "\n"
    )

    def checkpoint(command, history):
        np.savez_compressed(
            args.out / "candidate.npz",
            source_velocity_m_s=command,
            target_coefficients_m=cache["target_coefficients_m"],
        )
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    command, history = optimize(
        cache,
        source,
        cfg["max_source_speed_m_s"],
        inputs.get("steps", 100),
        inputs.get("cutoff", 1e-12),
        inputs.get("annulus_weight", 0.05),
        checkpoint,
        inputs.get("precondition", False),
        include_carrier=inputs.get("include_carrier", True),
        clear_radius_m=cfg.get("clear_radius_m"),
        front_rim_upper_bound_m=inputs.get("front_rim_upper_bound_m"),
    )
    checkpoint(command, history)
    (args.out / "report.md").write_text(
        "# Sequential convex frozen-target inverse\n\n"
        + json.dumps(history[-1], indent=2)
        + "\n\nNo coupled surface, physical trajectory or stability is certified. "
        "The reported combined metric is a conservative sum of separate sampled maxima.\n"
    )
