"""Independent driven wave versus cached all-source quadratic traction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..provenance import capture_execution
from .acoustics import DualAcoustics
from .config import DualConfig
from .surface import DualSurface


def run(configuration, output):
    inputs = json.loads(configuration.read_text())
    cache_dir = Path(inputs["cache_directory"])
    cache_path = cache_dir / "frozen-operator.npz"
    command_path = Path(inputs["source_file"])
    cfg = DualConfig(**json.loads((cache_dir / "resolved-config.json").read_text()))
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    capture_execution(output)
    provenance = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [cache_path, command_path]
    }
    (output / "input-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with np.load(cache_path) as cache:
        q = cache["target_coefficients_m"]
        drive = np.load(command_path)["source_velocity_m_s"]
        kernels = cache["force_kernels"]
        cached_force = np.einsum("a,iab,b->i", drive.conj(), kernels, drive).real
    del kernels
    space = DualSurface(cfg)
    required, stiffness, _ = space.mechanics(q)
    response = DualAcoustics(cfg, space, linear_solver="static_condensed").solve(q, drive)
    fresh_force = response.force(np.ones(1, complex))

    def errors(force, pupil):
        delta = np.linalg.solve(stiffness, force).reshape(2, -1)
        return [
            space.polynomial_maximum(v, upper_m=cfg.clear_radius_m if pupil else cfg.radius_m)[
                "max_abs_m"
            ]
            for v in delta
        ]

    report = {
        "cached_defect_pupil_m": errors(cached_force - required, True),
        "fresh_defect_pupil_m": errors(fresh_force - required, True),
        "fresh_defect_full_surface_m": errors(fresh_force - required, False),
        "fresh_minus_cached_pupil_m": errors(fresh_force - cached_force, True),
        "fresh_minus_cached_full_surface_m": errors(fresh_force - cached_force, False),
        "wave_diagnostics": response.diagnostics(np.ones(1, complex)),
        "scope": "Fixed target geometry and command; force-compliance diagnostics, not equilibrium.",
    }
    np.savez_compressed(
        output / "force-comparison.npz",
        coefficients_m=q,
        source_velocity_m_s=drive,
        cached_force_n=cached_force,
        fresh_force_n=fresh_force,
        required_force_n=required,
    )
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Independent driven-wave/cache agreement\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)
