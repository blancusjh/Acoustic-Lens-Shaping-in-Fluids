"""Compose existing height/spot observations with a full static response matrix."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..provenance import capture_execution


def replace_mechanical_compliance(observe, stiffness, coupled_compliance):
    """O K^-1 -> O (K-A)^-1; offsets and physical observation units are unchanged."""
    return observe @ stiffness @ coupled_compliance


def run(configuration, output):
    inputs = json.loads(configuration.read_text())
    response_path = Path(inputs["linearization_file"])
    observation_path = Path(inputs["observation_file"])
    cache_path = Path(inputs["cache_file"])
    report_path = response_path.parent / "validation.json"
    report = json.loads(report_path.read_text())
    if report["acoustic_shape_directions_truncated"]:
        raise ValueError("This experiment requires the complete discrete shape Jacobian")
    checks = report["independent_directional_relative_errors"]
    if not checks or max(checks) > inputs.get("maximum_directional_relative_error", 1e-5):
        raise ValueError("Independent directional derivative checks are missing or unresolved")
    with np.load(response_path) as response, np.load(cache_path) as cache:
        np.testing.assert_array_equal(response["coefficients_m"], cache["target_coefficients_m"])
        observation = dict(np.load(observation_path))
        observation["compliance_observation_m_n"] = replace_mechanical_compliance(
            observation["compliance_observation_m_n"],
            response["mechanical_stiffness_n_m"],
            response["compliance_m_n"],
        )
    observation["observation_response_kind"] = np.array("full_coupled_static_at_reference_command")
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    (output / "input-provenance.json").write_text(
        json.dumps(
            {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [response_path, observation_path, cache_path, report_path]
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(output / "observation-operator.npz", **observation)
    validation = {
        "observations": len(observation["observation_faces"]),
        "full_shape_dimension": report["full_shape_dimension"],
        "source_optimization_executed": False,
        "coupled_equilibrium_executed": False,
        "scope": "Local full-static response at one reference command. Finite source changes still require fresh coupled solves and updated derivatives when needed.",
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Full-static local observations\n\n" + json.dumps(validation, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)
