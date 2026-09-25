"""Independent saved-state energy/refinement audit; preserves original outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.core.provenance import capture_execution
from acoustic_freeform.mechanics.surface import DualSurface


def run(directories, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = [json.loads((d / "config.json").read_text()) for d in directories]
    trajectories = [np.load(d / "trajectory.npz") for d in directories]
    reports = [json.loads((d / "validation.json").read_text()) for d in directories]
    (output / "config.json").write_text(
        json.dumps({"runs": list(map(str, directories))}, indent=2) + "\n"
    )
    capture_execution(output)
    identity = [
        "apparatus",
        "indices",
        "stigmatic_z_m",
        "vertex_displacement_m",
        "viscosities_pa_s",
        "duration_s",
    ]
    for config in settings:
        for key in identity:
            assert config[key] == settings[0][key], f"Changed physical/surface input: {key}"
    cfg = DualConfig(**settings[0]["apparatus"])
    space = DualSurface(cfg)
    force = np.load(directories[0] / "prescribed-load.npz")["generalized_force_n"]
    qstar = space.prescribed_equilibrium(force)
    eq_energy = space.mechanics(qstar)[2] - float(force @ qstar.ravel())
    radius = np.linspace(0, cfg.radius_m, 8001)
    checks = []
    for directory, setting, data, report in zip(directories, settings, trajectories, reports):
        np.testing.assert_array_equal(
            force, np.load(directory / "prescribed-load.npz")["generalized_force_n"]
        )
        assert np.isclose(data["time_s"][-1], setting["duration_s"])
        assert len(data["time_s"]) == round(setting["duration_s"] / setting["step_s"]) + 1
        computed = np.array(
            [space.mechanics(q)[2] - float(force @ q.ravel()) for q in data["coefficients_m"]]
        )
        np.testing.assert_allclose(data["loaded_energy_j"], computed, atol=1e-18, rtol=1e-12)
        assert np.max(np.diff(computed)) <= 1e-16
        assert computed.min() >= eq_energy - 1e-16
        missing_diagnostic_export = not (directory / "fluid-diagnostics.json").exists()
        status = {
            "numerical_time_steps_complete": True,
            "independent_saved_energy_check_passed": True,
            "original_report_export_failed": missing_diagnostic_export,
            "reason": (
                "Original process exited after saving trajectory and validation: NumPy int32 in "
                "fluid diagnostics was not JSON serializable. No time steps were lost. "
                "Per-step fluid diagnostics were not saved and are not reconstructed here."
                if missing_diagnostic_export
                else "Original diagnostic exports present"
            ),
            "audited_in": str(output),
        }
        if not (directory / "execution-status.json").exists():
            (directory / "execution-status.json").write_text(json.dumps(status, indent=2) + "\n")
        if not (directory / "report.md").exists():
            (directory / "report.md").write_text(
                "# Prescribed-load viscous formation\n\n"
                "Numerical trajectory completed; original diagnostics export failed. "
                "See execution-status.json for recovery scope. Saved states and validation "
                "are preserved unchanged. No physical or acoustic certification.\n\n```json\n"
                + json.dumps(report, indent=2)
                + "\n```\n"
            )
        checks.append(
            {
                "run": str(directory),
                "final": report["observations"][-1],
                "final_4001_ray_max_radius_m": report["final_max_geometric_spot_radius_m"],
                "final_4001_rays_all_transmitted": report["final_all_rays_transmitted"],
                "energy_above_equilibrium_final_j": float(computed[-1] - eq_energy),
                "energy_monotone": True,
                "status": status,
            }
        )
    comparisons = []
    for a, b in zip(range(len(directories) - 1), range(1, len(directories))):
        ta, tb = trajectories[a], trajectories[b]
        ia, ib = [], []
        for i, t in enumerate(ta["time_s"]):
            j = round(t / settings[b]["step_s"])
            if j < len(tb["time_s"]) and np.isclose(tb["time_s"][j], t):
                ia.append(i)
                ib.append(j)
        differences = ta["coefficients_m"][ia] - tb["coefficients_m"][ib]
        dh = np.einsum("rj,tfj->tfr", space.basis(radius), differences)
        comparisons.append(
            {
                "a": str(directories[a]),
                "b": str(directories[b]),
                "maximum_common_time_height_difference_m": np.max(abs(dh), axis=(0, 2)).tolist(),
                "final_height_difference_m": np.max(abs(dh[-1]), axis=1).tolist(),
            }
        )
    result = {
        "checks": checks,
        "comparisons": comparisons,
        "source_files_sha256": {
            str(d / f): hashlib.sha256((d / f).read_bytes()).hexdigest()
            for d in directories
            for f in ("config.json", "trajectory.npz", "validation.json")
        },
        "interpretation": "Fixed prescribed-load numerical checks only; no continuum/physical accuracy certification",
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Viscous ideal-load verification\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n"
    )
    print(json.dumps({"comparisons": comparisons, "checks": checks}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.runs, args.out)
