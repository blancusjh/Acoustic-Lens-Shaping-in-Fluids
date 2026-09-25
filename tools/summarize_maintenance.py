"""Audit and plot the completed Cartesian maintenance campaign; never run a solver."""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import matplotlib
import numpy as np

from acoustic_freeform.single_interface.config import LensConfig
from acoustic_freeform.single_interface.surface import SurfaceSpace

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_case(root, name):
    folder = root / name
    report = json.loads((folder / "report.json").read_text())
    data = dict(np.load(folder / "trajectory.npz"))
    cfg = LensConfig(**report["configuration"])
    space = SurfaceSpace(cfg)
    pupil = cfg.clear_radius_m / cfg.radius_m * np.sqrt((np.arange(801) + 0.5) / 801)
    data["height_m"] = np.array(
        [cfg.radius_m * space.evaluate(c, pupil) for c in data["coefficients"]]
    )
    return report, data, space


def compare(left, right):
    a, b = left[1], right[1]
    tmax = min(a["times_s"][-1], b["times_s"][-1])
    ia = np.flatnonzero(a["times_s"] <= tmax + 1e-12)
    ib = np.array([np.argmin(abs(b["times_s"] - a["times_s"][i])) for i in ia])
    if not np.allclose(a["times_s"][ia], b["times_s"][ib], atol=1e-12):
        raise ValueError("Comparison requires matching physical timestamps.")
    rms = np.sqrt(np.mean((a["height_m"][ia] - b["height_m"][ib]) ** 2, axis=1))
    return {
        "times_s": a["times_s"][ia].tolist(),
        "pupil_height_difference_rms_m": rms.tolist(),
        "maximum_pupil_rms_difference_m": float(rms.max()),
        "time_and_pupil_rms_difference_m": float(np.sqrt(np.mean(rms**2))),
        "last_pupil_rms_difference_m": float(rms[-1]),
        "last_common_time_s": float(tmax),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve()
    temporal = [
        "step-formation",
        "step-formation-dt0025",
        "step-formation-dt00125",
        "step-formation-dt000625",
    ]
    recoveries = ["fixed-drive-recovery", "recovery-negative-mode6"]
    names = temporal + ["step-formation-spatial", "formation"] + recoveries
    cases = {name: read_case(root, name) for name in names}
    holding = np.load(root / "operating-state/stationary.npz")["drive_m_s"]
    rows = []
    for name, (r, d, _space) in cases.items():
        history = r["history"]
        fixed = name != "formation"
        same_drive = np.array_equal(d["drive_m_s"], np.broadcast_to(holding, d["drive_m_s"].shape))
        assert same_drive or not fixed, f"Changed physical drive: {name}"
        assert np.all(np.isfinite(d["coefficients"])), name
        relative_volume = max(abs(x["volume_error_m3"]) for x in history) / r["liquid_volume_m3"]
        assert relative_volume < 1e-10, (name, relative_volume)
        rows.append(
            {
                "case": name,
                "duration_s": r["final"]["time_s"],
                "step_s": r["configuration"]["step_s"],
                "resolution": [
                    r["configuration"][k] for k in ("mesh_radial", "mesh_vertical", "surface_modes")
                ],
                "identical_fixed_drive_at_every_frame": same_drive,
                "initial_pupil_height_error_to_hold_m": r["initial"][
                    "shape_rms_to_holding_state_m"
                ],
                "final_pupil_height_error_to_hold_m": r["final"]["shape_rms_to_holding_state_m"],
                "final_ray_rms_m": r["final"]["rms_spot_at_target_m"],
                "final_opd_rms_m": r["final"]["opd_rms_m"],
                "maximum_relative_volume_error": relative_volume,
                "maximum_reynolds_number": max(x["flow_reynolds_number"] for x in history),
                "maximum_pressure_pa": max(x["peak_acoustic_pressure_pa"] for x in history),
                "maximum_source_power_w": max(x["source_power_w"] for x in history),
                "trajectory_sha256": hashlib.sha256(
                    (root / name / "trajectory.npz").read_bytes()
                ).hexdigest(),
            }
        )
    time_checks = [
        {"coarse": a, "fine": b, **compare(cases[a], cases[b])} for a, b in pairwise(temporal)
    ]
    mesh_check = compare(cases["step-formation-dt0025"], cases["step-formation-spatial"])
    mesh = json.loads((root / "viscous-1mhz/convergence/report.json").read_text())["cases"]
    budget = json.loads((root / "operating-state/acoustic-budget/report.json").read_text())
    out = root / "verification"
    out.mkdir(exist_ok=True)
    report = {
        "scope": "Finite-horizon nonlinear axisymmetric, isothermal numerical evidence. Fixed-drive formation/recovery with bulk absorption and leading viscous acoustic wall loss. No experimental or arbitrary-disturbance stability certificate.",
        "temporal_note": "Compare the full early trajectory as well as the final state. Stable terminal optics alone do not establish a precise settling time.",
        "runs": rows,
        "temporal_comparisons": time_checks,
        "spatial_trajectory_comparison": mesh_check,
        "stationary_mesh_cases": mesh,
        "energy_budget_w": {
            k: budget[k]
            for k in ("source_power_w", "volume_absorbed_power_w", "viscous_wall_dissipation_w")
        },
        "power_balance_relative_error": budget["power_balance_relative_error"],
        "missing_physics": [
            "thermal feedback",
            "wall-layer mean streaming",
            "3D azimuthal disturbances",
            "optical-index modulation",
            "elastic/piezo hardware calibration",
            "curing",
        ],
    }
    (out / "maintenance.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), layout="constrained")
    r, data, space = cases[temporal[-1]]
    radius = np.linspace(0, space.config.radius_m, 401)
    for c, label, color, ls in [
        (data["initial"], "Unforced liquid", "#6e7e90", "-"),
        (data["target"], "Cartesian target", "#d7952f", "--"),
        (data["coefficients"][-1], "Computed held surface", "#008c91", "-"),
    ]:
        axes[0, 0].plot(
            radius * 1000,
            space.config.radius_m * space.evaluate(c, radius / space.config.radius_m) * 1000,
            label=label,
            color=color,
            ls=ls,
        )
    axes[0, 0].axvline(3, color="#a7abb0", lw=1, ls=":")
    axes[0, 0].set(
        xlabel="Radius (mm)",
        ylabel="Height above rim (mm)",
        title="A: Formation at conserved liquid volume",
    )
    axes[0, 0].legend(fontsize=8)
    for name in temporal:
        rec = cases[name][0]
        axes[0, 1].semilogy(
            [x["time_s"] * 1000 for x in rec["history"]],
            [x["rms_spot_at_target_m"] * 1e6 for x in rec["history"]],
            label=f"Δt = {rec['configuration']['step_s'] * 1000:g} ms",
        )
    axes[0, 1].set(
        xlim=(0, 100),
        xlabel="Time after fixed-drive switch-on (ms)",
        ylabel="Geometric ray RMS (µm)",
        title="B: Time refinement; early motion still differs",
    )
    axes[0, 1].legend(fontsize=8)
    for name in recoveries:
        rec = cases[name][0]
        axes[1, 0].semilogy(
            [x["time_s"] * 1000 for x in rec["history"]],
            [max(x["shape_rms_to_holding_state_m"] * 1e9, 1e-6) for x in rec["history"]],
            label=f"{rec['perturbation_full_aperture_rms_m'] * 1e6:+g} µm, volume-null mode {rec['perturbation_volume_null_mode']}",
        )
    axes[1, 0].set(
        xlabel="Time after disturbance (ms)",
        ylabel="Pupil RMS height error to hold (nm)",
        title="C: Fixed-drive recovery, zero initial mean velocity",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(
        range(len(mesh)), [m["ray_rms_at_target_m"] * 1e6 for m in mesh], "o-", color="#008c91"
    )
    axes[1, 1].set_xticks(
        range(len(mesh)), [m["case"].replace("P4-", "").replace("-", "/") for m in mesh]
    )
    axes[1, 1].set(
        xlabel="Radial / axial base divisions / surface modes (P4)",
        ylabel="Geometric ray RMS (µm)",
        title="D: Same physical drive, independently solved shape",
        ylim=(0, 0.065),
    )
    row_ids = np.arange(1, len(holding) + 1)
    axes[2, 0].bar(row_ids, abs(holding), color="#008c91", alpha=0.75)
    phase = axes[2, 0].twinx()
    phase.plot(
        row_ids, np.degrees(np.angle(holding * holding[0].conjugate())), "o-", color="#b47423", ms=3
    )
    phase.set(ylabel="Phase relative to row 1 (degrees)", ylim=(-190, 190))
    axes[2, 0].set(
        xlabel="Array row from bottom",
        ylabel="Peak wall velocity (m/s)",
        title="E: Unchanged 1 MHz holding excitation",
    )
    axes[2, 1].bar(
        ["Source", "Bulk absorption", "Viscous wall loss"],
        list(report["energy_budget_w"].values()),
        color=["#49677c", "#008c91", "#d7952f"],
    )
    axes[2, 1].set(ylabel="Acoustic power (W)", title="F: Energy balance at held geometry")
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle(
        "Cartesian diopter: fixed-drive formation and recovery in the declared numerical model\n"
        "nₒ=1.52, zₒ=−50 mm → nᵢ=1, zᵢ=+20 mm · 6 mm clear aperture · isothermal, axisymmetric",
        fontsize=13,
    )
    fig.savefig(out / "maintenance.png", dpi=180)
    fig.savefig(out / "maintenance.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                "runs": rows,
                "time_comparison_max_um": [
                    x["maximum_pupil_rms_difference_m"] * 1e6 for x in time_checks
                ],
                "spatial_trajectory_max_nm": mesh_check["maximum_pupil_rms_difference_m"] * 1e9,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
