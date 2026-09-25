"""Offline two-face time playback, using saved forward states without morphing."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ..apparatus.config import DualConfig
from ..apparatus.sources import source_regions
from ..core.paths import data_path
from ..forward.inertial_formation import envelope
from ..mechanics.surface import CartesianPatch, DualSurface
from ..optics.raytrace import trace_pair


def prepare(root, destination):
    root, destination = Path(root), Path(destination)
    run = root / "artifacts/studies/S02-independent-two-face/dual-formation-2026-09-23/dt000125"
    coarse = root / "artifacts/studies/S02-independent-two-face/dual-formation-2026-09-23/dt00025"
    cfg = json.loads((run / "config.json").read_text())
    if not json.loads((run / "validation.json").read_text())["completed"]:
        raise ValueError("Selected run is incomplete")
    apparatus = DualConfig(**cfg["apparatus"])
    space = DualSurface(apparatus)
    saved = np.load(run / "trajectory.npz")
    comparison = np.load(coarse / "trajectory.npz")
    time, q = saved["time_s"], saved["coefficients_m"]
    r = np.linspace(0, apparatus.radius_m, 401)
    pupil = r <= apparatus.clear_radius_m
    profiles = np.einsum("rc,tfc->tfr", space.basis(r), q)
    target_config = json.loads(
        (root / "artifacts/studies/S02-independent-two-face/dual-cartesian-2026-09-22/sources-112/config.json").read_text()
    )["cases"][0]["faces"]
    target = np.array(
        [CartesianPatch(apparatus, j, **data).evaluate(r) for j, data in enumerate(target_config)]
    )
    error = profiles[:, :, pupil] - target[:, pupil]
    peak = np.max(abs(error), axis=2)
    traced = [trace_pair(space, state, **cfg["optical"]) for state in q]
    spots = np.array([item["spots_m"] for item in traced])
    rms = np.array([item["rms_radius_m"] for item in traced])
    matched = np.searchsorted(time, comparison["time_s"])
    np.testing.assert_allclose(time[matched], comparison["time_s"], atol=1e-14)
    difference = np.einsum("rc,tfc->tfr", space.basis(r), q[matched] - comparison["coefficients_m"])
    source = data_path(cfg["source_state"])
    drive = np.load(source)["source_velocity_m_s"]
    checks = {
        "scope": cfg["scope"],
        "formation_interval_s": [float(time[0]), float(time[-1])],
        "frames": len(time),
        "both_interfaces_move": bool(np.all(np.max(abs(profiles[-1]), axis=1) > 0)),
        "fixed_command_time_step_discrepancy_max_m": np.max(abs(difference), axis=(0, 2)).tolist(),
        "final_sampled_pupil_error_m": peak[-1].tolist(),
        "minimum_joint_sampled_pupil_error_m": float(np.min(np.max(peak, axis=1))),
        "sampled_error_is_not_continuous_maximum_certificate": True,
        "final_spot_rms_radius_m": float(rms[-1]),
        "transmitted_ray_count_initial_final": [
            int(traced[0]["transmitted"].sum()),
            int(traced[-1]["transmitted"].sum()),
        ],
        "source_regions": len(drive),
        "spot_detector_z_m": cfg["optical"]["detector_z_m"],
        "spot_object_z_m": cfg["optical"]["object_z_m"],
        "optical_scope": "On-axis geometric rays through both modeled interfaces; chamber-window optics not implemented. Pair of independent Cartesian targets is not a whole-lens stigmatism design.",
        "ten_nm_verified": False,
        "dynamic_mode_and_acoustic_mesh_convergence": "Not established",
        "earlier_longer_runs": "dt001 and dt0005 stopped at 300 um displacement guard before ramp completion",
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (destination / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    inputs = [
        run / "trajectory.npz",
        run / "config.json",
        coarse / "trajectory.npz",
        source,
        Path(__file__),
        Path(__file__).with_name("formation_viewer.html"),
        Path(__file__).with_name("optics.py"),
    ]
    (destination / "provenance.json").write_text(
        json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, indent=2)
        + "\n"
    )
    np.savez_compressed(
        destination / "display-data-si.npz",
        time_s=time,
        radius_m=r,
        displacement_m=profiles,
        target_displacement_m=target,
        pupil_error_m=error,
        spot_xy_m=spots,
        spot_rms_radius_m=rms,
    )
    data = {
        "t": time.tolist(),
        "r": r.tolist(),
        "profile": profiles.tolist(),
        "target": target.tolist(),
        "error": error.tolist(),
        "peak": peak.tolist(),
        "rms": rms.tolist(),
        "spots": [
            [point.tolist() if np.isfinite(point).all() else None for point in frame]
            for frame in spots
        ],
        "levels": apparatus.levels_m,
        "R": apparatus.radius_m,
        "a": apparatus.clear_radius_m,
        "regions": source_regions(apparatus),
        "drive": abs(drive).tolist(),
        "envelope": [envelope(t, cfg["ramp_s"]) for t in time],
        "checks": checks,
    }
    html = (
        Path(__file__)
        .with_name("formation_viewer.html")
        .read_text()
        .replace("__DATA__", json.dumps(data, allow_nan=False))
    )
    (destination / "viewer.html").write_text(html)
    (destination / "report.md").write_text(
        "# Two-face formation playback\n\n" + json.dumps(checks, indent=2) + "\n"
    )
    return html, checks
