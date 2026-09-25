"""Surface-observation feedback derived from the unstable fluid/acoustic mode.

Control values are physical complex wall velocities. A rank-one stabilizer is
constructed from the left unstable Stokes eigenvector, then checked against the
complete inertial reduction and sampled-data delay model. Multiple unstable
directions require a different synthesis, not silent reuse of this controller.
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig, expm

from .config import LensConfig
from .dynamics import state_matrices
from .surface import SurfaceSpace


def design_feedback(source, stability_directory, destination, decay_s_inv=40.0, period_s=0.005):
    source, stability, out = Path(source), Path(stability_directory), Path(destination)
    if (out / "report.json").exists():
        raise FileExistsError(f"Completed controller exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = LensConfig(**json.loads((source / "configuration.json").read_text()))
    space = SurfaceSpace(cfg)
    derivative = dict(np.load(stability / "linearization.npz"))
    linearization_scope = json.loads((stability / "report.json").read_text()).get("scope", "")
    fluid = dict(np.load(stability / "fluid-reduction.npz"))
    mobility = fluid["static_mobility"]
    a_stokes = -mobility @ derivative["effective_stiffness"] / cfg.capillary_time_s
    b_stokes = mobility @ derivative["actuator_jacobian"] / cfg.capillary_time_s
    values, left = eig(a_stokes, left=True, right=False)
    unstable = np.flatnonzero(values.real > 0)
    if len(unstable) == 0:
        sensor_mode = np.zeros(a_stokes.shape[0])
        gain = np.zeros((b_stokes.shape[1], a_stokes.shape[0]))
    else:
        if len(unstable) != 1 or abs(values[unstable[0]].imag) > 1e-6:
            raise ValueError("This synthesis supports zero or one real unstable mode.")
        k = unstable[0]
        sensor_mode = left[:, k].real
        sensor_mode /= np.linalg.norm(sensor_mode)
        authority = b_stokes.T @ sensor_mode
        if np.linalg.norm(authority) < 1e-12:
            raise ValueError("The unstable mode is not controllable through this array.")
        gain = (
            (values[k].real + decay_s_inv)
            / np.dot(authority, authority)
            * np.outer(authority, sensor_mode)
        )
    a, b = state_matrices(
        fluid,
        derivative["effective_stiffness"],
        derivative["actuator_jacobian"],
        cfg.capillary_time_s,
    )
    full_gain = np.hstack([gain, np.zeros((gain.shape[0], a.shape[0] - gain.shape[1]))])
    closed = a - b @ full_gain
    rates = np.linalg.eigvals(closed)
    samples = []
    for period in sorted({period_s, 0.0025, 0.005, 0.01, 0.02, 0.04}):
        n, m = b.shape
        hold = expm(np.block([[a, b], [np.zeros((m, n + m))]]) * period)
        ad, bd = hold[:n, :n], hold[:n, n:]
        now = np.linalg.eigvals(ad - bd @ full_gain)
        delayed = np.linalg.eigvals(np.block([[ad, bd], [-full_gain, np.zeros((m, m))]]))
        samples.append(
            {
                "sample_period_s": period,
                "zero_delay_spectral_radius": float(abs(now).max()),
                "one_sample_delay_spectral_radius": float(abs(delayed).max()),
            }
        )
    chosen = next(row for row in samples if row["sample_period_s"] == period_s)
    if rates.real.max() >= 0 or chosen["one_sample_delay_spectral_radius"] >= 1:
        raise RuntimeError(
            "The proposed shape-only feedback failed its inertial/sample-delay check."
        )
    # Full-surface height observations; station placement resolves the rim modes.
    angles = np.pi * (np.arange(192) + 0.5) / 192
    stations = np.sqrt((1 - np.cos(angles)) / 2)
    observation = cfg.radius_m * space.basis(stations) @ space.tangent
    reconstruction = np.linalg.pinv(observation)
    np.savez_compressed(
        out / "controller.npz",
        gain=gain,
        sensor_mode=sensor_mode,
        radial_stations_m=stations * cfg.radius_m,
        reconstruction=reconstruction,
        reference_coefficients=derivative["coefficients"],
        reference_drive_m_s=derivative["drive_m_s"],
        sample_period_s=period_s,
    )
    report = {
        "mode": "fixed_drive" if len(unstable) == 0 else "surface_feedback",
        "scope": "Holding-drive or shape-feedback analysis for the recorded reduced model. Its approximations are given in linearization_scope. A nonzero feedback gain is checked with sample-and-hold and one sample of delay. Full nonlinear streaming trajectories require separate verification; no experimental calibration is claimed.",
        "linearization_scope": linearization_scope,
        "source": str(source.resolve()),
        "stability_source": str(stability.resolve()),
        "requested_stokes_mode_decay_s_inv": decay_s_inv,
        "inertial_closed_loop_largest_real_rate_s_inv": float(rates.real.max()),
        "sample_period_s": period_s,
        "sampled_checks": samples,
        "height_observation_stations": len(stations),
        "observation_condition_number": float(np.linalg.cond(observation)),
        "velocity_measurement_required": False,
        "gain_units": "Wall velocity [m/s] per dimensionless volume-null surface coefficient.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)
    return report
