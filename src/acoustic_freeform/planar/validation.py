"""Repeatable convergence studies; error measures are not accuracy guarantees."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import resample

from .acoustics import PlanarAcoustics
from .array import PressureArray
from .config import Grid, load_experiment
from .reference import rayleigh_incident_pressure
from .simulation import run
from .spectral import SpectralGrid


def relative_l2(a, b):
    return float(
        np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
        / max(np.linalg.norm(b), 1e-30)
    )


def validate_forward_model(directory: Path) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[4]
    water = load_experiment(root / "configs/verification/water_air.toml")
    stokes = load_experiment(root / "configs/verification/stokes_pair.toml")
    report = {
        "scope": "Numerical verification of the declared linear, periodic reference only; "
        "not a material calibration or experimental error estimate."
    }
    print("Checking independent Rayleigh integral...", flush=True)
    array = PressureArray(water.array, SpectralGrid(water.grid), water.lower.sound_speed_m_s)
    weights = array.beam_weights(water.beams[0])
    # Prespecified probes across both foci and surrounding region, not selected from results.
    xy = np.array(
        [[0, 0], [-0.00175, -0.000875], [0.0013125, 0.00109375], [0.0035, 0], [0, 0.0035]]
    )
    exact = rayleigh_incident_pressure(array, weights, xy, order=10)
    coarse = rayleigh_incident_pressure(array, weights, xy, order=6)
    acoustic = []
    for size in (0.028, 0.056, 0.084, 0.112):
        n = round(size / 0.028 * 128)
        g = SpectralGrid(Grid(n, n, size, size))
        a = PressureArray(water.array, g, water.lower.sound_speed_m_s)
        s = PlanarAcoustics(g, water.array, water.lower, water.upper).solve(a.spectrum(weights))
        values = np.array(
            [np.sum(s.incident * np.exp(1j * (g.kxx * x + g.kyy * y))) for x, y in xy]
        )
        acoustic.append(
            {
                "box_m": size,
                "n": n,
                "complex_pressure_relative_l2": float(
                    np.linalg.norm(values - exact) / np.linalg.norm(exact)
                ),
            }
        )
    report["independent_incident_acoustics"] = {
        "reference": "Unbounded Rayleigh half-space integral, square patches, order 10",
        "quadrature_order6_vs10_relative_l2": float(
            np.linalg.norm(coarse - exact) / np.linalg.norm(exact)
        ),
        "probes_xy_m": xy.tolist(),
        "periodic_angular_spectrum": acoustic,
        "interpretation": "Difference includes periodic images and omitted evanescent waves; "
        "five incident-pressure probes do not bound full surface-shape error.",
    }
    print("Checking smooth-forcing time refinement...", flush=True)
    time_results = []
    steps = [0.0004, 0.0002, 0.0001, 0.00005]
    for dt in steps:
        e = replace(water, timing=replace(water.timing, step_s=dt, save_every=round(0.004 / dt)))
        r = run(e)
        time_results.append(r.height_m)
    time_errors = [relative_l2(h, time_results[-1]) for h in time_results[:-1]]
    report["time_refinement"] = {
        "reference_dt_s": steps[-1],
        "norm": "Space-time L2 of height, all domain, every 4 ms",
        "levels": [
            {"dt_s": dt, "relative_l2": error}
            for dt, error in zip(steps[:-1], time_errors, strict=True)
        ],
        "coarse_to_middle_error_ratio": time_errors[0] / max(time_errors[1], 1e-30),
    }
    del time_results
    print("Checking spatial resolution...", flush=True)
    resolutions = [128, 192, 256]
    spatial = []
    for n in resolutions:
        e = replace(
            water, grid=replace(water.grid, nx=n, ny=n), timing=replace(water.timing, save_every=40)
        )
        spatial.append(run(e).height_m)
    reference = spatial[-1]
    report["spatial_refinement"] = {
        "reference_n": resolutions[-1],
        "box_m": water.grid.lx_m,
        "norm": "Space-time L2, fine periodic field Fourier-resampled to each grid",
        "levels": [
            {
                "n": n,
                "relative_l2": relative_l2(
                    values, resample(resample(reference, n, axis=1), n, axis=2)
                ),
            }
            for n, values in zip(resolutions[:-1], spatial[:-1], strict=True)
        ],
        "interpretation": "With propagating-only acoustics and linear dynamics, spatial spectra "
        "are band limited. Saturation above Nyquist coverage is expected.",
    }
    del spatial, reference
    print("Checking domain enlargement for both fluid limits...", flush=True)
    report["domain_enlargement"] = {}
    for experiment in (water, stokes):
        regions = []
        for factor in (1, 2, 3):
            e = replace(
                experiment,
                grid=Grid(128 * factor, 128 * factor, 0.028 * factor, 0.028 * factor),
                timing=replace(
                    experiment.timing, save_every=round(0.005 / experiment.timing.step_s)
                ),
            )
            r = run(e)
            indices = np.flatnonzero(np.abs(r.grid.x) <= 0.004)
            regions.append(r.height_m[:, indices][:, :, indices])
        report["domain_enlargement"][experiment.name] = {
            "reference_box_m": 0.084,
            "spacing_m": 0.028 / 128,
            "end_s": experiment.timing.end_s,
            "roi_half_width_m": 0.004,
            "norm": "Space-time L2 of central height, every 5 ms, same physical samples",
            "levels": [
                {"box_m": 0.028 * (i + 1), "relative_l2": relative_l2(h, regions[-1])}
                for i, h in enumerate(regions[:-1])
            ],
            "interpretation": "Changing the box changes acoustic replicas, hydrodynamic wave "
            "wraparound, and the finite-cell volume constraint together.",
        }
    report["checks"] = {
        "rayleigh_quadrature_converged_below_1e-8": bool(
            report["independent_incident_acoustics"]["quadrature_order6_vs10_relative_l2"] < 1e-8
        ),
        "time_errors_decrease": bool(time_errors[0] > time_errors[1] > time_errors[2]),
        "time_ratio_compatible_with_second_order": bool(
            3.0 < time_errors[0] / time_errors[1] < 6.0
        ),
    }
    (directory / "convergence.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if not all(report["checks"].values()):
        raise AssertionError(
            f"Numerical verification failed; inspect {directory / 'convergence.json'}"
        )
    return report
