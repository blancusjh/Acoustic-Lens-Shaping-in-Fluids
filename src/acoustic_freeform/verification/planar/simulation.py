"""Reproducible forward experiments and physical diagnostics, without optimization."""

import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .acoustics import PlanarAcoustics, RadiationBasis
from .array import PressureArray
from .config import Experiment
from .interface import SurfaceDynamics
from .spectral import SpectralGrid


@dataclass
class Result:
    experiment: Experiment
    grid: SpectralGrid
    time_s: np.ndarray
    height_m: np.ndarray
    velocity_m_s: np.ndarray
    amplitudes: np.ndarray
    radiation_kernels_pa: np.ndarray
    element_weights: np.ndarray
    metrics: dict[str, np.ndarray]
    report: dict
    acoustic_model: PlanarAcoustics
    pressure_array: PressureArray
    dynamics: SurfaceDynamics

    def pressure(self, frame: int) -> np.ndarray:
        a = self.amplitudes[frame]
        return np.einsum("i,j,ijyx->yx", a, a, self.radiation_kernels_pa)

    def combined_weights(self, frame: int) -> np.ndarray:
        return np.einsum("i,iyx->yx", self.amplitudes[frame], self.element_weights)


def beam_amplitudes(experiment: Experiment, time_s: float) -> np.ndarray:
    index = {b.name: i for i, b in enumerate(experiment.beams)}
    amplitudes = np.zeros(len(index))
    for pulse in experiment.pulses:
        amplitudes[index[pulse.beam]] += pulse.amplitude(time_s)
    return amplitudes


def run(experiment: Experiment, element_weights: np.ndarray | None = None) -> Result:
    """Simulate foci-derived weights, or a supplied complex (beam, row, col) array.

    Explicit weights expose every element independently. They are saved in the
    result for reproduction; the TOML foci are ignored when weights are supplied.
    """
    experiment.validate()
    started = time.perf_counter()
    g = SpectralGrid(experiment.grid)
    a = PressureArray(experiment.array, g, experiment.lower.sound_speed_m_s)
    acoustic = PlanarAcoustics(g, experiment.array, experiment.lower, experiment.upper)
    weights = (
        np.stack([a.beam_weights(b) for b in experiment.beams])
        if element_weights is None
        else np.asarray(element_weights, dtype=complex).copy()
    )
    expected_shape = (len(experiment.beams), experiment.array.ny, experiment.array.nx)
    if weights.shape != expected_shape or not np.isfinite(weights).all():
        raise ValueError(f"Expected finite element weights of shape {expected_shape}.")
    if np.max(np.abs(weights)) > 1 + 1e-12:
        raise ValueError("Normalize each beam's element weights to magnitude <= 1.")
    solutions = [acoustic.solve(a.spectrum(w)) for w in weights]
    basis = RadiationBasis(solutions, experiment.lower, experiment.upper)
    kernels_hat = g.forward(basis.kernels)
    dynamics = SurfaceDynamics(
        g, experiment.lower, experiment.upper, experiment.interface, experiment.timing.step_s
    )
    steps = round(experiment.timing.end_s / experiment.timing.step_s)
    saved_steps = sorted(set(range(0, steps + 1, experiment.timing.save_every)) | {steps})
    times = np.array(saved_steps) * experiment.timing.step_s
    shape = (len(times), experiment.grid.ny, experiment.grid.nx)
    height = np.empty(shape, dtype=np.float32)
    velocity = np.empty(shape, dtype=np.float32)
    amplitudes = np.empty((len(times), len(experiment.beams)))
    metric_names = (
        "min_height_m",
        "max_height_m",
        "rms_height_m",
        "max_speed_m_s",
        "volume_change_m3",
        "max_slope",
        "potential_energy_j",
        "kinetic_energy_j",
        "dissipation_w",
        "forcing_work_rate_w",
        "edge_rms_height_m",
    )
    metrics = {key: np.empty(len(times)) for key in metric_names}
    edge_mask = (np.abs(g.xx) > 0.45 * experiment.grid.lx_m) | (
        np.abs(g.yy) > 0.45 * experiment.grid.ly_m
    )
    h = np.zeros(g.k.shape, dtype=complex)
    v = np.zeros_like(h)
    frame = 0
    for step in range(steps + 1):
        t = step * experiment.timing.step_s
        if step == saved_steps[frame]:
            amps = beam_amplitudes(experiment, t)
            p = np.einsum("i,j,ijyx->yx", amps, amps, kernels_hat)
            v_out = dynamics.endpoint_velocity(h, v, p)
            hp, vp = g.inverse(h).real, g.inverse(v_out).real
            height[frame], velocity[frame], amplitudes[frame] = hp, vp, amps
            grad_x, grad_y = g.gradient(h)
            potential, kinetic, dissipation = dynamics.energy(h, v_out)
            values = (
                hp.min(),
                hp.max(),
                np.sqrt(np.mean(hp * hp)),
                np.max(np.abs(vp)),
                np.mean(hp) * g.area,
                np.hypot(grad_x, grad_y).max(),
                potential,
                kinetic,
                dissipation,
                g.area * np.real(np.sum(p * v_out.conj())),
                np.sqrt(np.mean(hp[edge_mask] ** 2)),
            )
            for key, value in zip(metric_names, values, strict=True):
                metrics[key][frame] = value
            frame += 1
            if step == steps:
                break
        amps = beam_amplitudes(experiment, t + experiment.timing.step_s / 2)
        p = np.einsum("i,j,ijyx->yx", amps, amps, kernels_hat)
        h, v = dynamics.step(h, v, p)
    if not np.isfinite(height).all() or not np.isfinite(velocity).all():
        raise FloatingPointError("Non-finite surface state.")
    power_error = max(
        abs(s.powers_w["lower_net"] - s.powers_w["transmitted"])
        / max(s.powers_w["incident"], 1e-30)
        for s in solutions
    )
    continuity_p = max(
        float(np.max(np.abs(s.below.pressure - s.above.pressure)))
        / max(float(np.max(np.abs(s.below.pressure))), 1e-30)
        for s in solutions
    )
    continuity_v = max(
        float(np.max(np.abs(s.below.velocity[2] - s.above.velocity[2])))
        / max(float(np.max(np.abs(s.below.velocity[2]))), 1e-30)
        for s in solutions
    )
    max_h = float(np.abs(height).max())
    wavelength = experiment.lower.sound_speed_m_s / experiment.array.frequency_hz
    diagnostics = {
        "max_abs_height_m": max_h,
        "max_slope": float(metrics["max_slope"].max()),
        "incident_k_times_max_height": 2 * np.pi / wavelength * max_h,
        "max_abs_volume_change_m3_float64_state": float(np.abs(metrics["volume_change_m3"]).max()),
        "max_saved_float32_volume_change_m3": float(
            np.abs(height.astype(float).mean(axis=(1, 2)) * g.area).max()
        ),
        "relative_acoustic_power_balance_error": power_error,
        "relative_pressure_continuity_error": continuity_p,
        "relative_normal_velocity_continuity_error": continuity_v,
        "acoustic_wavelength_lower_m": wavelength,
        "samples_per_lower_wavelength": wavelength
        / max(experiment.grid.lx_m / experiment.grid.nx, experiment.grid.ly_m / experiment.grid.ny),
        "max_edge_rms_over_global_peak_height": float(
            metrics["edge_rms_height_m"].max() / max(max_h, 1e-30)
        ),
    }
    notices = [
        "Acoustics is solved on a flat interface; moving-interface scattering is not implemented.",
        "The transverse domain is periodic. Late capillary waves can wrap around the domain.",
        "Hydrodynamics is a linear small-slope deep-fluid limit, not full Navier-Stokes.",
        "No curing, heating, acoustic streaming, contact line, finite frame or transducer mechanics.",
    ]
    if diagnostics["max_slope"] > 0.1 or diagnostics["incident_k_times_max_height"] > 0.1:
        notices.append(
            "GEOMETRY DIAGNOSTIC: slope or k*h exceeds 0.1. This heuristic flags "
            "possible failure of the flat/small-slope reference; it is not an error bound."
        )
    if diagnostics["samples_per_lower_wavelength"] < 4:
        notices.append(
            "RESOLUTION DIAGNOSTIC: fewer than four samples per acoustic wavelength; "
            "quadratic radiation-pressure spectra may alias."
        )
    if experiment.array.retain_evanescent:
        notices.append(
            "Evanescent modes enabled: propagating-band Nyquist arguments no longer "
            "suffice; independently refine spatial resolution and source gap."
        )
    if experiment.interface.model == "weakly_damped":
        notices.append(
            "Weak-viscosity damping is a liquid/light-gas approximation; upper-gas "
            "viscous drag and the full unsteady viscous boundary layer are omitted."
        )
    report = {
        "experiment": experiment.as_dict(),
        "diagnostics": diagnostics,
        "beam_acoustic_powers_w": {
            b.name: s.powers_w for b, s in zip(experiment.beams, solutions, strict=True)
        },
        "limitations": notices,
        "simulation_wall_s": time.perf_counter() - started,
        "weights_source": "foci" if element_weights is None else "explicit_complex_array",
        "status": "numerical research reference; not experimental validation",
    }
    return Result(
        experiment,
        g,
        times,
        height,
        velocity,
        amplitudes,
        basis.kernels,
        weights,
        metrics,
        report,
        acoustic,
        a,
        dynamics,
    )


def save_result(result: Result, directory: str | Path, source_config: Path | None = None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.npz").exists():
        raise FileExistsError(f"Existing experiment output: {directory}; choose a new directory.")
    np.savez_compressed(
        directory / "result.npz",
        time_s=result.time_s,
        x_m=result.grid.x,
        y_m=result.grid.y,
        height_m=result.height_m,
        velocity_m_s=result.velocity_m_s,
        amplitudes=result.amplitudes,
        radiation_kernels_pa=result.radiation_kernels_pa,
        element_weights=result.element_weights,
        **{f"metric_{k}": v for k, v in result.metrics.items()},
    )
    report = dict(result.report)
    report["software"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pyvista", "vtk"]},
    }
    report["source_sha256"] = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(Path(__file__).parent.glob("*.py"))
    }
    report["result_sha256"] = hashlib.sha256((directory / "result.npz").read_bytes()).hexdigest()
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if source_config is not None:
        (directory / "experiment.toml").write_bytes(source_config.read_bytes())
    return directory


def load_result(directory: str | Path) -> Result:
    from .config import experiment_from_dict

    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    experiment = experiment_from_dict(report["experiment"])
    checksum = hashlib.sha256((directory / "result.npz").read_bytes()).hexdigest()
    if checksum != report["result_sha256"]:
        raise ValueError("Result checksum mismatch; the stored fields have changed.")
    with np.load(directory / "result.npz", allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    g = SpectralGrid(experiment.grid)
    a = PressureArray(experiment.array, g, experiment.lower.sound_speed_m_s)
    acoustic = PlanarAcoustics(g, experiment.array, experiment.lower, experiment.upper)
    dynamic = SurfaceDynamics(
        g, experiment.lower, experiment.upper, experiment.interface, experiment.timing.step_s
    )
    return Result(
        experiment,
        g,
        data["time_s"],
        data["height_m"],
        data["velocity_m_s"],
        data["amplitudes"],
        data["radiation_kernels_pa"],
        data["element_weights"],
        {k.removeprefix("metric_"): v for k, v in data.items() if k.startswith("metric_")},
        report,
        acoustic,
        a,
        dynamic,
    )
