"""Independent analytic limits and fixed-drive convergence of the finite lens."""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import root
from scipy.special import jv

from .acoustics import CavityAcoustics
from .config import LensConfig
from .optics import trace_surface
from .surface import SurfaceSpace


def analytic_flat_cavity(cfg, drive, points, terms=100, normal_velocity=False):
    """Independent separated Bessel series for the same flat, closed cylinder.

    Rigid base, pressure-release top and the declared tapered wall velocities.
    Coordinates and pressure potential use the solver's nondimensionalization.
    Integration splits at each patch boundary; no FEM matrices enter this result.
    """
    depth = cfg.depth_m / cfg.radius_m
    k = (
        2 * np.pi * cfg.frequency_hz / cfg.sound_speed_m_s + 1j * cfg.attenuation_np_m
    ) * cfg.radius_m
    vertical = (np.arange(terms) + 0.5) * np.pi / depth
    nodes, weights = leggauss(80)
    coefficients = np.zeros(terms, complex)
    pitch = depth / cfg.array_rows
    half = 0.5 * pitch * cfg.element_fill
    for row, value in enumerate(drive):
        center = -depth + (row + 0.5) * pitch
        z = center + half * nodes
        patch = 0.5 * (1 + np.cos(np.pi * nodes))
        coefficients += (
            2 / depth * value * (np.cos(vertical[:, None] * (z + depth)) @ (weights * half * patch))
        )
    radial_wave = np.sqrt(k * k - vertical * vertical)
    amplitudes = -1j * coefficients / (radial_wave * jv(1, radial_wave))
    radial, z = np.asarray(points)
    radial_function = jv(0, radial_wave[:, None] * radial)
    if normal_velocity:
        axial = -vertical[:, None] * np.sin(vertical[:, None] * (z + depth))
        return -1j * np.sum(amplitudes[:, None] * radial_function * axial, axis=0)
    axial = np.cos(vertical[:, None] * (z + depth))
    return np.sum(amplitudes[:, None] * radial_function * axial, axis=0)


def flat_cavity_check():
    cfg = LensConfig(
        frequency_hz=420000,
        array_rows=3,
        mesh_radial=28,
        mesh_vertical=48,
        acoustic_order=3,
        surface_modes=12,
    )
    space = SurfaceSpace(cfg)
    field = CavityAcoustics(cfg, space).solve_basis(np.zeros(space.count))
    drive = np.array([0.009 + 0.004j, -0.003 + 0.007j, 0.006 - 0.002j])
    nodes = field.basis.nodal_dofs[0]
    points = field.basis.doflocs[:, nodes]
    exact = analytic_flat_cavity(cfg, drive, points)
    computed = field.solutions[nodes] @ drive
    velocity = field.normal_velocity_basis @ drive
    exact_velocity = analytic_flat_cavity(
        cfg,
        drive,
        np.array([field.radial_samples, np.zeros_like(field.radial_samples)]),
        normal_velocity=True,
    )
    stress, exact_stress = np.abs(velocity) ** 2, np.abs(exact_velocity) ** 2
    w = field.quadrature_weights
    return {
        "pressure_relative_l2": float(np.linalg.norm(computed - exact) / np.linalg.norm(exact)),
        "radiation_stress_relative_l2": float(
            np.sqrt(np.sum(w * (stress - exact_stress) ** 2) / np.sum(w * exact_stress**2))
        ),
        "helmholtz_algebraic_residual": field.residual,
        "reference": "Independent Fourier-Bessel series, flat cylinder, identical wall drive.",
    }


def fixed_drive_equilibrium(space, drive, initial):
    """Solve force balance while re-solving acoustics on each trial geometry."""
    initial = np.asarray(initial).copy()
    q = space.tangent
    _, hessian = space.derivatives(initial)
    preconditioner = np.linalg.inv(q.T @ hessian @ q)
    acoustic = CavityAcoustics(space.config, space)

    def residual(x):
        coefficients = initial + q @ x
        gradient, _ = space.derivatives(coefficients)
        field = acoustic.solve_basis(coefficients)
        return preconditioner @ (q.T @ (gradient - field.force(drive)))

    solution = root(
        residual,
        np.zeros(q.shape[1]),
        method="broyden1",
        options={
            "fatol": 1e-10,
            "maxiter": 60,
            "line_search": "armijo",
            "jac_options": {"alpha": -1},
        },
    )
    error = float(np.linalg.norm(residual(solution.x)))
    if not solution.success:
        raise RuntimeError(
            f"Coupled stationary solve did not converge: {solution.message}; {error}"
        )
    return initial + q @ solution.x, error


def spatial_convergence(result_directory):
    result = Path(result_directory)
    data = np.load(result / "trajectory.npz")
    original = LensConfig(**json.loads((result / "configuration.json").read_text()))
    records = []
    for order, nr, nz, modes in [
        (3, 64, 96, 32),
        (4, 48, 64, 32),
        (4, 64, 96, 32),
        (4, 80, 128, 32),
        (4, 80, 128, 48),
    ]:
        cfg = replace(
            original, acoustic_order=order, mesh_radial=nr, mesh_vertical=nz, surface_modes=modes
        )
        space = SurfaceSpace(cfg)
        c = np.pad(data["coefficients"][-1], (0, modes - original.surface_modes))
        begin = time.perf_counter()
        equilibrium, residual = fixed_drive_equilibrium(space, data["drive_m_s"][-1], c)
        r = cfg.clear_radius_m / cfg.radius_m * np.sqrt((np.arange(401) + 0.5) / 401)
        optics = trace_surface(space, equilibrium)
        record = {
            "mesh": [nr, nz],
            "order": order,
            "modes": modes,
            "seconds": time.perf_counter() - begin,
            "converged": True,
            "residual": residual,
            "surface_change_rms_m": float(np.sqrt(np.mean(space.evaluate(equilibrium - c, r) ** 2)))
            * cfg.radius_m,
            "rms_spot_at_target_m": optics["rms_spot_at_target_m"],
            "rms_spot_at_best_focus_m": optics["rms_spot_at_best_focus_m"],
            "coefficients": equilibrium.tolist(),
        }
        records.append(record)
        print({k: v for k, v in record.items() if k != "coefficients"}, flush=True)
        (result / "spatial-convergence.json").write_text(json.dumps(records, indent=2) + "\n")
    return records
