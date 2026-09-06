"""Fit physical complex wall velocities to nonlinear capillary force requirements."""

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .acoustics import CavityAcoustics, CavityField
from .optics import pupil_intercepts
from .surface import SurfaceSpace


@dataclass
class ArrayDesign:
    drive_m_s: np.ndarray
    target_coefficients: np.ndarray
    predicted_shape_error_m: float
    force_residual: float
    objective: float
    attempts: list[dict]


def fit_array(
    space: SurfaceSpace,
    field: CavityField,
    target,
    starts=5,
    initial_drive=None,
    regularization=1e-5,
    pressure_penalty=None,
) -> ArrayDesign:
    cfg, q = space.config, space.tangent
    if pressure_penalty is None:
        pressure_penalty = cfg.pressure_penalty
    gradient, hessian = space.derivatives(target)
    # Project the force mismatch through capillary stiffness. This gives a
    # local shape correction, with pressure/volume gauge removed exactly.
    compliance = q @ np.linalg.solve(q.T @ hessian @ q, q.T)
    pupil = cfg.clear_radius_m / cfg.radius_m * np.sqrt((np.arange(100) + 0.5) / 100)
    shape_map = space.basis(pupil) @ compliance
    shape_map *= cfg.radius_m / 1e-6 / np.sqrt(len(pupil))  # micrometres, pupil RMS
    plane = cfg.radius_m * space.evaluate(target, np.array([0]))[0] + cfg.focal_distance_m
    ray_jacobian = []
    for mode in range(space.count):
        change = np.eye(space.count)[mode] * 1e-7
        plus = pupil_intercepts(space, target + change, pupil * cfg.radius_m, plane)
        minus = pupil_intercepts(space, target - change, pupil * cfg.radius_m, plane)
        ray_jacobian.append((plus - minus) / 2e-7)
    ray_map = np.stack(ray_jacobian, axis=1) @ compliance / 1e-6 / np.sqrt(len(pupil))
    objective_map = np.vstack([0.2 * shape_map, ray_map])
    scale = 0.01
    bound = cfg.max_wall_speed_m_s / (np.sqrt(2) * scale)
    rows = cfg.array_rows
    matrices = field.force_kernels
    nodes = field.basis.nodal_dofs[0]
    radial = field.basis.doflocs[0, nodes]
    node_weights = np.sqrt(radial / max(radial.sum(), 1e-30))
    pressure_map = (
        cfg.density_kg_m3
        * 2
        * np.pi
        * cfg.frequency_hz
        * cfg.radius_m
        / 1e6
        * node_weights[:, None]
        * field.solutions[nodes]
    )
    # Volume-weighted pressure penalty, compressed without changing its norm.
    _, pressure_map = np.linalg.qr(pressure_map, mode="reduced")
    pressure_jacobian = (
        scale
        * pressure_penalty
        * np.block(
            [[pressure_map.real, -pressure_map.imag], [pressure_map.imag, pressure_map.real]]
        )
    )

    def unpack(x):
        return scale * (x[:rows] + 1j * x[rows:])

    def residual(x):
        drive = unpack(x)
        error = objective_map @ (field.force(drive) - gradient)
        return np.r_[error, regularization * x, pressure_jacobian @ x]

    def jacobian(x):
        drive = unpack(x)
        product = np.einsum("ijk,k->ij", matrices, drive)
        derivative = 2 * scale * np.concatenate([product.real, product.imag], axis=1)
        return np.vstack(
            [objective_map @ derivative, regularization * np.eye(2 * rows), pressure_jacobian]
        )

    rng = np.random.default_rng(638217)
    attempts = []
    best = None
    for i in range(starts):
        if i == 0 and initial_drive is not None:
            x = np.r_[initial_drive.real, initial_drive.imag] / scale
        else:
            x = rng.normal(0, 0.15, 2 * rows)
        x = np.clip(x, -bound * 0.95, bound * 0.95)
        begin = time.perf_counter()
        try:
            fit = least_squares(
                residual,
                x,
                jac=jacobian,
                bounds=(-bound, bound),
                max_nfev=1800,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-10,
            )
        except np.linalg.LinAlgError:
            fit = least_squares(
                residual,
                x,
                jac=jacobian,
                bounds=(-bound, bound),
                tr_solver="lsmr",
                max_nfev=1800,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-9,
            )
        error = residual(fit.x)
        row = {
            "start": i,
            "predicted_pupil_rms_um": float(
                np.linalg.norm(shape_map @ (field.force(unpack(fit.x)) - gradient))
            ),
            "predicted_ray_rms_um": float(np.linalg.norm(error[len(pupil) : 2 * len(pupil)])),
            "evaluations": fit.nfev,
            "seconds": time.perf_counter() - begin,
            "max_wall_speed_m_s": float(np.max(np.abs(unpack(fit.x)))),
        }
        attempts.append(row)
        if best is None or np.linalg.norm(error) < best[0]:
            best = (float(np.linalg.norm(error)), fit.x.copy())
        if row["predicted_ray_rms_um"] < 0.02:
            break
    objective, x = best
    drive = unpack(x)
    mismatch = q.T @ (field.force(drive) - gradient)
    return ArrayDesign(
        drive,
        np.asarray(target),
        float(np.linalg.norm(shape_map @ (field.force(drive) - gradient))) * 1e-6,
        float(np.linalg.norm(mismatch)),
        objective,
        attempts,
    )


def coupled_equilibrium(
    space, acoustic: CavityAcoustics, drive, volume, initial, max_iterations=100, relaxation=0.3
):
    """Actual steady forward iteration with the same fixed physical wall drive.

    This is equilibrium iteration, not a physical time trajectory.
    """
    c = np.asarray(initial).copy()
    history = []
    for i in range(max_iterations):
        field = acoustic.solve_basis(c)
        force = field.force(drive)
        next_c = space.equilibrium(force, volume, initial=c)
        error = float(np.max(np.abs(space.b @ (next_c - c))) * space.config.radius_m)
        history.append(error)
        if error < 1e-9:
            return next_c, history, field
        if np.max(np.abs(space.b @ next_c)) > 0.6:
            raise RuntimeError("Steady iteration left the single-valued lens geometry envelope")
        c += relaxation * (next_c - c)
    return c, history, field
