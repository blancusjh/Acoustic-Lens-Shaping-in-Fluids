"""Monochromatic meridional ray and optical-path checks of computed lens surfaces."""

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .surface import SurfaceSpace


def pupil_intercepts(space, coefficients, radial_m, plane_m):
    cfg = space.config
    h = cfg.radius_m * space.evaluate(coefficients, radial_m / cfg.radius_m)
    slope = space.evaluate(coefficients, radial_m / cfg.radius_m, 1)
    nr, nz = -slope / np.sqrt(1 + slope * slope), 1 / np.sqrt(1 + slope * slope)
    n = cfg.refractive_index
    root = 1 - n * n * (1 - nz * nz)
    if np.min(root) <= 0:
        raise ValueError("Total internal reflection in the clear aperture")
    a = np.sqrt(root) - n * nz
    return radial_m + (plane_m - h) * a * nr / (n + a * nz)


def trace_surface(space: SurfaceSpace, coefficients, count=401, target_plane_m=None):
    cfg = space.config
    # Uniform sampling in pupil area; skip the axial ray in longitudinal checks.
    r = cfg.clear_radius_m * np.sqrt((np.arange(count) + 0.5) / count)
    h = cfg.radius_m * space.evaluate(coefficients, r / cfg.radius_m)
    slope = space.evaluate(coefficients, r / cfg.radius_m, 1)
    n = cfg.refractive_index
    normal_r = -slope / np.sqrt(1 + slope * slope)
    normal_z = 1 / np.sqrt(1 + slope * slope)
    cos_i = normal_z
    root = 1 - n * n * (1 - cos_i * cos_i)
    if np.any(root <= 0):
        raise ValueError("Total internal reflection inside the requested clear optical aperture")
    correction = np.sqrt(root) - n * cos_i
    direction_r = correction * normal_r
    direction_z = n + correction * normal_z
    height0 = cfg.radius_m * space.evaluate(coefficients, np.array([0]))[0]
    goal, _ = space.target()
    target_plane = (
        cfg.radius_m * space.evaluate(goal, np.array([0]))[0] + cfg.focal_distance_m
        if target_plane_m is None
        else target_plane_m
    )

    def spots(z):
        return r + (z - h) * direction_r / direction_z

    best = minimize_scalar(
        lambda z: np.mean(spots(z) ** 2),
        bounds=(height0 + 0.5 * cfg.focal_distance_m, height0 + 1.5 * cfg.focal_distance_m),
        method="bounded",
        options={"xatol": 1e-13},
    )
    optical_path = n * h + np.sqrt((target_plane - h) ** 2 + r * r)
    opd = optical_path - optical_path.mean()
    focus = h - r * direction_z / direction_r
    return {
        "pupil_r_m": r,
        "surface_z_m": h,
        "direction_r": direction_r,
        "direction_z": direction_z,
        "target_plane_m": float(target_plane),
        "best_focus_plane_m": float(best.x),
        "best_focus_from_vertex_m": float(best.x - height0),
        "rms_spot_at_target_m": float(np.sqrt(np.mean(spots(target_plane) ** 2))),
        "rms_spot_at_best_focus_m": float(np.sqrt(best.fun)),
        "opd_rms_m": float(np.sqrt(np.mean(opd * opd))),
        "longitudinal_focus_range_m": float(np.ptp(focus)),
        "target_spot_r_m": spots(target_plane),
        "opd_m": opd,
    }


def best_fit_sphere(space: SurfaceSpace, coefficients):
    cfg = space.config
    r = cfg.clear_radius_m * np.sqrt((np.arange(501) + 0.5) / 501)
    h = cfg.radius_m * space.evaluate(coefficients, r / cfg.radius_m)
    initial_radius = cfg.focal_distance_m * (cfg.refractive_index - 1)

    def height(p):
        radius, vertex = p
        return vertex - r * r / (radius + np.sqrt(radius * radius - r * r))

    fit = least_squares(
        lambda p: (height(p) - h) / cfg.radius_m,
        [initial_radius, float(h.max())],
        bounds=([cfg.clear_radius_m * 1.001, -0.01], [0.2, 0.01]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    difference = h - height(fit.x)
    return {
        "radius_m": float(fit.x[0]),
        "vertex_m": float(fit.x[1]),
        "departure_rms_m": float(np.sqrt(np.mean(difference * difference))),
        "departure_peak_to_valley_m": float(np.ptp(difference)),
    }


def best_fit_conic(space: SurfaceSpace, coefficients):
    """Fit an optical conic to the computed clear aperture; do not prescribe it."""
    cfg = space.config
    r = cfg.clear_radius_m * np.sqrt((np.arange(701) + 0.5) / 701)
    h = cfg.radius_m * space.evaluate(coefficients, r / cfg.radius_m)

    def height(p):
        radius, conic, vertex = p
        return vertex - r * r / (radius + np.sqrt(radius * radius - (1 + conic) * r * r))

    fit = least_squares(
        lambda p: (height(p) - h) / cfg.radius_m,
        [cfg.focal_distance_m * (cfg.refractive_index - 1), cfg.conic_constant, float(h.max())],
        x_scale=[0.01, 1, 0.001],
        bounds=([cfg.clear_radius_m * 1.5, -10, -0.01], [0.1, 1, 0.01]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    return {
        "vertex_radius_m": float(fit.x[0]),
        "conic_constant": float(fit.x[1]),
        "vertex_m": float(fit.x[2]),
        "fit_rms_m": float(np.sqrt(np.mean((height(fit.x) - h) ** 2))),
    }


def spherical_optical_reference(space: SurfaceSpace, coefficients):
    """Optical performance of the best-fit sphere over the same clear pupil.

    This is an optical comparison surface, not a second fluid equilibrium.
    """
    cfg = space.config
    fit = best_fit_sphere(space, coefficients)
    r = cfg.clear_radius_m * np.sqrt((np.arange(701) + 0.5) / 701)
    radius, vertex = fit["radius_m"], fit["vertex_m"]
    h = vertex - r * r / (radius + np.sqrt(radius * radius - r * r))
    slope = -r / np.sqrt(radius * radius - r * r)
    nr, nz = -slope / np.sqrt(1 + slope * slope), 1 / np.sqrt(1 + slope * slope)
    a = np.sqrt(1 - cfg.refractive_index**2 * (1 - nz * nz)) - cfg.refractive_index * nz
    ray_slope = a * nr / (cfg.refractive_index + a * nz)
    intercept = r - h * ray_slope
    best_plane = -np.dot(intercept, ray_slope) / np.dot(ray_slope, ray_slope)
    rms = np.sqrt(np.mean((intercept + best_plane * ray_slope) ** 2))
    return {
        "rms_spot_at_best_focus_m": float(rms),
        "best_focus_from_vertex_m": float(best_plane - vertex),
        "scope": "Best-fit sphere, same clear aperture; freely refocused optical comparison.",
    }
