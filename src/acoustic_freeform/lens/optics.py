"""Monochromatic meridional ray and optical-path checks of computed lens surfaces."""

import numpy as np
from scipy.optimize import least_squares

from .surface import SurfaceSpace


def pupil_intercepts(space, coefficients, radial_m, plane_m):
    cfg = space.config
    h = cfg.radius_m * space.evaluate(coefficients, radial_m / cfg.radius_m)
    slope = space.evaluate(coefficients, radial_m / cfg.radius_m, 1)
    direction = cfg.diopter.refract(radial_m, h - space.optical_vertex_m, slope)
    return radial_m + (plane_m - h) * direction[:, 0] / direction[:, 1]


def trace_surface(space: SurfaceSpace, coefficients, count=401, target_plane_m=None):
    cfg = space.config
    # Uniform sampling in pupil area; skip the axial ray in longitudinal checks.
    r = cfg.clear_radius_m * np.sqrt((np.arange(count) + 0.5) / count)
    h = cfg.radius_m * space.evaluate(coefficients, r / cfg.radius_m)
    slope = space.evaluate(coefficients, r / cfg.radius_m, 1)
    vertex = space.optical_vertex_m
    direction = cfg.diopter.refract(r, h - vertex, slope)
    incident = cfg.diopter.incident_direction(r, h - vertex)
    direction_r, direction_z = direction.T
    height0 = cfg.radius_m * space.evaluate(coefficients, np.array([0]))[0]
    target_plane = vertex + cfg.focal_distance_m if target_plane_m is None else target_plane_m

    def spots(z):
        return r + (z - h) * direction_r / direction_z

    ray_slope = direction_r / direction_z
    best_plane = -np.dot(r - h * ray_slope, ray_slope) / np.dot(ray_slope, ray_slope)
    # The incident wavefront stays fixed in physical space for every state.
    # For an optional detector plane, replace only the outgoing optical path.
    optical_path = cfg.diopter.fermat(r, h - vertex)
    if target_plane_m is not None:
        optical_path += cfg.image_refractive_index * (
            np.hypot(r, target_plane - h) - np.hypot(r, vertex + cfg.focal_distance_m - h)
        )
    opd = optical_path - optical_path.mean()
    focus = h - r * direction_z / direction_r
    return {
        "pupil_r_m": r,
        "surface_z_m": h,
        "direction_r": direction_r,
        "direction_z": direction_z,
        "incident_direction_r": incident[:, 0],
        "incident_direction_z": incident[:, 1],
        "target_plane_m": float(target_plane),
        "best_focus_plane_m": float(best_plane),
        "best_focus_from_vertex_m": float(best_plane - height0),
        "rms_spot_at_target_m": float(np.sqrt(np.mean(spots(target_plane) ** 2))),
        "rms_spot_at_best_focus_m": float(np.sqrt(np.mean(spots(best_plane) ** 2))),
        "opd_rms_m": float(np.sqrt(np.mean(opd * opd))),
        "longitudinal_focus_range_m": float(np.ptp(focus)),
        "target_spot_r_m": spots(target_plane),
        "opd_m": opd,
    }


def best_fit_sphere(space: SurfaceSpace, coefficients):
    cfg = space.config
    r = cfg.clear_radius_m * np.sqrt((np.arange(501) + 0.5) / 501)
    h = cfg.radius_m * space.evaluate(coefficients, r / cfg.radius_m)
    initial_radius = -1 / cfg.diopter.vertex_curvature_m_inv

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
        [-1 / cfg.diopter.vertex_curvature_m_inv, cfg.conic_constant, float(h.max())],
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
    direction = cfg.diopter.refract(r, h - space.optical_vertex_m, slope)
    ray_slope = direction[:, 0] / direction[:, 1]
    intercept = r - h * ray_slope
    best_plane = -np.dot(intercept, ray_slope) / np.dot(ray_slope, ray_slope)
    rms = np.sqrt(np.mean((intercept + best_plane * ray_slope) ** 2))
    return {
        "rms_spot_at_best_focus_m": float(rms),
        "best_focus_from_vertex_m": float(best_plane - vertex),
        "scope": "Best-fit sphere, same clear aperture; freely refocused optical comparison.",
    }
