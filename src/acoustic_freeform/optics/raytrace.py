"""Meridional geometric rays through BOTH graphs, with a fixed laboratory detector."""

import numpy as np


def refract(direction, normal, n_in, n_out):
    cosine = np.sum(direction * normal, axis=-1)
    eta = n_in / n_out
    radicand = 1 - eta**2 * (1 - cosine**2)
    good = (cosine > 0) & (radicand >= 0)
    outgoing = (
        eta * direction + (np.sqrt(np.maximum(radicand, 0)) - eta * cosine)[..., None] * normal
    )
    return outgoing, good


def fixed_detector_spot_jacobian(
    space, q, indices, object_z_m, detector_z_m, launch_radius_m, launch_height_m
):
    """Differentiate both intersections and Snell maps for a fixed illumination cone.

    Returns meridional signed spot and its derivative with respect to the two
    concatenated surface coefficient vectors. Requires fully transmitted,
    non-grazing base rays. This is a local optical derivative, not a formed state.
    """
    traced = trace_pair(
        space,
        q,
        indices,
        object_z_m,
        detector_z_m,
        launch_radius_m=launch_radius_m,
        launch_height_m=launch_height_m,
    )
    if not traced["all_rays_transmitted"]:
        raise ValueError("Spot derivative requires all base rays to transmit")
    r = np.asarray(launch_radius_m)
    direction = np.column_stack([r, np.asarray(launch_height_m) - object_z_m])
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    origin = np.column_stack([np.zeros(len(r)), np.full(len(r), object_z_m)])
    dx = np.zeros((len(r), 2, 2 * space.count))
    dd = np.zeros_like(dx)
    for face in range(2):
        point = traced["intersections_m"][face]
        distance = (point[:, 1] - origin[:, 1]) / direction[:, 1]
        radius, sign = abs(point[:, 0]), np.sign(point[:, 0])
        slope = sign * space.evaluate(q[face], radius, 1)
        second = space.evaluate(q[face], radius, 2)
        explicit_height = np.zeros((len(r), 2 * space.count))
        explicit_slope = np.zeros_like(explicit_height)
        columns = slice(face * space.count, (face + 1) * space.count)
        explicit_height[:, columns] = space.basis(radius)
        explicit_slope[:, columns] = sign[:, None] * space.basis(radius, 1)
        fixed_time = dx + distance[:, None, None] * dd
        crossing = direction[:, 1] - slope * direction[:, 0]
        if np.any(abs(crossing) < 1e-8):
            raise ValueError("Grazing intersection has an ill-conditioned derivative")
        dt = (slope[:, None] * fixed_time[:, 0] + explicit_height - fixed_time[:, 1]) / crossing[
            :, None
        ]
        dx = fixed_time + direction[:, :, None] * dt[:, None, :]
        ds = second[:, None] * dx[:, 0] + explicit_slope
        stretch = np.sqrt(1 + slope**2)
        normal = np.column_stack([-slope, np.ones(len(r))]) / stretch[:, None]
        dn = (
            np.column_stack([-np.ones(len(r)), -slope])[:, :, None]
            * ds[:, None, :]
            / stretch[:, None, None] ** 3
        )
        eta = indices[face] / indices[face + 1]
        cosine = np.sum(direction * normal, axis=1)
        root = np.sqrt(1 - eta**2 * (1 - cosine**2))
        if np.any(root < 1e-8):
            raise ValueError("Critical refraction has an ill-conditioned derivative")
        dc = np.sum(dd * normal[:, :, None] + direction[:, :, None] * dn, axis=1)
        dd = (
            eta * dd
            + (eta**2 * cosine / root - eta)[:, None, None] * normal[:, :, None] * dc[:, None, :]
            + (root - eta * cosine)[:, None, None] * dn
        )
        direction, _ = refract(direction, normal, indices[face], indices[face + 1])
        origin = point
    distance = (detector_z_m - origin[:, 1]) / direction[:, 1]
    dt = (-dx[:, 1] - distance[:, None] * dd[:, 1]) / direction[:, 1, None]
    derivative = dx[:, 0] + distance[:, None] * dd[:, 0] + direction[:, 0, None] * dt
    return origin[:, 0] + distance * direction[:, 0], derivative


def trace_pair(
    space,
    q,
    indices,
    object_z_m,
    detector_z_m,
    rays=301,
    launch_radius_m=None,
    launch_height_m=None,
):
    return trace_graph_pair(
        space.config,
        lambda face, r, derivative=0: space.evaluate(q[face], r, derivative),
        indices,
        object_z_m,
        detector_z_m,
        rays,
        launch_radius_m,
        launch_height_m,
    )


def trace_graph_pair(
    cfg,
    evaluate,
    indices,
    object_z_m,
    detector_z_m,
    rays=301,
    launch_radius_m=None,
    launch_height_m=None,
):
    """Trace declared graphs; optional launch points are fixed laboratory points.

    Supplying target-pupil launch points defines a fixed source cone for a later
    trajectory. Do not recompute that cone from each moving physical state.
    The default retains the historical flat-reference launch convention.
    """
    if not object_z_m < cfg.levels_m[0] or not detector_z_m > cfg.levels_m[-1]:
        raise ValueError("Object and detector must bracket the cell")
    # Fixed equal-area launch directions, NOT retargeted at each moving surface.
    radius = (
        cfg.clear_radius_m * np.sqrt((np.arange(rays) + 0.5) / rays)
        if launch_radius_m is None
        else np.asarray(launch_radius_m, float)
    )
    if (
        radius.ndim != 1
        or not len(radius)
        or np.any(~np.isfinite(radius))
        or np.any((radius < 0) | (radius > cfg.clear_radius_m))
    ):
        raise ValueError("Launch radii must lie in the declared clear pupil")
    rays = len(radius)
    launch_z = np.broadcast_to(
        cfg.levels_m[1] if launch_height_m is None else launch_height_m, radius.shape
    )
    direction = np.column_stack([radius, launch_z - object_z_m])
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    origin = np.column_stack([np.zeros(rays), np.full(rays, object_z_m)])
    good = np.ones(rays, dtype=bool)
    intersections = []
    for face in range(2):
        level = cfg.levels_m[face + 1]
        nodes = np.linspace(0, cfg.radius_m, 1001)
        height = evaluate(face, nodes)
        low = (level + height.min() - 1e-9 - origin[:, 1]) / direction[:, 1]
        high = (level + height.max() + 1e-9 - origin[:, 1]) / direction[:, 1]
        for _ in range(45):
            mid = (low + high) / 2
            point = origin + mid[:, None] * direction
            rr = np.clip(abs(point[:, 0]), 0, cfg.radius_m)
            residual = point[:, 1] - level - evaluate(face, rr)
            high = np.where(residual > 0, mid, high)
            low = np.where(residual > 0, low, mid)
        distance = (low + high) / 2
        origin = origin + distance[:, None] * direction
        radial = abs(origin[:, 0])
        # Picometre intersection tolerance, not an adjustable physical aperture.
        good &= (distance > 0) & (radial <= cfg.clear_radius_m + 1e-12)
        good &= (
            abs(origin[:, 1] - level - evaluate(face, np.clip(radial, 0, cfg.radius_m))) <= 1e-11
        )
        slope = evaluate(face, np.clip(radial, 0, cfg.radius_m), 1) * np.sign(origin[:, 0])
        normal = np.column_stack([-slope, np.ones(rays)])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        direction, transmitted = refract(direction, normal, indices[face], indices[face + 1])
        good &= transmitted & (direction[:, 1] > 0)
        intersections.append(origin.copy())
    distance = (detector_z_m - origin[:, 1]) / direction[:, 1]
    good &= distance > 0
    signed_spot = origin[:, 0] + distance * direction[:, 0]
    angle = np.arange(rays) * np.pi * (3 - np.sqrt(5))
    spots = signed_spot[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])
    spots[~good] = np.nan
    return {
        "spots_m": spots,
        "transmitted": good,
        "intersections_m": np.array(intersections),
        "rms_radius_m": float(np.sqrt(np.mean(signed_spot[good] ** 2))) if good.any() else np.nan,
        "max_radius_m": float(np.max(abs(signed_spot[good]))) if good.any() else np.nan,
        "all_rays_transmitted": bool(good.all()),
        "sampled_1um_pass": bool(good.all() and np.max(abs(signed_spot)) <= 1e-6),
        "continuous_pupil_certified": False,
    }
