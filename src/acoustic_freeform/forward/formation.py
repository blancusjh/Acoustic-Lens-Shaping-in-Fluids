"""Formation time-stepping kernels; not themselves physical trajectories.

Flat-domain potential-flow inertia (NOT a viscous three-fluid solver), the
midpoint inertial step, the backward viscous step with old-geometry Stokes
mobility, and the drive envelope.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import j0, jn_zeros


def implicit_step(space, q, force, mobility, dt):
    """Backward mechanical gradient, old-geometry Stokes mobility; first order."""
    old = q.ravel()
    new = old.copy()
    for _ in range(20):
        grad, stiffness, _ = space.mechanics(new.reshape(q.shape))
        residual = new - old - dt * mobility @ (force - grad)
        change = np.linalg.solve(np.eye(len(old)) + dt * mobility @ stiffness, residual)
        new -= change
        if np.max(abs(change)) < 1e-14:
            return new.reshape(q.shape)
    raise RuntimeError("Implicit viscous step did not converge")


def layer_mass(k, depth, density):
    """Potential-flow mass per radial norm for two layer boundary velocities."""
    x = k * depth
    cross = 2 * np.exp(-x) / (-np.expm1(-2 * x))
    return density / k * np.array([[1 / np.tanh(x), -cross], [-cross, 1 / np.tanh(x)]])


def inertia(space, radial_modes=160):
    cfg = space.config
    k = jn_zeros(1, radial_modes) / cfg.radius_m
    # Independent, sufficiently resolved radial quadrature for modal projection.
    x, w = leggauss(max(600, 4 * radial_modes))
    r = cfg.radius_m * (x + 1) / 2
    weight = np.pi * cfg.radius_m * r * w
    phi = j0(r[:, None] * k)
    norm = np.pi * cfg.radius_m**2 * j0(k * cfg.radius_m) ** 2
    projection = (phi.T * weight) @ space.basis(r) / norm[:, None]
    result = np.zeros((2 * space.count, 2 * space.count))
    for mode, wave_number in enumerate(k):
        blocks = [
            layer_mass(wave_number, h, rho)
            for h, rho in zip(np.diff(cfg.levels_m), cfg.density_kg_m3)
        ]
        mass = np.array(
            [
                [blocks[0][1, 1] + blocks[1][0, 0], blocks[1][0, 1]],
                [blocks[1][1, 0], blocks[1][1, 1] + blocks[2][0, 0]],
            ]
        )
        result += np.kron(mass, norm[mode] * np.outer(projection[mode], projection[mode]))
    return result


def envelope(t, ramp):
    return float(np.sin(np.pi / 2 * np.clip(t / ramp, 0, 1)) ** 2)


def midpoint_step(q, v, dt, force, mechanics):
    """Mass-normalized coordinates; acoustic load is evaluated by the caller."""
    new = q + dt * v
    for _ in range(20):
        grad, hess, _ = mechanics((q + new) / 2)
        residual = new - q - dt * v - dt**2 / 2 * (force - grad)
        change = np.linalg.solve(np.eye(len(q)) + dt**2 / 4 * hess, residual)
        new -= change
        if np.linalg.norm(change) < 1e-14 * max(1, np.linalg.norm(new)):
            return new, 2 * (new - q) / dt - v
    raise RuntimeError("Mechanical midpoint solve failed")
