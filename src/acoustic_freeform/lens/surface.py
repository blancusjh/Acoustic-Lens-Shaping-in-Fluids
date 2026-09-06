"""Nonlinear capillarity of a pinned, finite, axisymmetric liquid volume.

Dimensionless r and h use the chamber radius. Shifted Legendre functions in r^2
enforce axis symmetry and a fixed rim. No small-slope curvature approximation.
Energy is normalized by 2*pi*sigma*radius^2; volume by 2*pi*radius^3.
"""

import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss
from scipy.linalg import null_space

from .config import LensConfig


class SurfaceSpace:
    def __init__(self, config: LensConfig):
        self.config = config
        self.count = config.surface_modes
        points, weights = leggauss(max(128, 6 * self.count))
        self.r = (points + 1) / 2
        self.w = weights / 2 * self.r
        self.b, self.dr = self.basis(self.r), self.basis(self.r, 1)
        self.mass = self.b.T @ (self.w[:, None] * self.b)
        self.volume_vector = self.b.T @ self.w
        self.tangent = null_space(self.volume_vector[None, :])

    def basis(self, r, derivative=0):
        r = np.asarray(r)
        x = 2 * r * r - 1
        columns = []
        for j in range(1, self.count + 1):
            p = Legendre.basis(j)
            if derivative == 0:
                columns.append(p(x) - 1)
            elif derivative == 1:
                columns.append(4 * r * p.deriv()(x))
            else:
                columns.append(4 * p.deriv()(x) + 16 * r * r * p.deriv(2)(x))
        return np.stack(columns, axis=-1)

    def evaluate(self, coefficients, r, derivative=0):
        return self.basis(r, derivative) @ coefficients

    def energy(self, c):
        h, slope = self.b @ c, self.dr @ c
        return float(np.sum(self.w * (np.sqrt(1 + slope * slope) + 0.5 * self.config.bond * h * h)))

    def derivatives(self, c):
        h, slope = self.b @ c, self.dr @ c
        gradient = self.dr.T @ (self.w * slope / np.sqrt(1 + slope * slope))
        gradient += self.config.bond * self.b.T @ (self.w * h)
        hessian = self.dr.T @ ((self.w / (1 + slope * slope) ** 1.5)[:, None] * self.dr)
        hessian += self.config.bond * self.mass
        return gradient, hessian

    def fit(self, heights):
        return np.linalg.solve(self.mass, self.b.T @ (self.w * heights))

    def equilibrium(self, force, volume, initial=None, tolerance=1e-11):
        c = np.zeros(self.count) if initial is None else np.array(initial, dtype=float).copy()
        if initial is None:
            c[0] = volume / self.volume_vector[0]
        v = self.volume_vector
        for iteration in range(60):
            gradient, hessian = self.derivatives(c)
            kkt = np.block([[hessian, v[:, None]], [v[None, :], np.zeros((1, 1))]])
            delta = np.linalg.solve(kkt, np.r_[force - gradient, volume - v @ c])[: self.count]
            if np.linalg.norm(delta) < tolerance:
                return c
            step = 1.0
            before = self.energy(c) - force @ c
            while step > 1e-6:
                candidate = c + step * delta
                if self.energy(candidate) - force @ candidate <= before + 1e-13:
                    c = candidate
                    break
                step /= 2
            else:
                raise RuntimeError("Capillary Newton line search failed")
        raise RuntimeError("Capillary equilibrium did not converge")

    def target(self):
        """Cartesian vertex branch, translated to meet the chamber rim."""
        if hasattr(self, "_target_cache"):
            c, volume = self._target_cache
            return c.copy(), volume
        cfg = self.config
        radius = cfg.radius_m
        heights = cfg.diopter.sag(np.r_[self.r * radius, radius])
        coefficients = self.fit((heights[:-1] - heights[-1]) / radius)
        volume = self.volume_vector @ coefficients
        self._target_cache = coefficients.copy(), float(volume)
        return coefficients, float(volume)

    @property
    def optical_vertex_m(self):
        target, _ = self.target()
        return float(self.config.radius_m * self.evaluate(target, np.array([0]))[0])
