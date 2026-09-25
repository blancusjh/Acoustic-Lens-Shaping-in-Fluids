"""Nonlinear, volume-conserving graph mechanics; all physical fields use SI.

Quintic splines use x=r/R internally. Both sealed outer volumes are fixed.
The basis enforces pinned rims, regular axis and zero volume displacement.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline, PPoly
from scipy.linalg import block_diag, null_space
from scipy.optimize import least_squares

from ..lens.cartesian import CartesianDiopter


def cartesian_derivatives(diopter, r):
    z = diopter.sag(r)
    fr, fz = diopter.fermat_gradient(r, z)
    hp = -fr / fz
    frr, frz, fzz = (np.zeros_like(z) for _ in range(3))
    for n, focus, side in ((diopter.n_o, diopter.z_o_m, -1), (diopter.n_i, diopter.z_i_m, 1)):
        if np.isfinite(focus):
            d = z - focus
            length = np.hypot(r, d)
            sign = side * np.sign(focus) * n
            frr += sign * d**2 / length**3
            frz -= sign * r * d / length**3
            fzz += sign * r**2 / length**3
    hpp = -(frr + 2 * frz * hp + fzz * hp**2) / fz
    return z, hp, hpp


def cartesian_taylor(diopter, radius_m, length_m, order):
    """Taylor jet of sag(radius + length*t) from the unsquared Fermat equation.

    Coefficients multiply powers of dimensionless t, avoiding large derivative
    units. Each new coefficient enters linearly through the nonzero vertex-branch
    partial derivative F_z. Square-root series are formed without finite differences.
    """
    z = np.zeros(order + 1)
    z[0] = diopter.sag(np.array([radius_m]))[0]
    fz = diopter.fermat_gradient(np.array([radius_m]), np.array([z[0]]))[1][0]
    if abs(fz) < 1e-10:
        raise ValueError("Taylor jet is ill-conditioned near a vertical tangent")
    radial = np.polynomial.Polynomial([radius_m, length_m])
    for k in range(1, order + 1):
        residual = 0.0
        for index, focus, side in (
            (diopter.n_o, diopter.z_o_m, -1),
            (diopter.n_i, diopter.z_i_m, 1),
        ):
            if np.isinf(focus):
                residual -= side * index * z[k]
                continue
            axial = np.polynomial.Polynomial(z) - focus
            squared = (radial * radial + axial * axial).coef
            squared = np.pad(squared, (0, max(0, k + 1 - len(squared))))
            root = np.zeros(k + 1)
            root[0] = np.sqrt(squared[0])
            for j in range(1, k + 1):
                root[j] = (squared[j] - np.dot(root[1:j], root[j - 1 : 0 : -1])) / (2 * root[0])
            residual += side * np.sign(focus) * index * root[k]
        z[k] = -residual / fz
    return z


class CartesianPatch:
    """Exact Cartesian pupil and a variational C2 or C4 polynomial annulus.

    The vertex displacement is supplied before optimization. Only the non-optical
    annulus changes to satisfy the fixed fill and pinning constraints. Its free
    coefficients minimize squared curvature load (Pa^2 area), not acoustic power.
    """

    def __init__(self, config, face, optical, vertex_displacement_m, annulus_match_order=2):
        self.config, self.face = config, face
        self.diopter = CartesianDiopter(**optical)
        self.vertex = vertex_displacement_m
        a, radius = config.clear_radius_m, config.radius_m
        length = radius - a
        if annulus_match_order not in (2, 4):
            raise ValueError("Supported pupil/annulus matching orders are C2 and C4")
        self.annulus_match_order = annulus_match_order
        h, hp, hpp = cartesian_derivatives(self.diopter, np.array([a]))
        if annulus_match_order == 2:
            fixed = np.array([h[0] + self.vertex, hp[0] * length, hpp[0] * length**2 / 2])
        else:
            fixed = cartesian_taylor(self.diopter, a, length, annulus_match_order)
            fixed[0] += self.vertex
        count = 2 * annulus_match_order + 4
        matched = len(fixed)
        x, w = leggauss(64)
        ri, wi = a * (x + 1) / 2, w * a / 2
        inner_volume = np.dot(wi * ri, self.diopter.sag(ri) + self.vertex)
        t, wt = (x + 1) / 2, w / 2 * length * (a + length * (x + 1) / 2)
        powers = t[:, None] ** np.arange(count)
        constraint = np.vstack([np.ones(count), wt @ powers])
        rhs = np.array([0.0, -inner_volume]) - constraint[:, :matched] @ fixed
        free = np.linalg.lstsq(constraint[:, matched:], rhs, rcond=None)[0]
        tangent = null_space(constraint[:, matched:])
        sigma = config.surface_tension_n_m[face]
        drho = config.density_kg_m3[face] - config.density_kg_m3[face + 1]

        def residual(q):
            coeff = np.r_[fixed, free + tangent @ q * 1e-4]
            f = np.polynomial.Polynomial(coeff)
            heights, slope, second = f(t), f.deriv()(t) / length, f.deriv(2)(t) / length**2
            k = -(
                second / (1 + slope**2) ** 1.5 + slope / ((a + length * t) * np.sqrt(1 + slope**2))
            )
            traction = sigma * k + drho * config.gravity_m_s2 * heights
            return np.sqrt(wt / wt.sum()) * traction

        opt = least_squares(
            residual, np.zeros(tangent.shape[1]), ftol=1e-12, xtol=1e-12, gtol=1e-10, max_nfev=400
        )
        if not opt.success:
            raise RuntimeError("Annulus load minimization failed.")
        self.annulus = np.polynomial.Polynomial(np.r_[fixed, free + tangent @ opt.x * 1e-4])
        self.annulus_cost_pa2 = float(np.dot(opt.fun, opt.fun))

    def evaluate(self, r, derivative=0):
        r = np.asarray(r, float)
        flat = r.ravel()
        inside = flat <= self.config.clear_radius_m
        result = np.zeros_like(flat)
        if np.any(inside):
            result[inside] = cartesian_derivatives(self.diopter, flat[inside])[derivative]
            if derivative == 0:
                result[inside] += self.vertex
        length = self.config.radius_m - self.config.clear_radius_m
        t = (flat[~inside] - self.config.clear_radius_m) / length
        result[~inside] = self.annulus.deriv(derivative)(t) / length**derivative
        return result.reshape(r.shape)


class DualSurface:
    def __init__(self, config):
        self.config = config
        config.validate()
        degree, radius = config.surface_degree, config.radius_m
        edges = np.unique(
            np.r_[np.linspace(0, 1, config.surface_elements + 1), config.clear_radius_m / radius]
        )
        # The variational annulus matches curvature, but not fourth derivatives.
        # A triple knot admits precisely C2 matching at the optical aperture.
        knots = np.sort(
            np.r_[
                np.repeat(0.0, degree),
                edges,
                np.repeat(1.0, degree),
                np.repeat(config.clear_radius_m / radius, 2),
            ]
        )
        count = len(knots) - degree - 1
        self.raw = BSpline(knots, np.eye(count), degree)
        x, w = leggauss(12)
        xr = (edges[:-1, None] + np.diff(edges)[:, None] * (x + 1) / 2).ravel()
        wr = (np.diff(edges)[:, None] * w[None, :] / 2).ravel()
        self.r = xr * radius
        self.weights = 2 * np.pi * self.r * wr * radius
        constraint = np.vstack(
            [
                self.raw(1),
                self.raw(0, 1),
                self.raw(0, 3),
                (self.weights @ self.raw(xr)) / self.weights.sum(),
            ]
        )
        self.transform = null_space(constraint)
        self.count = self.transform.shape[1]
        self.b, self.dr = self.basis(self.r), self.basis(self.r, 1)
        self.mass = self.b.T @ (self.weights[:, None] * self.b)

    def basis(self, r, derivative=0):
        return (
            self.raw(np.asarray(r) / self.config.radius_m, derivative)
            @ self.transform
            / self.config.radius_m**derivative
        )

    def evaluate(self, coefficients, r, derivative=0):
        # Direct spline evaluation avoids a large basis tensor on FE quadrature.
        spline = BSpline(self.raw.t, self.transform @ coefficients, self.raw.k)
        return (
            spline(np.asarray(r) / self.config.radius_m, derivative)
            / self.config.radius_m**derivative
        )

    def project(self, targets):
        return np.stack(
            [
                np.linalg.solve(self.mass, self.b.T @ (self.weights * t.evaluate(self.r)))
                for t in targets
            ]
        )

    def polynomial_maximum(self, coefficients, lower_m=0, upper_m=None):
        """Stationary-point search for a spline, not for a Cartesian residual.

        Includes interval ends and knots; polynomial roots are floating-point,
        not interval enclosures. Coefficients must represent the quantity being
        audited (e.g. a frozen compliance perturbation), not its target surface.
        """
        radius = self.config.radius_m
        upper_m = radius if upper_m is None else upper_m
        if not 0 <= lower_m < upper_m <= radius:
            raise ValueError("Invalid radial audit interval.")
        spline = BSpline(self.raw.t, self.transform @ coefficients, self.raw.k)
        roots = PPoly.from_spline(spline).derivative().roots(extrapolate=False)
        nodes = np.unique(np.r_[lower_m / radius, upper_m / radius, self.raw.t, roots])
        nodes = nodes[
            np.isfinite(nodes) & (nodes >= lower_m / radius) & (nodes <= upper_m / radius)
        ]
        values = abs(spline(nodes))
        peak = int(np.argmax(values))
        return {
            "max_abs_m": float(values[peak]),
            "radius_m": float(nodes[peak] * radius),
            "stationary_point_and_knot_count": len(nodes),
            "rigorous_interval_bound": False,
        }

    def mechanics(self, coefficients):
        forces, hessians, energies = [], [], []
        for j, coeff in enumerate(coefficients):
            sigma = self.config.surface_tension_n_m[j]
            drho = self.config.density_kg_m3[j] - self.config.density_kg_m3[j + 1]
            gravity = drho * self.config.gravity_m_s2
            height, slope = self.b @ coeff, self.dr @ coeff
            stretch = np.sqrt(1 + slope**2)
            forces.append(
                sigma * self.dr.T @ (self.weights * slope / stretch)
                + gravity * self.b.T @ (self.weights * height)
            )
            hessians.append(
                sigma * self.dr.T @ ((self.weights / stretch**3)[:, None] * self.dr)
                + gravity * self.mass
            )
            # Rationalization avoids cancellation near the flat reference state.
            energies.append(
                np.dot(self.weights, sigma * slope**2 / (stretch + 1) + 0.5 * gravity * height**2)
            )
        return np.concatenate(forces), block_diag(*hessians), float(sum(energies))

    def prescribed_equilibrium(self, force, initial=None):
        q = np.zeros((2, self.count)) if initial is None else np.array(initial).copy()
        for _ in range(40):
            grad, hess, _ = self.mechanics(q)
            step = np.linalg.solve(hess, force - grad).reshape(q.shape)
            q += step
            if np.max(np.abs(step)) < 1e-13:
                return q
        raise RuntimeError("Prescribed-traction equilibrium did not converge.")
