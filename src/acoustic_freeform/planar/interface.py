"""Linear small-slope interface dynamics in two deep fluid half-spaces.

Two separate limits, not a fabricated interpolation between rheological regimes:
  stokes: exact flat-interface, overdamped two-fluid Stokes mobility;
  weakly_damped: potential-flow inertia plus weak-viscosity free-surface damping,
                restricted to a liquid under a light gas.

Mean height is fixed by volume conservation in the periodic transverse domain.
"""

import numpy as np

from .config import Fluid, Interface
from .spectral import SpectralGrid


class SurfaceDynamics:
    def __init__(
        self, grid: SpectralGrid, lower: Fluid, upper: Fluid, definition: Interface, dt_s: float
    ):
        self.grid, self.lower, self.upper = grid, lower, upper
        self.definition, self.dt = definition, dt_s
        k = grid.k
        self.nonzero = k > 0
        self.stiffness = (
            definition.surface_tension_n_m * k * k
            + (lower.density_kg_m3 - upper.density_kg_m3) * definition.gravity_m_s2
        )
        self.mass = np.divide(
            lower.density_kg_m3 + upper.density_kg_m3, k, out=np.zeros_like(k), where=self.nonzero
        )
        self.equilibrium_inverse = np.divide(
            1.0, self.stiffness, out=np.zeros_like(k), where=self.nonzero
        )
        if definition.model == "stokes":
            self.friction = 2 * (lower.viscosity_pa_s + upper.viscosity_pa_s) * k
            self.mobility = np.divide(1.0, self.friction, out=np.zeros_like(k), where=self.nonzero)
            self.rate = self.stiffness * self.mobility
            self.decay = np.exp(-self.rate * dt_s)
            self.one_minus_decay = -np.expm1(-self.rate * dt_s)
        elif definition.model == "weakly_damped":
            if upper.density_kg_m3 / lower.density_kg_m3 > 0.05:
                raise ValueError(
                    "Weak-damping closure is restricted to a liquid under a light gas; "
                    "it is not a general viscous liquid/liquid solver."
                )
            self.omega2 = np.divide(
                self.stiffness, self.mass, out=np.zeros_like(k), where=self.nonzero
            )
            self.gamma = 2 * lower.viscosity_pa_s / lower.density_kg_m3 * k * k
            self.friction = 2 * self.gamma * self.mass
            disc = np.sqrt(self.gamma**2 - self.omega2 + 0j)
            ep = np.exp((-self.gamma + disc) * dt_s)
            em = np.exp((-self.gamma - disc) * dt_s)
            s = np.divide(ep - em, 2 * disc, out=np.zeros_like(disc), where=np.abs(disc) > 1e-12)
            s[np.abs(disc) <= 1e-12] = np.exp(-self.gamma[np.abs(disc) <= 1e-12] * dt_s) * dt_s
            c = (ep + em) / 2
            self.e11 = (c + self.gamma * s).real
            self.e12 = s.real
            self.e21 = (-self.omega2 * s).real
            self.e22 = (c - self.gamma * s).real
        else:
            raise ValueError("Unknown hydrodynamic regime.")

    def step(
        self, height: np.ndarray, velocity: np.ndarray, pressure: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact modal update for pressure held constant during this fixed step.

        The caller samples smooth external envelopes at the step midpoint,
        making the overall time treatment second order for smooth forcing.
        """
        equilibrium = pressure * self.equilibrium_inverse
        if self.definition.model == "stokes":
            h = self.decay * height + self.one_minus_decay * equilibrium
            v = self.mobility * (pressure - self.stiffness * h)
        else:
            displacement = height - equilibrium
            h = equilibrium + self.e11 * displacement + self.e12 * velocity
            v = self.e21 * displacement + self.e22 * velocity
        h[0, 0], v[0, 0] = 0, 0
        return h, v

    def endpoint_velocity(self, height, velocity, pressure):
        if self.definition.model == "stokes":
            return self.mobility * (pressure - self.stiffness * height)
        return velocity

    def energy(self, height: np.ndarray, velocity: np.ndarray) -> tuple[float, float, float]:
        potential = 0.5 * self.grid.area * np.sum(self.stiffness * np.abs(height) ** 2)
        kinetic = 0.0
        if self.definition.model == "weakly_damped":
            kinetic = 0.5 * self.grid.area * np.sum(self.mass * np.abs(velocity) ** 2)
        dissipation = self.grid.area * np.sum(self.friction * np.abs(velocity) ** 2)
        return float(potential), float(kinetic), float(dissipation)

    def flow_plane(self, velocity_hat: np.ndarray, z_m: float) -> np.ndarray:
        """Slow liquid motion, distinct from fast acoustic particle velocity.

        Potential-flow reconstruction in the weak-damping model; exact linear
        deep Stokes reconstruction for normal forcing in the overdamped model.
        """
        g, k = self.grid, self.grid.k
        decay = np.exp(-k * abs(z_m))
        if self.definition.model == "stokes":
            coefficients = [
                -1j * g.kxx * z_m * decay * velocity_hat,
                -1j * g.kyy * z_m * decay * velocity_hat,
                (1 + k * abs(z_m)) * decay * velocity_hat,
            ]
        else:
            inv_k = np.divide(1.0, k, out=np.zeros_like(k), where=k > 0)
            sign = 1 if z_m <= 0 else -1
            coefficients = [
                sign * 1j * g.kxx * inv_k * decay * velocity_hat,
                sign * 1j * g.kyy * inv_k * decay * velocity_hat,
                decay * velocity_hat,
            ]
        return np.stack([g.inverse(c).real for c in coefficients])
