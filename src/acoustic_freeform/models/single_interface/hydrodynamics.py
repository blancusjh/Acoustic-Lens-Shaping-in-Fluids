"""Finite-depth axisymmetric Stokes FEM coupled to volume-preserving surface modes.

Taylor-Hood P2/P1, full axisymmetric strain including hoop strain, no slip at
base/wall, zero radial velocity on the axis and natural free tangential stress.
The reduced mobility is obtained from fluid solves, not an invented relaxation
constant. It is recomputed on the actual curved chamber mesh.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import splu
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    FacetBasis,
    LinearForm,
    asm,
)
from skfem.helpers import ddot, div, dot, sym_grad

from .geometry import chamber_mesh
from .surface import SurfaceSpace


@dataclass
class StokesMobility:
    reduced: np.ndarray
    velocity_operator: np.ndarray
    velocity_basis: Basis
    divergence_matrix: object
    incompressibility_error: float


@dataclass
class FluidOperators:
    """Axisymmetric weak operators, with reciprocal force/kinematic coupling.

    Length is scaled by chamber radius, stress by sigma/radius, velocity by
    sigma/mu and time by mu*radius/sigma. The inertial coefficient is Oh^-2.
    Pressure is fixed by the free-surface traction, not an arbitrary gauge.
    """

    velocity_basis: Basis
    pressure_basis: Basis
    viscosity: object
    mass: object
    divergence: object
    coupling: np.ndarray
    free: np.ndarray
    inertia: float

    def solve(self, velocity_rhs, shift=0.0, extra_operator=None, viscosity_scale=1.0):
        """Solve (A + shift*M)u - B^T p = rhs, Bu=0 with the wall constraints."""
        operator = viscosity_scale * self.viscosity + shift * self.mass
        if extra_operator is not None:
            operator = operator + extra_operator
        matrix = bmat(
            [[operator, -self.divergence.T], [-self.divergence, None]],
            format="csc",
        )
        rhs = np.asarray(velocity_rhs)
        vector = rhs.ndim == 1
        if vector:
            rhs = rhs[:, None]
        full = np.vstack([rhs, np.zeros((self.pressure_basis.N, rhs.shape[1]))])
        solution = np.zeros(full.shape, dtype=np.result_type(full.dtype, matrix.dtype))
        # The indefinite velocity/pressure block needs a column ordering.
        # Symmetric minimum-degree ordering suitable for Helmholtz introduces
        # excessive fill at the zero pressure diagonal in this saddle system.
        solution[self.free] = splu(matrix[self.free][:, self.free], permc_spec="COLAMD").solve(
            full[self.free]
        )
        fluid = solution[: self.velocity_basis.N]
        return fluid[:, 0] if vector else fluid


def assemble_fluid(space: SurfaceSpace, coefficients) -> FluidOperators:
    mesh = chamber_mesh(space.config, space, coefficients)
    velocity = Basis(mesh, ElementVector(ElementTriP2()), intorder=6)
    pressure = Basis(mesh, ElementTriP1(), intorder=6)

    @BilinearForm
    def viscosity(u, v, w):
        return 2 * w.x[0] * (ddot(sym_grad(u), sym_grad(v)) + u[0] * v[0] / w.x[0] ** 2)

    @BilinearForm
    def divergence(u, q, w):
        return w.x[0] * q * (div(u) + u[0] / w.x[0])

    @BilinearForm
    def kinetic_mass(u, v, w):
        return w.x[0] * dot(u, v)

    a = asm(viscosity, velocity)
    b = asm(divergence, velocity, pressure)
    mass = asm(kinetic_mass, velocity)
    fixed = np.unique(
        np.r_[
            velocity.get_dofs("wall").all(),
            velocity.get_dofs("base").all(),
            velocity.get_dofs("axis").all(["u^1"]),
        ]
    )
    free = np.setdiff1d(np.arange(velocity.N + pressure.N), fixed)
    top = FacetBasis(mesh, velocity.elem, facets=mesh.boundaries["surface"], intorder=8)
    loads = []

    @LinearForm
    def normal_load(v, w):
        return w.x[0] * w.mode_value * dot(w.n, v)

    values = space.basis(top.global_coordinates()[0])
    for mode in range(space.count):
        loads.append(asm(normal_load, top, mode_value=values[..., mode]))
    loads = np.stack(loads, axis=1)
    tangent = space.tangent
    reduced_mass = tangent.T @ space.mass @ tangent
    inverse_mass = np.linalg.inv(reduced_mass)
    right = loads @ tangent @ inverse_mass
    cfg = space.config
    inertia = cfg.density_kg_m3 * cfg.surface_tension_n_m * cfg.radius_m / cfg.viscosity_pa_s**2
    return FluidOperators(velocity, pressure, a, mass, b, right.T, free, inertia)


def stokes_mobility(space: SurfaceSpace, coefficients) -> StokesMobility:
    operators = assemble_fluid(space, coefficients)
    fluid = operators.solve(operators.coupling.T)
    mobility = operators.coupling @ fluid
    mobility = 0.5 * (mobility + mobility.T)
    error = np.linalg.norm(operators.divergence @ fluid) / max(np.linalg.norm(fluid), 1e-30)
    if np.linalg.eigvalsh(mobility).min() < -1e-9:
        raise RuntimeError("Fluid mobility violates nonnegative viscous dissipation")
    return StokesMobility(
        mobility, fluid, operators.velocity_basis, operators.divergence, float(error)
    )


def step_surface(space, coefficients, force, mobility: StokesMobility, dt_s):
    gradient, hessian = space.derivatives(coefficients)
    q = space.tangent
    dt = dt_s / space.config.capillary_time_s
    rhs = dt * mobility.reduced @ (q.T @ (force - gradient))
    matrix = np.eye(q.shape[1]) + dt * mobility.reduced @ (q.T @ hessian @ q)
    change = q @ np.linalg.solve(matrix, rhs)
    return coefficients + change
