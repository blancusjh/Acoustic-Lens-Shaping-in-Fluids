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


def stokes_mobility(space: SurfaceSpace, coefficients) -> StokesMobility:
    mesh = chamber_mesh(space.config, space, coefficients)
    velocity = Basis(mesh, ElementVector(ElementTriP2()), intorder=6)
    pressure = Basis(mesh, ElementTriP1(), intorder=6)

    @BilinearForm
    def viscosity(u, v, w):
        return 2 * w.x[0] * (ddot(sym_grad(u), sym_grad(v)) + u[0] * v[0] / w.x[0] ** 2)

    @BilinearForm
    def divergence(u, q, w):
        return w.x[0] * q * (div(u) + u[0] / w.x[0])

    a = asm(viscosity, velocity)
    b = asm(divergence, velocity, pressure)
    matrix = bmat([[a, -b.T], [-b, None]], format="csc")
    fixed = np.unique(
        np.r_[
            velocity.get_dofs("wall").all(),
            velocity.get_dofs("base").all(),
            velocity.get_dofs("axis").all(["u^1"]),
        ]
    )
    free = np.setdiff1d(np.arange(matrix.shape[0]), fixed)
    top = FacetBasis(mesh, velocity.elem, facets=mesh.boundaries["surface"], intorder=8)
    loads = []

    @LinearForm
    def normal_load(v, w):
        value = space.basis(w.x[0])[..., w.surface_mode]
        return w.x[0] * value * dot(w.n, v)

    for mode in range(space.count):
        loads.append(asm(normal_load, top, surface_mode=mode))
    loads = np.stack(loads, axis=1)
    tangent = space.tangent
    reduced_mass = tangent.T @ space.mass @ tangent
    inverse_mass = np.linalg.inv(reduced_mass)
    right = loads @ tangent @ inverse_mass
    full_rhs = np.vstack([right, np.zeros((pressure.N, space.count - 1))])
    solution = np.zeros_like(full_rhs)
    solution[free] = splu(matrix[free][:, free]).solve(full_rhs[free])
    fluid = solution[: velocity.N]
    mobility = inverse_mass @ tangent.T @ loads.T @ fluid
    mobility = 0.5 * (mobility + mobility.T)
    error = np.linalg.norm(b @ fluid) / max(np.linalg.norm(fluid), 1e-30)
    if np.linalg.eigvalsh(mobility).min() < -1e-9:
        raise RuntimeError("Fluid mobility violates nonnegative viscous dissipation")
    return StokesMobility(mobility, fluid, velocity, b, float(error))


def step_surface(space, coefficients, force, mobility: StokesMobility, dt_s):
    gradient, hessian = space.derivatives(coefficients)
    q = space.tangent
    dt = dt_s / space.config.capillary_time_s
    rhs = dt * mobility.reduced @ (q.T @ (force - gradient))
    matrix = np.eye(q.shape[1]) + dt * mobility.reduced @ (q.T @ hessian @ q)
    change = q @ np.linalg.solve(matrix, rhs)
    return coefficients + change
