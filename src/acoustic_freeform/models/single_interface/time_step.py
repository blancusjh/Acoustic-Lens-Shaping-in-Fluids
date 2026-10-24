"""First-order ALE momentum/kinematic evolution with actual fluid inertia.

Geometry and acoustic forcing are explicit. Capillarity is linearized implicitly.
Convection uses fluid velocity relative to the moving mesh. Spatial and temporal
convergence are required independently of a successful algebraic solve.
"""

import numpy as np
from scipy.sparse import coo_matrix
from skfem import BilinearForm, asm
from skfem.helpers import dot


def advance_fluid(
    space, coefficients, velocity, fluid, radiation_force, body_rhs, dt_s, acoustic_jacobian=None
):
    q, cfg = space.tangent, space.config
    dt = dt_s / cfg.capillary_time_s
    gradient, hessian = space.derivatives(coefficients)
    stiffness = q.T @ hessian @ q
    if acoustic_jacobian is not None:
        # A frozen radiation Jacobian is a W-method preconditioner. Actual
        # radiation still comes from the current geometry and physical drive;
        # this is a consistent first-order method even when the Jacobian is
        # approximate. It removes the artificial damping caused by splitting
        # two large, nearly balancing surface forces.
        stiffness = stiffness - acoustic_jacobian
    r, z = fluid.velocity_basis.global_coordinates()
    depth = cfg.depth_m / cfg.radius_m
    h = space.evaluate(coefficients, r)
    z_reference = (z - h) / (1 + h / depth)
    modal_rate = q @ (fluid.coupling @ velocity)
    mesh_velocity = (1 + z_reference / depth) * space.evaluate(modal_rate, r)
    advection = np.asarray(fluid.velocity_basis.interpolate(velocity)).copy()
    advection[1] -= mesh_velocity

    @BilinearForm
    def transport(u, v, w):
        return w.x[0] * dot(np.einsum("j...,ij...->i...", w.advection, u.grad), v)

    convection = fluid.inertia * asm(transport, fluid.velocity_basis, advection=advection)
    force = q.T @ (radiation_force - gradient)
    rhs = fluid.inertia / dt * (fluid.mass @ velocity) + fluid.coupling.T @ force + body_rhs
    # C has support only on the free-surface velocity DOFs. Eliminating
    # delta_xi=dt*C*u adds a small dense boundary block to a sparse fluid
    # operator; this avoids one expensive fluid solve per surface mode.
    boundary = np.flatnonzero(np.any(fluid.coupling != 0.0, axis=0))
    trace = fluid.coupling[:, boundary]
    block = dt * trace.T @ stiffness @ trace
    capillary = coo_matrix(
        (block.ravel(), (np.repeat(boundary, len(boundary)), np.tile(boundary, len(boundary)))),
        shape=fluid.mass.shape,
    ).tocsc()
    new_velocity = fluid.solve(rhs, fluid.inertia / dt, convection + capillary)
    delta = dt * (fluid.coupling @ new_velocity)
    return coefficients + q @ delta, new_velocity
