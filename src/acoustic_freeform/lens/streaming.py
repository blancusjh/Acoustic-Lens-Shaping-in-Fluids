"""Bulk absorption streaming for the stated homogeneous, isothermal fluid model.

Eckart force is (2*alpha/c_sound)*I with I=Re(P V*)/2; see Bach & Bruus
(2018), Eq. 55, doi:10.1121/1.5049579. Boundary-layer streaming and
temperature-dependent body forces remain separate, unimplemented mechanisms.
"""

import numpy as np
from skfem import Basis, LinearForm, asm
from skfem.helpers import dot

from .hydrodynamics import assemble_fluid


def absorption_rhs(space, field, drive, fluid):
    cfg = space.config
    phi = field.basis.interpolate(field.solutions @ drive)
    p = cfg.density_kg_m3 * 2 * np.pi * cfg.frequency_hz * cfg.radius_m * phi
    velocity = -1j * phi.grad
    intensity = 0.5 * np.real(p[None, ...] * velocity.conj())
    body = 2 * cfg.attenuation_np_m / cfg.sound_speed_m_s * intensity
    integration = Basis(
        field.basis.mesh, fluid.velocity_basis.elem, quadrature=field.basis.quadrature
    )

    @LinearForm
    def load(v, w):
        return w.x[0] * dot(w.body, v)

    return asm(load, integration, body=body * cfg.radius_m**2 / cfg.surface_tension_n_m)


def stationary_streaming_kernels(space, coefficients, field):
    """Equivalent generalized force for zero interface velocity in steady Stokes flow.

    These kernels are for stationary inversion and overdamped surface motion.
    They do not replace a distributed body load in inertial fluid dynamics.
    Reciprocity eliminates repeated fluid solves for every array cross term.
    """
    cfg, q = space.config, space.tangent
    fluid = assemble_fluid(space, coefficients)
    response = fluid.solve(fluid.coupling.T)
    mobility = fluid.coupling @ response
    adjoints = np.linalg.solve(mobility.T, response.T).T
    integration = Basis(
        field.basis.mesh, fluid.velocity_basis.elem, quadrature=field.basis.quadrature
    )
    weights = (field.basis.dx * field.basis.global_coordinates()[0]).ravel()
    potentials = [field.basis.interpolate(column) for column in field.solutions.T]
    p = (
        cfg.density_kg_m3
        * 2
        * np.pi
        * cfg.frequency_hz
        * cfg.radius_m
        * np.stack([v.ravel() for v in potentials], axis=1)
    )
    velocities = np.stack([(-1j * v.grad).reshape(2, -1) for v in potentials], axis=2)
    factor = (
        2 * cfg.attenuation_np_m / cfg.sound_speed_m_s * cfg.radius_m**2 / cfg.surface_tension_n_m
    )
    reduced_kernels = []
    for column in adjoints.T:
        u = integration.interpolate(column).reshape(2, -1)
        weighted = (
            u[0, :, None] * velocities[0].conj() + u[1, :, None] * velocities[1].conj()
        ) * weights[:, None]
        product = weighted.T @ p
        reduced_kernels.append(0.25 * factor * (product + product.conj().T))
    return np.einsum("ki,iab->kab", q, np.asarray(reduced_kernels))
