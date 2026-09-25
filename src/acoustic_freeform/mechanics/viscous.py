"""Three-phase, moving-faceted-domain Stokes mobility in SI units.

Axisymmetric P2 velocity, P1 pressure with interface jumps, no-slip walls.
No inertia, acoustic forcing, streaming, thermal or non-axisymmetric solver.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import bmat, vstack
from scipy.sparse.linalg import splu
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    InteriorFacetBasis,
    LinearForm,
    MeshTri,
    asm,
)
from skfem.helpers import ddot, div, dot, sym_grad


@dataclass
class ViscousMobility:
    mobility: np.ndarray
    velocity_operator: np.ndarray
    velocity_basis: object
    diagnostics: dict
    solve_velocity: object


def assemble_mobility(space, coefficients, viscosities_pa_s, radial_cells=32, cells_per_layer=12):
    cfg = space.config
    mu = np.asarray(viscosities_pa_s, float)
    if mu.shape != (3,) or np.any(mu <= 0):
        raise ValueError("Supply three positive dynamic viscosities in Pa s")
    R = cfg.radius_m
    levels = np.asarray(cfg.levels_m)
    r = np.linspace(0, R, radial_cells + 1)
    z = np.unique(
        np.concatenate(
            [np.linspace(a, b, cells_per_layer + 1) for a, b in pairwise(levels)]
        )
    )
    flat = MeshTri.init_tensor(r, z)
    midpoint = flat.p[:, flat.t].mean(axis=1)
    phases = np.searchsorted(levels[1:3], midpoint[1])
    faces = [
        np.flatnonzero(
            np.all(np.isclose(flat.p[1, flat.facets], level, atol=1e-14, rtol=0), axis=0)
        )
        for level in levels[1:3]
    ]
    boundary = flat.boundary_facets()
    fp = flat.p[:, flat.facets[:, boundary]]
    axis = boundary[np.all(np.isclose(fp[0], 0, atol=1e-14), axis=0)]
    wall = np.setdiff1d(boundary, axis)
    p = flat.p.copy()
    h = np.array([space.evaluate(q, p[0]) for q in coefficients])
    bounds = np.vstack(
        [np.full(p.shape[1], levels[0]), levels[1:3, None] + h, np.full(p.shape[1], levels[3])]
    )
    if np.min(np.diff(bounds, axis=0)) <= 0:
        raise ValueError("Interface crossing in viscous mesh")
    for j in range(3):
        inside = (flat.p[1] >= levels[j]) & (flat.p[1] <= levels[j + 1])
        s = (flat.p[1, inside] - levels[j]) / (levels[j + 1] - levels[j])
        p[1, inside] = (1 - s) * bounds[j, inside] + s * bounds[j + 1, inside]
    mesh = MeshTri(p, flat.t)
    velocity = Basis(mesh, ElementVector(ElementTriP2()), intorder=8)
    pressure = Basis(mesh, ElementTriP1(), intorder=8)

    @BilinearForm
    def strain(u, v, w):
        return 4 * np.pi * w.x[0] * (ddot(sym_grad(u), sym_grad(v)) + u[0] * v[0] / w.x[0] ** 2)

    @BilinearForm
    def divergence(u, q, w):
        return 2 * np.pi * w.x[0] * q * (div(u) + u[0] / w.x[0])

    a, blocks = None, []
    for j in range(3):
        elements = np.flatnonzero(phases == j)
        vb, pb = velocity.with_elements(elements), pressure.with_elements(elements)
        aj = mu[j] * asm(strain, vb)
        a = aj if a is None else a + aj
        bj = asm(divergence, vb, pb)
        # Duplicate interface pressure nodes between phases; keep velocity shared.
        active = np.unique(pressure.element_dofs[:, elements])
        blocks.append(bj[active])
    b = vstack(blocks, format="csr")
    fixed = np.unique(
        np.r_[velocity.get_dofs(facets=wall).all(), velocity.get_dofs(facets=axis).all(["u^1"])]
    )
    # Only global pressure constant is a null mode. Interface jumps are physical.
    free = np.setdiff1d(np.arange(velocity.N + b.shape[0]), np.r_[fixed, velocity.N])
    coupling = []

    @LinearForm
    def load(v, w):
        return 2 * np.pi * w.x[0] * w.mode * np.sign(w.n[1]) * dot(w.n, v)

    for facets in faces:
        fb = InteriorFacetBasis(mesh, velocity.elem, facets=facets, side=0, intorder=12)
        modes = space.basis(fb.global_coordinates()[0])
        coupling.append(
            np.stack([asm(load, fb, mode=modes[..., k]) for k in range(space.count)], axis=1)
        )
    loads = np.concatenate(coupling, axis=1)
    surface_mass = block_diag(space.mass, space.mass)
    graph = np.linalg.solve(surface_mass, loads.T)
    # Balance the SI saddle blocks; physical pressure is the solved multiplier
    # times this factor. This is not pressure stabilization or a changed PDE.
    scaled_b = b * (max(mu) / R)
    matrix = bmat([[a, -scaled_b.T], [-scaled_b, None]], format="csc")
    factor = splu(matrix[free][:, free], permc_spec="COLAMD")

    def solve_velocity(load):
        vector = load.ndim == 1
        load = load[:, None] if vector else load
        rhs = np.vstack([load, np.zeros((b.shape[0], load.shape[1]))])
        solution = np.zeros_like(rhs)
        solution[free] = factor.solve(rhs[free])
        fluid = solution[: velocity.N]
        return fluid[:, 0] if vector else fluid

    u = solve_velocity(graph.T)
    mobility = graph @ u
    reciprocal_error = np.linalg.norm(mobility - mobility.T) / np.linalg.norm(mobility)
    work = u.T @ (a @ u)
    work_error = np.linalg.norm(work - mobility) / np.linalg.norm(mobility)
    mobility = (mobility + mobility.T) / 2
    eigen = np.linalg.eigvalsh(mobility)
    if eigen[0] <= 0 or reciprocal_error > 1e-7 or work_error > 1e-7:
        raise RuntimeError("Viscous mobility fails positivity/reciprocity/work checks")
    return ViscousMobility(
        mobility,
        u,
        velocity,
        {
            "velocity_dofs": int(velocity.N),
            "pressure_dofs": int(b.shape[0]),
            "minimum_mobility_m_per_n_s": float(eigen[0]),
            "reciprocity_relative_error": float(reciprocal_error),
            "work_relative_error": float(work_error),
            "phasewise_divergence_relative_residual": float(
                np.linalg.norm(b @ u) / max(np.linalg.norm(u), 1e-30)
            ),
        },
        solve_velocity,
    )
