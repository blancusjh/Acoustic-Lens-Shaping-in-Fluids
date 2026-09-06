"""Curved-cavity Helmholtz FEM and radiation forcing from a cylindrical array.

Axisymmetric drive ties all azimuthal sectors of each physical row. The free
liquid/air interface is a pressure-release boundary, a stated high-impedance-
contrast approximation. Wall rows prescribe normal velocity; base and gaps are
rigid. Acoustic source, cavity and evolving surface share the same geometry.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import splu
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP2,
    ElementTriP3,
    ElementTriP4,
    FacetBasis,
    LinearForm,
    asm,
)
from skfem.helpers import dot, grad

from .config import LensConfig
from .geometry import chamber_mesh
from .surface import SurfaceSpace


@dataclass
class CavityField:
    basis: Basis
    solutions: np.ndarray
    force_kernels: np.ndarray
    radial_samples: np.ndarray
    normal_velocity_basis: np.ndarray
    quadrature_weights: np.ndarray
    wall_loads: np.ndarray
    residual: float

    def force(self, drive):
        return np.einsum("j,ijk,k->i", drive.conj(), self.force_kernels, drive).real

    def surface_pressure(self, drive, density):
        return density / 4 * np.abs(self.normal_velocity_basis @ drive) ** 2


class CavityAcoustics:
    def __init__(self, config: LensConfig, surface: SurfaceSpace):
        self.config, self.surface = config, surface

    def solve_basis(self, coefficients):
        cfg, space = self.config, self.surface
        mesh = chamber_mesh(cfg, space, coefficients)
        element = {2: ElementTriP2, 3: ElementTriP3, 4: ElementTriP4}[cfg.acoustic_order]()
        order = 2 * cfg.acoustic_order + 2
        basis = Basis(mesh, element, intorder=order)
        omega = 2 * np.pi * cfg.frequency_hz
        wave_number = (omega / cfg.sound_speed_m_s + 1j * cfg.attenuation_np_m) * cfg.radius_m

        @BilinearForm(dtype=np.complex128)
        def helmholtz(u, v, w):
            return w.x[0] * (dot(grad(u), grad(v)) - wave_number**2 * u * v)

        matrix = asm(helmholtz, basis).tocsc()
        wall = FacetBasis(mesh, element, facets=mesh.boundaries["wall"], intorder=max(8, order))
        loads = []
        depth = cfg.depth_m / cfg.radius_m
        pitch = depth / cfg.array_rows

        @LinearForm(dtype=np.complex128)
        def patch(v, w):
            offset = (w.x[1] - w.patch_center) / w.patch_half
            envelope = np.where(np.abs(offset) < 1, 0.5 * (1 + np.cos(np.pi * offset)), 0)
            return 1j * w.x[0] * envelope * v

        for row in range(cfg.array_rows):
            center = -depth + (row + 0.5) * pitch
            half = pitch * cfg.element_fill / 2
            loads.append(asm(patch, wall, patch_center=center, patch_half=half))
        rhs = np.stack(loads, axis=1)
        fixed = basis.get_dofs("surface").all()
        free = np.setdiff1d(np.arange(basis.N), fixed)
        solution = np.zeros((basis.N, cfg.array_rows), dtype=complex)
        lu = splu(matrix[free][:, free])
        solution[free] = lu.solve(rhs[free])
        residual = np.linalg.norm(matrix[free] @ solution - rhs[free]) / max(
            np.linalg.norm(rhs[free]), 1e-30
        )
        top = FacetBasis(mesh, element, facets=mesh.boundaries["surface"], intorder=max(8, order))
        coordinates = top.global_coordinates()
        radial = coordinates[0].ravel()
        slope = space.evaluate(coefficients, radial, 1)
        # ds*n_z = dr; the graph force does work Pi*delta_h*2*pi*r*dr.
        weights = top.dx.ravel() / np.sqrt(1 + slope * slope) * radial
        normal_velocity = []
        for i in range(cfg.array_rows):
            gradient = top.interpolate(solution[:, i]).grad
            normal_velocity.append((-1j * np.sum(gradient * top.normals, axis=0)).ravel())
        velocities = np.stack(normal_velocity, axis=1)
        b = space.basis(radial)
        factor = cfg.density_kg_m3 * cfg.radius_m / (4 * cfg.surface_tension_n_m)
        kernels = factor * np.einsum("s,si,sj,sk->ijk", weights, b, velocities.conj(), velocities)
        return CavityField(
            basis, solution, kernels, radial, velocities, weights, rhs, float(residual)
        )
