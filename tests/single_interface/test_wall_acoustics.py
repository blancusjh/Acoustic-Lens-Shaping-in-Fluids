"""Independent quarter-wave damping limit and total acoustic energy balance."""

import numpy as np
from scipy.sparse.linalg import eigs
from skfem import Basis, BilinearForm, ElementTriP3, asm
from skfem.helpers import dot, grad

from acoustic_freeform.single_interface.acoustics import CavityAcoustics, wall_gradient_matrix
from acoustic_freeform.single_interface.config import LensConfig
from acoustic_freeform.single_interface.geometry import chamber_mesh
from acoustic_freeform.single_interface.surface import SurfaceSpace


def test_quarter_wave_viscous_wall_frequency_and_damping():
    cfg = LensConfig(
        surface_modes=8,
        mesh_radial=16,
        mesh_vertical=24,
        viscosity_pa_s=0.001,
        attenuation_np_m=0.0,
        geometry_mapping="exact_graph",
    )
    space = SurfaceSpace(cfg)
    basis = Basis(chamber_mesh(cfg, space, np.zeros(space.count)), ElementTriP3(), intorder=8)

    @BilinearForm
    def stiffness(u, v, w):
        return w.x[0] * dot(grad(u), grad(v))

    @BilinearForm
    def mass(u, v, w):
        return w.x[0] * u * v

    omega0 = np.pi * cfg.sound_speed_m_s / (2 * cfg.depth_m)
    delta = np.sqrt(2 * cfg.viscosity_pa_s / (cfg.density_kg_m3 * omega0))
    matrix = asm(stiffness, basis).astype(complex) - (1 + 1j) * delta / (
        2 * cfg.radius_m
    ) * wall_gradient_matrix(basis)
    m = asm(mass, basis)
    free = np.setdiff1d(np.arange(basis.N), basis.get_dofs("surface").all())
    target = (omega0 * cfg.radius_m / cfg.sound_speed_m_s) ** 2
    eigen = eigs(matrix[free][:, free], M=m[free][:, free], k=1, sigma=target)[0][0]
    ratio = np.sqrt(eigen / target)
    # Rayleigh perturbation of the analytic p=cos(k_z*(z+d)) cavity mode:
    # delta(omega)/omega = -(1+i)*delta/(2*R). Base tangential loss is zero.
    expected = -(1 + 1j) * delta / (2 * cfg.radius_m)
    assert abs((ratio - 1) - expected) / abs(expected) < 0.01


def test_wall_and_bulk_absorption_equal_source_work():
    cfg = LensConfig(
        surface_modes=10,
        mesh_radial=18,
        mesh_vertical=24,
        acoustic_order=3,
        frequency_hz=420000,
        array_rows=3,
        geometry_mapping="exact_graph",
        viscous_wall_acoustics=True,
    )
    space = SurfaceSpace(cfg)
    c, _ = space.target()
    field = CavityAcoustics(cfg, space).solve_basis(c)
    drive = np.array([0.01 + 0.02j, 0.004 - 0.01j, 0.005j])
    phi = field.solutions @ drive
    omega = 2 * np.pi * cfg.frequency_hz
    p = cfg.density_kg_m3 * omega * cfg.radius_m * phi
    source = -np.pi * cfg.radius_m**2 * np.real(np.vdot(-1j * (field.wall_loads @ drive), p))
    heat = (
        cfg.attenuation_np_m
        * abs(field.basis.interpolate(p)) ** 2
        / (cfg.density_kg_m3 * cfg.sound_speed_m_s)
    )
    bulk = (
        2
        * np.pi
        * cfg.radius_m**3
        * np.sum(field.basis.dx * field.basis.global_coordinates()[0] * heat)
    )
    delta = np.sqrt(2 * cfg.viscosity_pa_s / (cfg.density_kg_m3 * omega))
    wall = (
        np.pi
        * cfg.viscosity_pa_s
        * cfg.radius_m**2
        / delta
        * np.vdot(phi, wall_gradient_matrix(field.basis) @ phi).real
    )
    assert source > 0 and bulk > 0 and wall > 0
    np.testing.assert_allclose(source, bulk + wall, rtol=1e-8)
