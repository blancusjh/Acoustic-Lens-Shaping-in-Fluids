"""Geometric conservation and independent physical limits for the graph map."""

from dataclasses import replace

import numpy as np
from skfem import Basis, ElementTriP3, FacetBasis

from acoustic_freeform.lens.acoustics import CavityAcoustics
from acoustic_freeform.lens.config import LensConfig
from acoustic_freeform.lens.geometry import chamber_mesh
from acoustic_freeform.lens.surface import SurfaceSpace
from acoustic_freeform.lens.validation import analytic_flat_cavity


def test_graph_volume_normals_and_inverse_follow_analytic_paraboloid():
    cfg = LensConfig(
        geometry_mapping="exact_graph",
        mesh_radial=8,
        mesh_vertical=10,
        surface_modes=8,
        mesh_radial_distribution="rim_clustered",
    )
    space = SurfaceSpace(cfg)
    c = np.zeros(space.count)
    cap = 0.23
    c[0] = -cap / 2
    mesh = chamber_mesh(cfg, space, c)
    basis = Basis(mesh, ElementTriP3(), intorder=10)
    mapping = basis.mapping
    x = mapping.F(basis.X)
    assert abs(np.sum(basis.dx * x[0]) - (cfg.depth_m / cfg.radius_m / 2 + cap / 4)) < 2e-13
    top = FacetBasis(mesh, ElementTriP3(), facets=mesh.boundaries["surface"], intorder=10)
    r, z = top.global_coordinates()
    np.testing.assert_allclose(z, cap * (1 - r * r), atol=2e-15)
    norm = np.sqrt(1 + (2 * cap * r) ** 2)
    np.testing.assert_allclose(top.normals[0], 2 * cap * r / norm, atol=3e-13)
    np.testing.assert_allclose(top.normals[1], 1 / norm, atol=3e-13)
    assert abs(np.sum(top.dx * top.normals[1] * r) - 0.5) < 2e-14
    pulled = mapping.invF(x)
    np.testing.assert_allclose(pulled, np.broadcast_to(basis.X[:, None], pulled.shape), atol=2e-13)
    identity = np.einsum("ij... , jk... -> ik...", mapping.invDF(basis.X), mapping.DF(basis.X))
    np.testing.assert_allclose(
        identity, np.broadcast_to(np.eye(2)[:, :, None, None], identity.shape), atol=1e-13
    )


def test_graph_acoustics_matches_independent_flat_bessel_solution():
    cfg = LensConfig(
        geometry_mapping="exact_graph",
        align_array_mesh=True,
        frequency_hz=420000,
        array_rows=3,
        mesh_radial=28,
        mesh_vertical=48,
        acoustic_order=3,
        surface_modes=8,
    )
    space = SurfaceSpace(cfg)
    field = CavityAcoustics(cfg, space).solve_basis(np.zeros(space.count))
    drive = np.array([0.009 + 0.004j, -0.003 + 0.007j, 0.006 - 0.002j])
    points = np.array([field.radial_samples, np.zeros_like(field.radial_samples)])
    exact = analytic_flat_cavity(cfg, drive, points, normal_velocity=True)
    actual = field.normal_velocity_basis @ drive
    weights = field.quadrature_weights
    error = np.sqrt(
        np.sum(weights * (abs(actual) ** 2 - abs(exact) ** 2) ** 2)
        / np.sum(weights * abs(exact) ** 4)
    )
    assert error < 3e-3
    assert field.residual < 1e-10


def test_exact_geometry_remains_exact_for_resolved_high_surface_modes():
    cfg = LensConfig(
        geometry_mapping="exact_graph", surface_modes=24, mesh_radial=18, mesh_vertical=20
    )
    space = SurfaceSpace(cfg)
    c, _ = space.target()
    c[-1] += 1e-5
    for distribution in ("uniform", "rim_clustered"):
        local = replace(cfg, mesh_radial_distribution=distribution)
        mesh = chamber_mesh(local, space, c)
        top = FacetBasis(mesh, ElementTriP3(), facets=mesh.boundaries["surface"], intorder=10)
        r, z = top.global_coordinates()
        h, hp = space.evaluate(c, r), space.evaluate(c, r, 1)
        np.testing.assert_allclose(z, h, atol=2e-15)
        np.testing.assert_allclose(top.normals[0] / top.normals[1], -hp, atol=3e-12)
