"""Absorption power and independent fluid solves verify second-order forcing."""

import numpy as np

from acoustic_freeform.single_interface.acoustics import CavityAcoustics
from acoustic_freeform.single_interface.config import LensConfig
from acoustic_freeform.single_interface.hydrodynamics import assemble_fluid
from acoustic_freeform.single_interface.streaming import absorption_rhs
from acoustic_freeform.single_interface.surface import SurfaceSpace


def test_bulk_streaming_kernels_match_direct_force_and_dissipated_power():
    cfg = LensConfig(
        surface_modes=10,
        mesh_radial=18,
        mesh_vertical=24,
        acoustic_order=3,
        frequency_hz=420000,
        array_rows=3,
        geometry_mapping="exact_graph",
        bulk_streaming=True,
    )
    space = SurfaceSpace(cfg)
    c, _ = space.target()
    drive = np.array([0.01 + 0.02j, 0.004 - 0.01j, 0.005j])
    field = CavityAcoustics(cfg, space).solve_basis(c)
    fluid = assemble_fluid(space, c)
    load = absorption_rhs(space, field, drive, fluid)
    velocity = fluid.solve(load)
    response = fluid.solve(fluid.coupling.T)
    mobility = fluid.coupling @ response
    equivalent = space.tangent.T @ (field.force(drive) - field.radiation_force(drive))
    np.testing.assert_allclose(
        equivalent, np.linalg.solve(mobility, fluid.coupling @ velocity), rtol=1e-8, atol=1e-13
    )
    held = velocity - response @ equivalent
    assert np.linalg.norm(fluid.coupling @ held) < 1e-10
    # Work supplied by the body load balances dissipation, with zero interface work.
    np.testing.assert_allclose(held @ (fluid.viscosity @ held), held @ load, rtol=1e-8, atol=1e-15)
    p = cfg.density_kg_m3 * 2 * np.pi * cfg.frequency_hz * cfg.radius_m * (field.solutions @ drive)
    source = -np.pi * cfg.radius_m**2 * np.real(np.vdot(-1j * (field.wall_loads @ drive), p))
    heat = (
        cfg.attenuation_np_m
        * abs(field.basis.interpolate(p)) ** 2
        / (cfg.density_kg_m3 * cfg.sound_speed_m_s)
    )
    absorbed = (
        2
        * np.pi
        * cfg.radius_m**3
        * np.sum(field.basis.dx * field.basis.global_coordinates()[0] * heat)
    )
    np.testing.assert_allclose(source, absorbed, rtol=1e-9)
