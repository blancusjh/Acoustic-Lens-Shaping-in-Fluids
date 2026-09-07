"""Passive energy, moving-volume constraint and the zero-inertia asymptote."""

from dataclasses import replace

import numpy as np

from acoustic_freeform.lens.config import LensConfig
from acoustic_freeform.lens.hydrodynamics import assemble_fluid, step_surface, stokes_mobility
from acoustic_freeform.lens.surface import SurfaceSpace
from acoustic_freeform.lens.time_step import advance_fluid


def test_ale_step_has_the_stokes_limit_and_passive_energy_decay():
    cfg = LensConfig(
        surface_modes=10, mesh_radial=18, mesh_vertical=22, geometry_mapping="exact_graph"
    )
    space = SurfaceSpace(cfg)
    target, volume = space.target()
    f = assemble_fluid(space, target)
    zero = np.zeros(f.velocity_basis.N)
    small_mass = replace(f, inertia=1e-8)
    result, _ = advance_fluid(space, target, zero, small_mass, np.zeros(space.count), zero, 0.001)
    reference = step_surface(
        space, target, np.zeros(space.count), stokes_mobility(space, target), 0.001
    )
    assert np.linalg.norm(result - reference) / np.linalg.norm(reference - target) < 1e-6
    c, u = target.copy(), zero.copy()
    energy = space.energy(c)
    for _ in range(6):
        fluid = assemble_fluid(space, c)
        u = fluid.solve(fluid.mass @ u, shift=1.0, viscosity_scale=0.0)
        c, u = advance_fluid(space, c, u, fluid, np.zeros(space.count), zero, 0.0005)
        next_fluid = assemble_fluid(space, c)
        u = next_fluid.solve(next_fluid.mass @ u, shift=1.0, viscosity_scale=0.0)
        next_energy = space.energy(c) + 0.5 * next_fluid.inertia * u @ (next_fluid.mass @ u)
        assert next_energy <= energy + 1e-12
        assert abs(space.volume_vector @ c - volume) < 1e-13
        assert np.linalg.norm(next_fluid.divergence @ u) < 1e-11
        energy = next_energy
