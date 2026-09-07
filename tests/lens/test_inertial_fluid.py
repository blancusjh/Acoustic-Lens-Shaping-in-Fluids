"""Energy and independent full-fluid responses constrain the inertial reduction."""

import numpy as np

from acoustic_freeform.lens.config import LensConfig
from acoustic_freeform.lens.dynamics import fluid_reduction, state_matrices
from acoustic_freeform.lens.surface import SurfaceSpace


def test_inertial_reduction_preserves_passive_energy_and_dynamic_response():
    cfg = LensConfig(
        surface_modes=9,
        mesh_radial=16,
        mesh_vertical=20,
        geometry_mapping="exact_graph",
        mesh_radial_distribution="rim_clustered",
    )
    space = SurfaceSpace(cfg)
    _, volume = space.target()
    rest = space.equilibrium(np.zeros(space.count), volume)
    fluid, v, reduced = fluid_reduction(space, rest, shifts=(0.0, 1.0, 20.0, 200.0))
    assert np.linalg.norm(fluid.divergence @ v) / np.linalg.norm(v) < 1e-10
    np.testing.assert_allclose(reduced["mass"], np.eye(v.shape[1]), atol=2e-9)
    assert reduced["static_relative_error"] < 1e-7
    _, hessian = space.derivatives(rest)
    stiffness = space.tangent.T @ hessian @ space.tangent
    a, _ = state_matrices(reduced, stiffness, np.zeros((space.count - 1, 0)), cfg.capillary_time_s)
    ns, nv = reduced["coupling"].shape
    energy = np.zeros_like(a)
    energy[:ns, :ns] = stiffness
    energy[ns:, ns:] = reduced["inertia"] * np.eye(nv)
    dissipation = -(energy @ a + a.T @ energy) / 2
    expected = np.zeros_like(a)
    expected[ns:, ns:] = reduced["damping"] / cfg.capillary_time_s
    np.testing.assert_allclose(dissipation, expected, atol=2e-8)
    assert np.linalg.eigvals(a).real.max() < 0
    for frequency in (0.7, 7.0, 70.0):
        exact = fluid.coupling @ fluid.solve(
            fluid.coupling.T.astype(complex), 1j * frequency * fluid.inertia
        )
        h = reduced["coupling"]
        ritz = h @ np.linalg.solve(
            reduced["damping"] + 1j * frequency * fluid.inertia * np.eye(nv), h.T
        )
        assert np.linalg.norm(exact - ritz) / np.linalg.norm(exact) < 1e-3
