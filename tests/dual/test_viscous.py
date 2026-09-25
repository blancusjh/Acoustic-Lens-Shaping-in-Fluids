import numpy as np
from skfem import LinearForm, asm
from skfem.helpers import dot

from acoustic_freeform.dual.config import DualConfig
from acoustic_freeform.dual.surface import DualSurface
from acoustic_freeform.dual.viscous import assemble_mobility


def test_three_layer_reciprocity_dissipation_and_viscosity_scaling():
    space = DualSurface(DualConfig(surface_elements=4))
    q = np.zeros((2, space.count))
    a = assemble_mobility(space, q, [1, 2, 3], 12, 4)
    b = assemble_mobility(space, q, [2, 4, 6], 12, 4)
    np.testing.assert_allclose(b.mobility, a.mobility / 2, rtol=2e-9, atol=1e-12)
    assert a.diagnostics["work_relative_error"] < 1e-9
    assert np.linalg.norm(a.mobility[: space.count, space.count :]) > 0


def test_manufactured_recirculation_converges_with_interface_pressure_jumps():
    """Independent polynomial solution of -mu Laplacian(u) = body force.

    Streamfunction enforces exact divergence-free flow and no slip on all walls.
    Homogeneous viscosity makes internal phase cuts fictitious: the same exact
    smooth solution must survive their independently discontinuous pressures.
    """
    space = DualSurface(DualConfig(surface_elements=4))
    cfg = space.config
    R, H = cfg.radius_m, cfg.levels_m[-1] - cfg.levels_m[0]
    speed = 1e-3

    def fields(x):
        s, t = x[0] / R, (x[1] - cfg.levels_m[0]) / H
        A, B = s - 2 * s**3 + s**5, 2 - 8 * s**2 + 6 * s**4
        g, gp = t**2 * (1 - t) ** 2, 2 * t - 6 * t**2 + 4 * t**3
        velocity = np.array([-speed * R / H * A * gp, speed * B * g])
        force = np.array(
            [
                speed * R / H * ((-16 * s + 24 * s**3) / R**2 * gp + A / H**2 * (-12 + 24 * t)),
                -speed * ((-32 + 96 * s**2) / R**2 * g + B / H**2 * (2 - 12 * t + 12 * t**2)),
            ]
        )
        return velocity, force

    @LinearForm
    def body(v, w):
        return 2 * np.pi * w.x[0] * dot(fields(w.x)[1], v)

    errors = []
    for nr, nz in [(8, 3), (16, 6), (24, 9)]:
        fluid = assemble_mobility(space, np.zeros((2, space.count)), [1, 1, 1], nr, nz)
        vb = fluid.velocity_basis
        numerical = vb.interpolate(fluid.solve_velocity(asm(body, vb)))
        exact = fields(vb.global_coordinates())[0]
        weight = 2 * np.pi * vb.global_coordinates()[0] * vb.dx
        error = np.sqrt(
            np.sum(weight * np.sum((numerical - exact) ** 2, axis=0))
            / np.sum(weight * np.sum(exact**2, axis=0))
        )
        errors.append(error)
    assert errors[1] < errors[0] / 3
    assert errors[2] < errors[1] / 2
    assert errors[-1] < 0.001
