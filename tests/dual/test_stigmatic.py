import numpy as np

from acoustic_freeform.dual.config import DualConfig
from acoustic_freeform.dual.optics import fixed_detector_spot_jacobian, trace_graph_pair, trace_pair
from acoustic_freeform.dual.stigmatic import pair_prescription, stigmatic_pair
from acoustic_freeform.dual.surface import DualSurface


def test_both_interface_spot_derivative_against_independent_perturbed_traces():
    cfg = DualConfig(surface_elements=20)
    space = DualSurface(cfg)
    indices = [1.33, 1.5, 1.33]
    targets = stigmatic_pair(cfg, indices, [-0.101, 0.201, 0.151], [-0.00015, 0.00012])
    q = space.project(targets)
    # Interior rays avoid the nondifferentiable aperture-loss gate in this
    # derivative test; production acceptance still includes the full pupil rim.
    r = np.linspace(0, 0.0019, 101)
    launch_z = -0.001 + targets[0].evaluate(r)
    _, jac = fixed_detector_spot_jacobian(space, q, indices, -0.101, 0.151, r, launch_z)
    angle = np.arange(len(r)) * np.pi * (3 - np.sqrt(5))
    radial_unit = np.column_stack([np.cos(angle), np.sin(angle)])
    rng = np.random.default_rng(27)
    for _ in range(3):
        direction = rng.normal(size=q.shape)

        def signed_spot(state):
            traced = trace_pair(
                space, state, indices, -0.101, 0.151, launch_radius_m=r, launch_height_m=launch_z
            )
            assert traced["all_rays_transmitted"]
            return np.sum(traced["spots_m"] * radial_unit, axis=1)

        eps = 1e-8
        finite = (signed_spot(q + eps * direction) - signed_spot(q - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(jac @ direction.ravel(), finite, rtol=2e-6, atol=1e-6)


def test_lab_conjugates_share_intermediate_focus():
    cfg = DualConfig()
    faces = pair_prescription(cfg, [1.33, 1.5, 1.33], [-0.101, 0.201, 0.151], [-0.00015, 0.00012])
    vertices = np.array(cfg.levels_m[1:3]) + [-0.00015, 0.00012]
    assert faces[1]["optical"]["z_o_m"] > 0  # virtual object for the back interface
    np.testing.assert_allclose(
        [vertices[0] + faces[0]["optical"]["z_i_m"], vertices[1] + faces[1]["optical"]["z_o_m"]],
        [0.201, 0.201],
        atol=1e-16,
    )


def test_joint_pair_snell_trace_and_constant_total_path():
    cfg = DualConfig()
    indices = [1.33, 1.5, 1.33]
    pair = stigmatic_pair(cfg, indices, [-0.101, 0.201, 0.151], [-0.00015, 0.00012])
    radius = np.linspace(0, cfg.clear_radius_m, 401)  # includes axis and pupil rim
    traced = trace_graph_pair(
        cfg,
        lambda j, r, derivative=0: pair[j].evaluate(r, derivative),
        indices,
        -0.101,
        0.151,
        launch_radius_m=radius,
        launch_height_m=cfg.levels_m[1] + pair[0].evaluate(radius),
    )
    assert traced["all_rays_transmitted"]
    assert traced["sampled_1um_pass"]
    assert traced["max_radius_m"] < 1e-12
    a, b = traced["intersections_m"]
    # Independent physical positive path lengths, not signed single-oval residuals.
    path = (
        indices[0] * np.linalg.norm(a - [0, -0.101], axis=1)
        + indices[1] * np.linalg.norm(b - a, axis=1)
        + indices[2] * np.linalg.norm(b - [0, 0.151], axis=1)
    )
    assert np.ptp(path) < 2e-15


def test_lost_rays_cannot_pass_spot_gate():
    cfg = DualConfig()
    traced = trace_graph_pair(
        cfg,
        lambda j, r, derivative=0: np.zeros_like(r),
        [1.33, 1.5, 1.33],
        -0.101,
        0.151,
        rays=1001,
    )
    assert not traced["all_rays_transmitted"]
    assert not traced["sampled_1um_pass"]
    assert traced["max_radius_m"] >= traced["rms_radius_m"]


def test_front_rim_height_sign_controls_fixed_edge_ray_interception():
    cfg = DualConfig()
    indices = [1.33, 1.5, 1.33]
    pair = stigmatic_pair(cfg, indices, [-0.101, 0.201, 0.151], [-0.00015, 0.00012])
    r = np.linspace(0, cfg.clear_radius_m, 101)
    for shift, expected_transmission in [(-5e-9, True), (5e-9, False)]:

        def shifted(j, radius, derivative=0, shift=shift):
            # This geometric piston perturbation is not a volume-constrained
            # mechanical state; it independently tests the interception sign.
            return pair[j].evaluate(radius, derivative) + (
                shift if j == 0 and derivative == 0 else 0
            )

        trace = trace_graph_pair(
            cfg,
            shifted,
            indices,
            -0.101,
            0.151,
            launch_radius_m=r,
            launch_height_m=cfg.levels_m[1] + pair[0].evaluate(r),
        )
        assert trace["all_rays_transmitted"] == expected_transmission
