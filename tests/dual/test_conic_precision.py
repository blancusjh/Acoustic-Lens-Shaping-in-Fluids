"""Independent separable quadratic target for optional sequential convex steps."""

import numpy as np
import pytest

pytest.importorskip("cvxpy")

from acoustic_freeform.dual.conic_precision import optimize


@pytest.mark.parametrize("precondition", [False, True])
@pytest.mark.parametrize("include_carrier", [False, True])
@pytest.mark.parametrize("second_order_corrections", [0, 3])
def test_two_scalar_quadratic_targets_recover_unit_commands(
    precondition, include_carrier, second_order_corrections
):
    kernels = np.zeros((2, 2, 2), complex)
    kernels[0, 0, 0] = kernels[1, 1, 1] = 1
    cache = {
        "force_kernels": kernels,
        "required_force_n": np.ones(2),
        "compliance_observation_m_n": 1e-8 * np.eye(2),
        "observation_weights": np.ones(2),
        "observation_faces": np.array([0, 1]),
        "carrier_height_response_s": (
            np.zeros((2, 2), complex) if include_carrier else 1e-6 * np.eye(2, dtype=complex)
        ),
        "carrier_faces": np.array([0, 1]),
    }
    source, history = optimize(
        cache,
        np.array([0.8, 0.7], complex),
        1,
        steps=35,
        precondition=precondition,
        include_carrier=include_carrier,
        second_order_corrections=second_order_corrections,
    )
    np.testing.assert_allclose(abs(source), 1, atol=1e-5)
    merits = [r["merit"] for r in history]
    assert np.all(np.diff(merits) <= 0)
    assert np.max(abs(source)) <= 1 + 1e-14
    assert max(history[-1]["frozen_compliance_max_m"]) < 1e-12
    if include_carrier:
        assert history[-1]["conservative_frozen_combined_max_m"] < 1e-12
    else:
        assert min(history[-1]["frozen_carrier_max_m"]) > 0.9e-6


@pytest.mark.parametrize("nonlinear_rim_projection", [False, True])
def test_one_sided_rim_constraint_recovers_known_quadratic_optimum(nonlinear_rim_projection):
    kernels = np.zeros((2, 2, 2), complex)
    kernels[0, 0, 0] = kernels[1, 1, 1] = 1
    cache = {
        "force_kernels": kernels,
        "required_force_n": np.ones(2),
        "compliance_observation_m_n": 1e-8 * np.eye(2),
        "observation_weights": np.ones(2),
        "observation_faces": np.array([0, 1]),
        "observation_radii_m": np.array([0.002, 0.002]),
        "carrier_height_response_s": np.zeros((2, 2), complex),
        "carrier_faces": np.array([0, 1]),
    }
    source, history = optimize(
        cache,
        np.array([0.8, 0.7], complex),
        1,
        steps=35,
        include_carrier=False,
        clear_radius_m=0.002,
        front_rim_upper_bound_m=-1e-9,
        nonlinear_rim_projection=nonlinear_rim_projection,
    )
    # The analytic constrained minimax optimum has |a_front|²=0.9.
    np.testing.assert_allclose(abs(source[0]) ** 2, 0.9, atol=2e-6)
    assert history[-1]["predicted_front_rim_bound_met"]
