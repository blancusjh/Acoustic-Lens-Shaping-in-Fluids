"""Internal source derivative verification, not independent physical validation."""

import numpy as np
import pytest

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.inverse.screening import (
    PrecisionObjective,
    WhitenedObjective,
    add_pupil_slope_observations,
    constrained_optimize,
)
from acoustic_freeform.mechanics.surface import DualSurface


def test_svd_source_coordinates_resolve_weak_rotated_mode_without_gram_squaring():
    rotation = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    kernel = rotation @ np.diag([1.0, 1e-10]) @ rotation.T
    cache = {
        "force_kernels": kernel[None].astype(complex),
        "required_force_n": np.zeros(1),
        "compliance_observation_m_n": np.ones((1, 1)) * 1e-8,
        "observation_weights": np.ones(1),
        "carrier_height_response_s": np.zeros((1, 2), complex),
    }
    base = PrecisionObjective(cache, 0, 1.0, 0.0)
    working = WhitenedObjective(base, cutoff=1e-24, spectral_method="svd")
    assert working.channels == 2
    np.testing.assert_allclose(np.sqrt(working.eigenvalues), [1e-10, 1.0], rtol=2e-6)
    source = np.array([0.2 + 0.3j, -0.1 + 0.2j])
    np.testing.assert_allclose(working.unpack(working.encode(source)), source, atol=1e-11)
    np.testing.assert_allclose(
        working.metrics(working.encode(source))[0],
        base.metrics(np.r_[source.real, source.imag])[0],
        atol=1e-11,
    )


def test_slope_observations_preserve_height_rows_and_match_height_difference():
    space = DualSurface(DualConfig(surface_elements=8))
    _, stiffness, _ = space.mechanics(np.zeros((2, space.count)))
    cache = {
        "compliance_observation_m_n": np.zeros((2, 2 * space.count)),
        "observation_radii_m": np.array([0.0, 0.0]),
        "observation_faces": np.array([0, 1]),
        "observation_weights": np.ones(2),
    }
    add_pupil_slope_observations(cache, space, stiffness, 1e-3)
    assert np.array_equal(cache["observation_kind"][:2], [0, 0])
    assert np.all(cache["observation_kind"][2:] == 1)
    q = np.random.default_rng(31).normal(size=(2, space.count)) * 1e-8
    observed = cache["compliance_observation_m_n"] @ stiffness @ q.ravel()
    for face in range(2):
        mask = (cache["observation_kind"] == 1) & (cache["observation_faces"] == face)
        r = cache["observation_radii_m"][mask]
        eps = 1e-9
        finite = (space.evaluate(q[face], r + eps) - space.evaluate(q[face], r - eps)) / (2 * eps)
        np.testing.assert_allclose(observed[mask], finite * 1e-3, rtol=1e-6, atol=1e-15)


def test_scaled_disk_solver_recovers_known_source_from_infeasible_seed():
    # One positive scalar kernel: |a|² = 1/4, with |a| <= 1.
    cache = {
        "force_kernels": np.ones((1, 1, 1), complex),
        "required_force_n": np.array([0.25]),
        "compliance_observation_m_n": np.array([[1e-4]]),
        "observation_weights": np.ones(1),
        "carrier_height_response_s": np.zeros((1, 1), complex),
    }
    base = PrecisionObjective(cache, 0, 1 / np.sqrt(2), effort_weight=0)
    working = WhitenedObjective(base)
    result = constrained_optimize(working, np.array([2.0, 0.0]), 100, objective_scale=1e-8)
    assert result.success
    assert result.initial_projected_source_peak_m_s == 2
    np.testing.assert_allclose(abs(working.unpack(result.x)), [0.5], atol=1e-7)
    assert np.min(working.source_slack(result.x)) >= 0
    residual = working.residual(result.x)[: -base.channels]
    np.testing.assert_allclose(result.cost, 0.5 * residual @ residual, rtol=1e-12)


@pytest.mark.parametrize("peak_power", [None, 8, 16])
@pytest.mark.parametrize("include_carrier", [False, True])
def test_complex_carrier_and_force_residual_jacobian(peak_power, include_carrier):
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
    cache = {
        "force_kernels": (raw + raw.conj().transpose(0, 2, 1)) * 1e-5,
        "required_force_n": rng.normal(size=3) * 1e-5,
        "compliance_observation_m_n": rng.normal(size=(7, 3)) * 1e-3,
        "observation_weights": np.ones(7),
        "observation_offset_m": rng.normal(size=7) * 1e-8,
        "observation_radii_m": np.array([0, 1, 2, 0, 1, 2, 3]) * 1e-3,
        "observation_faces": np.array([0, 0, 0, 1, 1, 1, 1]),
        "carrier_faces": np.array([0, 0, 1, 1, 1]),
        "carrier_height_response_s": (rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4)))
        * 1e-8,
    }
    objective = PrecisionObjective(
        cache,
        0.7,
        1,
        peak_power=peak_power,
        clear_radius_m=1e-3,
        include_carrier_in_peak=include_carrier,
    )
    reduced = WhitenedObjective(objective)
    point = rng.normal(size=2 * reduced.channels)
    physical = reduced.real_transform @ point
    for actual, expected in zip(reduced.metrics(point), objective.metrics(physical), strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    reduced_direction = rng.normal(size=len(point))
    delta = 1e-5 * reduced_direction
    finite_observation = (
        reduced.metrics(point + delta)[0] - reduced.metrics(point - delta)[0]
    ) / 2e-5
    np.testing.assert_allclose(
        reduced.observation_jacobian(point) @ reduced_direction,
        finite_observation,
        atol=1e-9,
        rtol=1e-8,
    )
    x = rng.normal(size=8)
    direction = rng.normal(size=8)
    step = 1e-5
    finite = (
        objective.residual(x + step * direction) - objective.residual(x - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(objective.jacobian(x) @ direction, finite, rtol=1e-8, atol=1e-9)
    if peak_power is not None:
        h, b = objective.metrics(x)
        for j, (m, c) in enumerate(zip(objective.mean_masks, objective.carrier_masks)):
            assert objective.residual(x)[j] >= np.max(abs(h[m])) + (
                np.max(abs(b[c])) if include_carrier else 0
            )
        assert objective.peak_norm(np.zeros(5))[0] == 0
    reduced = WhitenedObjective(objective, cutoff=1e-14)
    y = reduced.encode((x[:4] + 1j * x[4:]) * 3)
    np.testing.assert_allclose(
        reduced.residual(y)[:-4],
        objective.residual(reduced.real_transform @ y),
        rtol=1e-10,
        atol=1e-10,
    )
    direction = rng.normal(size=len(y))
    finite = (reduced.residual(y + step * direction) - reduced.residual(y - step * direction)) / (
        2 * step
    )
    np.testing.assert_allclose(reduced.jacobian(y) @ direction, finite, rtol=1e-7, atol=1e-6)
    finite_slack = (
        reduced.source_slack(y + step * direction) - reduced.source_slack(y - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(
        reduced.source_slack_jacobian(y) @ direction, finite_slack, rtol=1e-8, atol=1e-8
    )
    restricted_cache = dict(cache)
    restricted_cache["carrier_height_response_s"] = np.zeros((5, 4), complex)
    restricted_cache["carrier_height_response_s"][0, 0] = 1e-8
    restricted = WhitenedObjective(
        PrecisionObjective(restricted_cache, 0, 1), carrier_ceiling_m=3e-9
    )
    command = restricted.unpack(rng.normal(size=2 * restricted.channels))
    command *= min(1, restricted.limit / np.max(abs(command)))
    assert np.max(abs(restricted_cache["carrier_height_response_s"] @ command)) <= 3e-9
    assert restricted.carrier_subspace_bound_m <= 3e-9
