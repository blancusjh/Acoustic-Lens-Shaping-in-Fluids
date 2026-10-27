"""The convex capability bound on small problems with known answers."""

from types import SimpleNamespace

import numpy as np
import pytest

from acoustic_freeform.inverse.capability import best_tolerance

pytest.importorskip("cvxpy")


def tone(kernels, power):
    return SimpleNamespace(frequency_hz=1e6, kernels=np.asarray(kernels, complex), power=np.asarray(power, complex),
                           pressure=np.eye(len(power), dtype=complex), kernel_derivatives=None)


def test_reachable_demand_has_zero_bound():
    # The demand is the force of a known single tone: the bound must vanish.
    rng = np.random.default_rng(1)
    m, n = 4, 3
    a = rng.normal(size=(n, m, m)) + 1j * rng.normal(size=(n, m, m))
    kernels = (a + np.conj(np.swapaxes(a, 1, 2))) / 2
    c = rng.normal(size=m) + 1j * rng.normal(size=m)
    demand = np.real(np.einsum("i,nij,j->n", c.conj(), kernels, c))
    observe = 1e-9 * np.eye(n)
    result = best_tolerance([tone(kernels, np.eye(m))], observe, demand, power_cap_w=10 * np.vdot(c, c).real)
    assert result["pupil_max_error_m"] < 1e-14


def test_unreachable_demand_bound_is_exact():
    # Positive semidefinite kernels can only push forces up. A demand of (-1, 0) is
    # then at best matched by F = 0, with error max(|0 + 1|, 0) = 1 in observation units.
    kernels = np.array([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])])
    observe = 1e-9 * np.eye(2)
    result = best_tolerance([tone(kernels, np.eye(2))], observe, np.array([-1.0, 0.0]), power_cap_w=5.0)
    assert result["pupil_max_error_m"] == pytest.approx(1e-9, rel=1e-4)


def test_power_cap_limits_reach():
    # One kernel, force |c|^2, power |c|^2: with a cap of 0.25 the best force is 0.25.
    observe = 1e-9 * np.eye(1)
    result = best_tolerance([tone(np.eye(1)[None], np.eye(1))], observe, np.array([1.0]), power_cap_w=0.25)
    assert result["pupil_max_error_m"] == pytest.approx(0.75e-9, rel=1e-4)
