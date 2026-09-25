"""An independent scalar optimum for the optional semidefinite diagnostic."""

import numpy as np
import pytest

pytest.importorskip("cvxpy")

from acoustic_freeform.inverse.relaxation import solve_relaxation


def test_scalar_exact_force_fixes_carrier_minimum():
    # |a|^2 = 1 and b = 20 nm * a force exactly a 20 nm carrier amplitude.
    result, matrix = solve_relaxation(
        np.ones((1, 1, 1), complex), np.ones(1), np.array([[2e-8]], complex), np.ones((1, 1)), 2
    )
    assert result["status"] == "optimal"
    np.testing.assert_allclose(matrix, [[1]], rtol=1e-5)
    assert abs(result["sampled_relaxed_carrier_peak_m"] - 2e-8) < 1e-12
