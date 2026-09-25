from types import SimpleNamespace

import numpy as np

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface
from acoustic_freeform.verify.metrics import maximum_error


def test_independent_quartic_interior_maximum():
    cfg = DualConfig(radius_m=0.004, clear_radius_m=0.0036)
    space = DualSurface(cfg)
    amplitude = 1e-5

    def height(r):
        x = r / cfg.radius_m
        return amplitude * (1 - x**2) * (x**2 - 1 / 3)

    q = np.linalg.solve(space.mass, space.b.T @ (space.weights * height(space.r)))
    target = SimpleNamespace(
        evaluate=lambda r, derivative=0: np.full_like(r, -amplitude / 3 if derivative == 0 else 0)
    )
    result = maximum_error(space, q, target)
    # Exact extremum is r/R=sqrt(2/3), not one of the sampling radii.
    np.testing.assert_allclose(result["max_error_m"], 4 * amplitude / 9, rtol=1e-12)
    assert result["rigorous_interval_bound"] is False


def test_endpoint_maximum_and_zero_error():
    cfg = SimpleNamespace(clear_radius_m=0.002)
    space = SimpleNamespace(
        config=cfg,
        evaluate=lambda q, r, derivative=0: q * (r if derivative == 0 else np.ones_like(r)),
    )
    target = SimpleNamespace(evaluate=lambda r, derivative=0: np.zeros_like(r))
    assert maximum_error(space, 2e-5, target)["max_error_m"] == 4e-8
    assert maximum_error(space, 0, target)["max_error_m"] == 0
