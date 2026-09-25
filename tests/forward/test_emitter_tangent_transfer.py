import numpy as np
from scipy.linalg import block_diag

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.forward.held_command import project_acoustic_tangent
from acoustic_freeform.mechanics.surface import DualSurface


def test_reference_tangent_transfer_exact_on_common_polynomial():
    old = DualSurface(DualConfig(surface_elements=8))
    new = DualSurface(DualConfig(surface_elements=14))
    original = block_diag(old.mass, old.mass)
    np.testing.assert_allclose(
        project_acoustic_tangent(old, old, original), original, atol=1e-18, rtol=1e-11
    )
    # This pinned, axis-regular quartic has zero radial volume integral.
    x = new.r / new.config.radius_m
    h = (1 - x * x) * (x * x - 1 / 3)
    q = np.linalg.solve(new.mass, new.b.T @ (new.weights * h))
    pair = np.r_[q, -0.7 * q]
    projected = project_acoustic_tangent(old, new, original)
    exact = block_diag(new.mass, new.mass)
    np.testing.assert_allclose(projected @ pair, exact @ pair, atol=1e-18, rtol=1e-11)
