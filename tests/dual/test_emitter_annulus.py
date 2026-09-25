import numpy as np

from acoustic_freeform.dual.config import DualConfig
from acoustic_freeform.dual.emitter_free_annulus import annulus_spaces
from acoustic_freeform.dual.surface import DualSurface


def test_annulus_variations_leave_both_optical_profiles_unchanged():
    cfg = DualConfig(surface_elements=24)
    space = DualSurface(cfg)
    annulus, complement = annulus_spaces(space)
    assert annulus.shape[1] > 0
    rng = np.random.default_rng(231)
    perturbation = (annulus @ rng.normal(size=annulus.shape[1])).reshape(2, -1)
    independent_radii = rng.uniform(0, cfg.clear_radius_m, 1001)
    for derivative in (0, 1, 2):
        values = space.basis(independent_radii, derivative) @ perturbation.T
        np.testing.assert_allclose(values * cfg.radius_m**derivative, 0, atol=2e-11)
    assert (
        np.max(
            abs(space.basis(np.linspace(cfg.clear_radius_m, cfg.radius_m, 107)) @ perturbation.T)
        )
        > 0.01
    )
    np.testing.assert_allclose(complement.T @ annulus, 0, atol=2e-14)
    np.testing.assert_allclose(
        annulus @ annulus.T + complement @ complement.T, np.eye(2 * space.count), atol=2e-14
    )
