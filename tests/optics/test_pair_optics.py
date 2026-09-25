import numpy as np

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface
from acoustic_freeform.optics.raytrace import trace_pair


def test_parallel_plate_against_snell_lateral_displacement():
    cfg = DualConfig()
    space = DualSurface(cfg)
    q = np.zeros((2, space.count))
    result = trace_pair(space, q, [1.33, 1.5, 1.33], -0.101, 0.151, 101)
    r = cfg.clear_radius_m * np.sqrt((np.arange(101) + 0.5) / 101)
    theta = np.arctan(r / 0.1)
    inside = np.arcsin(1.33 / 1.5 * np.sin(theta))
    expected = r + 0.002 * np.tan(inside) + 0.15 * np.tan(theta)
    mask = result["transmitted"]
    np.testing.assert_allclose(
        np.linalg.norm(result["spots_m"][mask], axis=1), expected[mask], atol=1e-14
    )


def test_equal_indices_ignore_shape():
    cfg = DualConfig()
    space = DualSurface(cfg)
    q = np.random.default_rng(13).normal(size=(2, space.count)) * 1e-6
    result = trace_pair(space, q, [1.4] * 3, -0.101, 0.151, 101)
    expected = cfg.clear_radius_m * np.sqrt((np.arange(101) + 0.5) / 101) * 0.252 / 0.1
    mask = result["transmitted"]
    np.testing.assert_allclose(
        np.linalg.norm(result["spots_m"][mask], axis=1), expected[mask], atol=1e-13
    )
