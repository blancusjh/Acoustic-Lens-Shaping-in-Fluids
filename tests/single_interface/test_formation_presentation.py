"""Independent geometry checks for the read-only formation presentation."""

import numpy as np

from acoustic_freeform.single_interface.config import LensConfig
from acoustic_freeform.single_interface.formation_presentation import (
    exact_height_error,
    height_maximum,
)
from acoustic_freeform.single_interface.surface import SurfaceSpace


def test_exact_cartesian_rays_intersect_fixed_detector():
    cfg = LensConfig(object_distance_m=-.05, surface_modes=16)
    r = np.linspace(0, cfg.clear_radius_m, 503)
    z = cfg.diopter.sag(r)
    fr, fz = cfg.diopter.fermat_gradient(r, z)
    direction = cfg.diopter.refract(r, z, -fr/fz)
    hits = r + (cfg.focal_distance_m-z)*direction[:, 0]/direction[:, 1]
    assert np.max(abs(hits)) < 1e-11


def test_height_search_checks_exact_target_not_projected_target():
    space = SurfaceSpace(LensConfig(object_distance_m=-.05, surface_modes=12))
    c, _ = space.target()
    # Independent very dense sampling cross-checks the derivative-root search.
    c = c.copy()
    c[3] += .002
    r = np.linspace(0, space.config.clear_radius_m, 60001)
    dense = np.max(abs(exact_height_error(space, c, r)))
    found = height_maximum(space, c)
    np.testing.assert_allclose(found, dense, rtol=1e-7, atol=1e-12)
