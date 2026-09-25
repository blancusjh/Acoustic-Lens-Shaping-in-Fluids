import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import (
    CartesianPatch,
    cartesian_derivatives,
    cartesian_taylor,
)
from acoustic_freeform.optics.cartesian import CartesianDiopter


@pytest.mark.parametrize("object_z,image_z", [(-0.1, 0.2), (-np.inf, 0.2), (-0.1, np.inf)])
def test_fermat_taylor_jet_matches_exact_branch(object_z, image_z):
    diopter = CartesianDiopter(1.33, object_z, 1.5, image_z)
    a, length = 0.002, 0.002
    coeff = cartesian_taylor(diopter, a, length, 4)
    h, first, second = cartesian_derivatives(diopter, np.array([a]))
    np.testing.assert_allclose(
        coeff[:3], [h[0], first[0] * length, second[0] * length**2 / 2], rtol=1e-11, atol=1e-15
    )
    t = np.array([0.1, 0.05, 0.025])
    approx = np.polynomial.polynomial.polyval(t, coeff)
    np.testing.assert_allclose(approx, diopter.sag(a + length * t), atol=2e-10, rtol=0)


def test_c4_annulus_has_fifth_order_optical_path_contact_and_preserves_fill():
    cfg = DualConfig()
    optical = {"n_o": 1.33, "z_o_m": -0.09985, "n_i": 1.5, "z_i_m": 0.20215}
    patch = CartesianPatch(cfg, 0, optical, -0.00015, annulus_match_order=4)
    t = np.array([0.1, 0.05, 0.025])
    r = cfg.clear_radius_m + (cfg.radius_m - cfg.clear_radius_m) * t
    path_error = abs(patch.diopter.fermat(r, patch.evaluate(r) - patch.vertex))
    assert np.all(path_error[1:] < 0.06 * path_error[:-1])
    assert abs(patch.evaluate(np.array([cfg.radius_m]))[0]) < 2e-13
    x, w = leggauss(128)
    volume = 0.0
    for lo, hi in [(0, cfg.clear_radius_m), (cfg.clear_radius_m, cfg.radius_m)]:
        r = lo + (hi - lo) * (x + 1) / 2
        volume += np.dot(w * (hi - lo) / 2, 2 * np.pi * r * patch.evaluate(r))
    assert abs(volume) < 1e-18
