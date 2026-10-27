"""Load demand against closed-form geometry (theory section 10f)."""

import numpy as np
import pytest

from acoustic_freeform.mechanics.demand import demanded_traction, sag_series, series_demand


class Conic:
    """Conic sag h = c r^2 / (1 + sqrt(1 - (1 + k) c^2 r^2)) with analytic derivatives."""

    def __init__(self, c, k):
        self.c, self.k = c, k

    def evaluate(self, r, derivative=0):
        r = np.asarray(r, float)
        c, s = self.c, 1 + self.k
        root = np.sqrt(1 - s * c**2 * r**2)
        return [c * r**2 / (1 + root), c * r / root, c / root**3][derivative]


def test_sphere_needs_uniform_traction():
    # A sphere of radius R has mean curvature 2/R: the demand is the constant -2 sigma / R.
    radius, sigma = 3e-3, 0.025
    r = np.linspace(1e-5, 1e-3, 50)
    traction = demanded_traction(Conic(1 / radius, 0.0), sigma, 0.0, r)
    np.testing.assert_allclose(traction, -2 * sigma / radius, rtol=1e-12)


def test_series_matches_exact_conic_demand():
    # Q2 and Q4 from the series must equal the r^2, r^4 coefficients of the exact demand.
    c, k, sigma, drho_g = 250.0, -0.6, 0.025, 30.0
    conic = Conic(c, k)
    r = np.linspace(1e-6, 2e-4, 400)
    t = demanded_traction(conic, sigma, drho_g, r)
    x = r / r[-1]
    fit = np.polynomial.polynomial.polyfit(x**2, t, 4)
    q2_exact, q4_exact = fit[1] / r[-1] ** 2, fit[2] / r[-1] ** 4
    series = series_demand(c, (1 + k) * c**3 / 8, (1 + k) ** 2 * c**5 / 16, sigma, drho_g)
    assert series["Q2_pa_m2"] == pytest.approx(q2_exact, rel=1e-6)
    assert series["Q4_pa_m4"] == pytest.approx(q4_exact, rel=1e-4)


def test_conic_matching_rule_removes_leading_demand():
    # drho g / sigma = 4 k c^2 cancels the r^2 term of a conic face: what is left of
    # the exact demand is the fourth-order term Q4 r^4 alone.
    c, k, sigma = 300.0, -0.4, 0.03
    drho_g = 4 * k * c**2 * sigma
    conic = Conic(c, k)
    r = np.linspace(1e-6, 1e-4, 200)
    matched = demanded_traction(conic, sigma, drho_g, r)
    q4 = series_demand(c, (1 + k) * c**3 / 8, (1 + k) ** 2 * c**5 / 16, sigma, drho_g)["Q4_pa_m4"]
    np.testing.assert_allclose(matched - matched[0], q4 * (r**4 - r[0] ** 4), rtol=1e-3, atol=1e-12)
    unmatched = demanded_traction(conic, sigma, 0.0, r)
    assert np.ptp(unmatched) > 1e3 * np.ptp(matched)


def test_sag_series_recovers_conic_coefficients():
    c, k = 200.0, -0.8
    fit = sag_series(Conic(c, k), radius_m=2e-3)
    assert fit["c"] == pytest.approx(c, rel=1e-8)
    assert fit["a4"] == pytest.approx((1 + k) * c**3 / 8, rel=1e-5)
