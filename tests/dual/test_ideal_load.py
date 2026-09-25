"""Independent spherical Laplace-pressure and hydrostatic limits."""

import numpy as np

from acoustic_freeform.dual.ideal_load_audit import traction_components


def test_spherical_cap_laplace_pressure_including_axis():
    radius = 0.02

    class Cap:
        def evaluate(self, r, derivative=0):
            if derivative == 0:
                return np.sqrt(radius**2 - r**2) - radius
            if derivative == 1:
                return -r / np.sqrt(radius**2 - r**2)
            return -(radius**2) / (radius**2 - r**2) ** 1.5

    r = np.linspace(0, 0.008, 91)
    capillary, gravity = traction_components(Cap(), r, 0.025, 100, 9.81)
    np.testing.assert_allclose(capillary, 2 * 0.025 / radius, rtol=1e-14)
    np.testing.assert_allclose(gravity, 981 * Cap().evaluate(r))
