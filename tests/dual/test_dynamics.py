import numpy as np
from numpy.polynomial.legendre import leggauss

from acoustic_freeform.dual.dynamics import layer_mass, midpoint_step


def test_layer_mass_against_volume_kinetic_energy():
    # Integrate the independently constructed velocity potential in the layer.
    k, h, rho, u, v = 800.0, 0.002, 1100.0, 0.013, -0.008
    x, w = leggauss(80)
    z, dz = h * (x + 1) / 2, h * w / 2
    a = (v - u * np.cosh(k * h)) / (k * np.sinh(k * h))
    potential = a * np.cosh(k * z) + u / k * np.sinh(k * z)
    vertical = k * a * np.sinh(k * z) + u * np.cosh(k * z)
    energy = 0.5 * rho * np.dot(dz, k * k * potential**2 + vertical**2)
    boundary = np.array([u, v])
    np.testing.assert_allclose(
        0.5 * boundary @ layer_mass(k, h, rho) @ boundary, energy, rtol=1e-12
    )
    assert np.linalg.eigvalsh(layer_mass(k, h, rho)).min() > 0


def test_deep_layers_decouple():
    np.testing.assert_allclose(layer_mass(1000.0, 0.1, 1200.0), np.eye(2) * 1.2, atol=1e-40)


def test_unforced_midpoint_conserves_oscillator_energy():
    def mechanics(q):
        return 7 * q, 7 * np.eye(2), 3.5 * q @ q

    q, v = np.array([0.2, -0.1]), np.array([0.3, 0.4])
    energy = 0.5 * v @ v + mechanics(q)[2]
    for _ in range(100):
        q, v = midpoint_step(q, v, 0.03, np.zeros(2), mechanics)
    np.testing.assert_allclose(0.5 * v @ v + mechanics(q)[2], energy, rtol=1e-12)
