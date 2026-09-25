import numpy as np
import pytest

from acoustic_freeform.acoustics.helmholtz import weighted_source_grams


@pytest.mark.parametrize("channels", [1, 7])
def test_gram_contraction_against_direct_quadrature_of_driven_trace(channels):
    rng = np.random.default_rng(917)
    weights = rng.uniform(0.01, 1, 31)
    basis = rng.normal(size=(31, 9))  # constrained shape functions can change sign
    trace = rng.normal(size=(31, channels)) + 1j * rng.normal(size=(31, channels))
    command = rng.normal(size=channels) + 1j * rng.normal(size=channels)
    kernels = weighted_source_grams(weights, basis, trace)
    direct = np.sum((weights * abs(trace @ command) ** 2)[:, None] * basis, axis=0)
    actual = np.einsum("a,iab,b->i", command.conj(), kernels, command)
    np.testing.assert_allclose(actual, direct, atol=2e-12, rtol=1e-12)
    np.testing.assert_allclose(kernels, kernels.conj().transpose(0, 2, 1), atol=2e-14)
    old = np.einsum("s,si,sa,sb->iab", weights, basis, trace.conj(), trace, optimize=True)
    np.testing.assert_allclose(kernels, old, rtol=1e-12, atol=2e-14)
