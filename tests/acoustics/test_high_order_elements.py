import numpy as np
import pytest
from skfem import Basis, ElementTriP4, InteriorFacetBasis, MeshTri

from acoustic_freeform.acoustics.elements import ElementTriLagrange


@pytest.mark.parametrize("degree", [4, 5, 6])
def test_complete_polynomial_reproduction_and_shared_traces(degree):
    element = ElementTriLagrange(degree)
    nodal = np.array([element.lbasis(element.doflocs.T, i)[0] for i in range(len(element.doflocs))])
    np.testing.assert_allclose(nodal, np.eye(len(nodal)), atol=1e-12)
    points = np.array([[0.13, 0.41, 0.1], [0.22, 0.21, 0.8]])
    values = np.array([element.lbasis(points, i)[0] for i in range(len(nodal))])
    gradients = np.array([element.lbasis(points, i)[1] for i in range(len(nodal))])
    for j in range(degree + 1):
        for k in range(degree + 1 - j):
            coefficients = element.doflocs[:, 0] ** j * element.doflocs[:, 1] ** k
            np.testing.assert_allclose(
                coefficients @ values, points[0] ** j * points[1] ** k, atol=2e-12
            )
            expected = np.array(
                [
                    j * points[0] ** max(j - 1, 0) * points[1] ** k,
                    k * points[0] ** j * points[1] ** max(k - 1, 0),
                ]
            )
            np.testing.assert_allclose(
                np.einsum("i,ijk->jk", coefficients, gradients), expected, atol=2e-11
            )
    if degree == 4:
        reference = ElementTriP4()
        np.testing.assert_allclose(element.doflocs, reference.doflocs)
        for i in range(len(nodal)):
            a, da = element.lbasis(points, i)
            b, db = reference.lbasis(points, i)
            np.testing.assert_allclose(a, b, atol=1e-13)
            np.testing.assert_allclose(da, db, atol=1e-12)
    mesh = MeshTri.init_tensor(np.linspace(0, 1, 4), np.linspace(0, 1, 5))
    basis = Basis(mesh, element)
    coefficients = np.random.default_rng(515).normal(size=basis.N)
    traces = [
        InteriorFacetBasis(mesh, element, side=side).interpolate(coefficients) for side in (0, 1)
    ]
    np.testing.assert_allclose(traces[0], traces[1], atol=1e-11)
