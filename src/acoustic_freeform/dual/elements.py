"""Equispaced triangular Lagrange elements for independent p-refinement."""

import numpy as np
from skfem.element import ElementH1
from skfem.refdom import RefTri


class ElementTriLagrange(ElementH1):
    refdom = RefTri
    nodal_dofs = 1

    def __init__(self, degree):
        if degree not in range(2, 7):
            raise ValueError("Verified degree range is 2 through 6.")
        self.maxdeg = degree
        self.facet_dofs = degree - 1
        self.interior_dofs = (degree - 1) * (degree - 2) // 2
        self.dofnames = ["u"] * (1 + self.facet_dofs + self.interior_dofs)
        p = degree
        triples = [(p, 0, 0), (0, p, 0), (0, 0, p)]
        triples += [(p - j, j, 0) for j in range(1, p)]
        triples += [(0, p - k, k) for k in range(1, p)]
        triples += [(p - k, 0, k) for k in range(1, p)]
        triples += [(p - j - k, j, k) for k in range(1, p - 1) for j in range(1, p - k)]
        self.triples = triples
        self.doflocs = np.asarray([(j / p, k / p) for _, j, k in triples])

    def falling(self, coordinate, count):
        value, derivative = np.ones_like(coordinate), np.zeros_like(coordinate)
        for index in range(count):
            factor = (self.maxdeg * coordinate - index) / (index + 1)
            derivative = derivative * factor + value * self.maxdeg / (index + 1)
            value = value * factor
        return value, derivative

    def lbasis(self, X, i):
        x, y = X
        if not 0 <= i < len(self.triples):
            self._index_error()
        a, b, c = [self.falling(t, n) for t, n in zip((1 - x - y, x, y), self.triples[i])]
        value = a[0] * b[0] * c[0]
        derivative = np.array(
            [
                (-a[1] * b[0] + a[0] * b[1]) * c[0],
                (-a[1] * c[0] + a[0] * c[1]) * b[0],
            ]
        )
        return value, derivative
