"""Exact graph-fitted coordinate mapping for the finite liquid chamber.

The reference chamber coordinates (r,z) map to
(r, z + (1+z/depth)*h(r)).  All metrics and boundary normals follow from
that same map.  No polynomial interpolation of the surface geometry enters
FEM integration; solution fields still have their declared finite-element order.
"""

import numpy as np
from skfem import MeshTri2
from skfem.mapping import MappingAffine


class GraphMapping(MappingAffine):
    def __init__(self, flat, surface, coefficients):
        super().__init__(flat)
        self.surface = surface
        self.coefficients = np.asarray(coefficients).copy()
        self.depth = surface.config.depth_m / surface.config.radius_m
        self._geometry_cache = {}

    def warp(self, points):
        result = np.array(points, copy=True)
        h = self.surface.evaluate(self.coefficients, points[0])
        stretch = 1 + h / self.depth
        if np.any(stretch <= 0):
            raise ValueError("The interface crosses the chamber base.")
        result[1] += (1 + points[1] / self.depth) * h
        return result

    def unwarp(self, points):
        result = np.array(points, copy=True)
        h = self.surface.evaluate(self.coefficients, points[0])
        result[1] = (points[1] - h) / (1 + h / self.depth)
        return result

    def _geometry(self, X, tind=None):
        # Content keys avoid stale array-id reuse and retain only a few quadratures.
        key = (X.shape, X.tobytes(), None if tind is None else np.asarray(tind).tobytes())
        if key not in self._geometry_cache:
            points = super().F(X, tind)
            h = self.surface.evaluate(self.coefficients, points[0])
            hp = self.surface.evaluate(self.coefficients, points[0], 1)
            stretch = 1 + h / self.depth
            if np.any(stretch <= 0):
                raise ValueError("Nonpositive graph-map Jacobian.")
            shear = (1 + points[1] / self.depth) * hp
            transformed = points.copy()
            transformed[1] += (1 + points[1] / self.depth) * h
            if len(self._geometry_cache) >= 12:
                self._geometry_cache.clear()
            self._geometry_cache[key] = transformed, stretch, shear
        return self._geometry_cache[key]

    def F(self, X, tind=None):
        return self._geometry(X, tind)[0]

    def invF(self, x, tind=None):
        return super().invF(self.unwarp(x), tind)

    def DF(self, X, tind=None):
        _, stretch, shear = self._geometry(X, tind)
        base = super().DF(X, tind)
        result = base.copy()
        result[1] = shear[None] * base[0] + stretch[None] * base[1]
        return result

    def detDF(self, X, tind=None):
        return super().detDF(X, tind) * self._geometry(X, tind)[1]

    def invDF(self, X, tind=None):
        _, stretch, shear = self._geometry(X, tind)
        base = super().invDF(X, tind)
        result = base.copy()
        result[:, 0] -= base[:, 1] * shear[None] / stretch[None]
        result[:, 1] /= stretch[None]
        return result

    def G(self, X, find=None):
        return self.warp(super().G(X, find))

    def detDG(self, X, find=None):
        points = super().G(X, find)
        h = self.surface.evaluate(self.coefficients, points[0])
        hp = self.surface.evaluate(self.coefficients, points[0], 1)
        tangent = self.B[:, 0] if find is None else self.B[:, 0, find]
        dr = np.broadcast_to(tangent[0, :, None], points[0].shape)
        dz = (1 + points[1] / self.depth) * hp * dr
        dz += (1 + h / self.depth) * tangent[1, :, None]
        return np.sqrt(dr * dr + dz * dz)


class GraphMesh(MeshTri2):
    """Display nodes fitted to a cap, with exact metrics for every FE basis."""

    def __init__(self, flat, surface, coefficients, boundaries):
        mapping = GraphMapping(flat, surface, coefficients)
        quadratic = MeshTri2.from_mesh(flat)
        super().__init__(mapping.warp(quadratic.p), quadratic.t, _boundaries=boundaries)
        self._cached_mapping = mapping
