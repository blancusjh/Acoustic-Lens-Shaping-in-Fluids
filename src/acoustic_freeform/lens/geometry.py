"""Quadratic mesh of the actual finite liquid chamber, fitted to the free surface."""

import numpy as np
from skfem import MeshTri, MeshTri2

from .config import LensConfig
from .surface import SurfaceSpace


def chamber_mesh(config: LensConfig, surface: SurfaceSpace, coefficients):
    depth = config.depth_m / config.radius_m
    radial = np.linspace(0, 1, config.mesh_radial + 1)
    vertical = np.linspace(-depth, 0, config.mesh_vertical + 1)
    flat = MeshTri.init_tensor(radial, vertical)
    boundaries = {
        "axis": flat.facets_satisfying(lambda x: np.isclose(x[0], 0)),
        "wall": flat.facets_satisfying(lambda x: np.isclose(x[0], 1)),
        "base": flat.facets_satisfying(lambda x: np.isclose(x[1], -depth)),
        "surface": flat.facets_satisfying(lambda x: np.isclose(x[1], 0)),
    }
    quadratic = MeshTri2.from_mesh(flat)
    points = quadratic.p.copy()
    h = surface.evaluate(coefficients, points[0])
    points[1] += (points[1] + depth) / depth * h
    return MeshTri2(points, quadratic.t, _boundaries=boundaries)
