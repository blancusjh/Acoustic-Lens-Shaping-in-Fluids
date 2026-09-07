"""Finite chamber meshes with declared geometry and source-patch resolution."""

import numpy as np
from skfem import MeshTri, MeshTri2

from .config import LensConfig
from .surface import SurfaceSpace


def chamber_mesh(config: LensConfig, surface: SurfaceSpace, coefficients):
    depth = config.depth_m / config.radius_m
    radial = np.linspace(0, 1, config.mesh_radial + 1)
    if config.mesh_radial_distribution == "rim_clustered":
        radial = np.sin(0.5 * np.pi * radial)
    vertical = np.linspace(-depth, 0, config.mesh_vertical + 1)
    if config.align_array_mesh:
        pitch = depth / config.array_rows
        centers = -depth + (np.arange(config.array_rows) + 0.5) * pitch
        half = pitch * config.element_fill / 2
        vertical = np.unique(np.r_[vertical, centers - half, centers + half])
    flat = MeshTri.init_tensor(radial, vertical)
    boundaries = {
        "axis": flat.facets_satisfying(lambda x: np.isclose(x[0], 0)),
        "wall": flat.facets_satisfying(lambda x: np.isclose(x[0], 1)),
        "base": flat.facets_satisfying(lambda x: np.isclose(x[1], -depth)),
        "surface": flat.facets_satisfying(lambda x: np.isclose(x[1], 0)),
    }
    if config.geometry_mapping == "exact_graph":
        from .mapping import GraphMesh

        return GraphMesh(flat, surface, coefficients, boundaries)
    quadratic = MeshTri2.from_mesh(flat)
    points = quadratic.p.copy()
    h = surface.evaluate(coefficients, points[0])
    points[1] += (points[1] + depth) / depth * h
    return MeshTri2(points, quadratic.t, _boundaries=boundaries)
