"""Kirchhoff-Helmholtz listener against an exact radiating field."""

import numpy as np
import pytest

from acoustic_freeform.acoustics.radiation import boundary_edges, listener_pressure


def meridional_mesh(radius, bottom, top, nr, nz):
    r, z = np.meshgrid(np.linspace(0, radius, nr), np.linspace(bottom, top, nz), indexing="ij")
    index = np.arange(nr * nz).reshape(nr, nz)
    a, b, c, d = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    triangles = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
    return np.vstack([r.ravel(), z.ravel()]), triangles


def monopole(points, source_z, k):
    distance = np.hypot(points[0], points[1] - source_z)
    return np.exp(1j * k * distance) / (4 * np.pi * distance)


@pytest.mark.parametrize("listener_z", [0.02, 0.05])
def test_point_source_inside_the_cell_is_reproduced_outside(listener_z):
    # A monopole inside the closed surface: the exterior representation is exact,
    # so the listener must hear the monopole itself (to the discretization error).
    k = 2 * np.pi * 1.5e6 / 1480.0
    xy, triangles = meridional_mesh(2.02e-3, -0.8e-3, 2.23e-3, 161, 241)
    source_z = 0.5e-3
    heard = listener_pressure(xy, triangles, monopole(xy, source_z, k), k, listener_z)
    exact = monopole(np.array([[0.0], [listener_z]]), source_z, k)[0]
    assert abs(heard - exact) < 0.005 * abs(exact)


def test_boundary_of_a_square_mesh():
    _xy, triangles = meridional_mesh(1.0, 0.0, 1.0, 4, 4)
    edges, owner = boundary_edges(triangles)
    assert len(edges) == 4 * 3  # three edges per side of a 4 x 4 node grid
    np.testing.assert_array_equal(np.sort(edges, axis=1), edges)
    assert np.all(np.isin(edges, triangles[owner]).all(axis=1))


def test_error_falls_at_second_order():
    k = 2 * np.pi * 1.5e6 / 1480.0
    errors = []
    for n in (41, 81):
        xy, triangles = meridional_mesh(2.02e-3, -0.8e-3, 2.23e-3, n, int(1.5 * n))
        heard = listener_pressure(xy, triangles, monopole(xy, 0.5e-3, k), k, 0.05)
        exact = monopole(np.array([[0.0], [0.05]]), 0.5e-3, k)[0]
        errors.append(abs(heard - exact) / abs(exact))
    assert errors[1] < errors[0] / 3.5
