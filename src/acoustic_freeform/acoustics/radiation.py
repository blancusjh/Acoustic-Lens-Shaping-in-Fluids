"""Sound radiated by an axisymmetric field to a listener on the axis.

Kirchhoff-Helmholtz representation of the field outside a closed surface S
that encloses all sources, in a homogeneous exterior medium (exp(-i omega t)):

    p(x) = integral over S of [p dG/dn - G dp/dn] dS,   G = exp(ikR) / (4 pi R),

with n the outward normal and R = |x - y|. The surface is the revolution of the
outer boundary of a meridional (r, z) triangulation. For x on the axis, R and
n.(x - y) do not depend on the azimuth, so the azimuthal integral is exact and
dS = 2 pi r ds. p and dp/dn at each boundary-edge midpoint come from a local
quadratic least-squares fit to the nearest nodal values (second-order accurate;
tests/test_radiation.py checks it against an exact monopole).
"""

import numpy as np


def boundary_edges(triangles):
    """Edges used by one triangle only, and that triangle's index."""
    edges = np.sort(np.r_[triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1)
    owner = np.tile(np.arange(len(triangles)), 3)
    unique, index, counts = np.unique(edges, axis=0, return_index=True, return_counts=True)
    return unique[counts == 1], owner[index[counts == 1]]


def listener_pressure(xy, triangles, pressure, wavenumber, z_listener):
    """Kirchhoff-Helmholtz pressure on the axis at z_listener, from the axisymmetric boundary data."""
    edges, owner = boundary_edges(triangles)
    r, z = xy
    a, b = edges[:, 0], edges[:, 1]
    mid_r, mid_z = (r[a] + r[b]) / 2, (z[a] + z[b]) / 2
    keep = mid_r > 1e-9 * r.max()  # the axis is not a boundary of the revolved cell
    keep &= ~((np.abs(r[a]) < 1e-12) & (np.abs(r[b]) < 1e-12))
    a, b, owner, mid_r, mid_z = a[keep], b[keep], owner[keep], mid_r[keep], mid_z[keep]
    tangent = np.c_[r[b] - r[a], z[b] - z[a]]
    length = np.linalg.norm(tangent, axis=1)
    normal = np.c_[tangent[:, 1], -tangent[:, 0]] / length[:, None]
    # Orient outward: away from the triangle's centroid.
    centroid = np.c_[r[triangles[owner]].mean(1), z[triangles[owner]].mean(1)]
    flip = np.einsum("ij,ij->i", normal, np.c_[mid_r, mid_z] - centroid) < 0
    normal[flip] *= -1
    # p and grad p at each edge midpoint from a local quadratic least-squares fit
    # over the nearest nodes (second order; a single boundary triangle is first order).
    p_mid, grad = _quadratic_fit(r, z, pressure, mid_r, mid_z, length)
    dpdn = np.einsum("ij,ij->i", grad, normal)
    dz = z_listener - mid_z
    distance = np.hypot(mid_r, dz)
    green = np.exp(1j * wavenumber * distance) / (4 * np.pi * distance)
    # dG/dn_y = G (ik - 1/R) dR/dn_y with dR/dn_y = -n.(x - y)/R and n.(x - y) = -r n_r + n_z dz.
    dRdn = -(-mid_r * normal[:, 0] + dz * normal[:, 1]) / distance
    dGdn = green * (1j * wavenumber - 1 / distance) * dRdn
    area = 2 * np.pi * mid_r * length
    return np.sum((p_mid * dGdn - green * dpdn) * area)


def _quadratic_fit(r, z, values, at_r, at_z, scale, neighbours=12):
    from scipy.spatial import cKDTree

    _, index = cKDTree(np.c_[r, z]).query(np.c_[at_r, at_z], k=neighbours)
    dr = (r[index] - at_r[:, None]) / scale[:, None]
    dz = (z[index] - at_z[:, None]) / scale[:, None]
    basis = np.stack([np.ones_like(dr), dr, dz, dr**2, dr * dz, dz**2], axis=-1)  # (n, k, 6)
    coefficients = np.array([np.linalg.lstsq(B, values[i], rcond=None)[0] for B, i in zip(basis, index)])
    return coefficients[:, 0], coefficients[:, 1:3] / scale[:, None]
