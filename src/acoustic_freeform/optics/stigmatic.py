"""Joint Cartesian prescription using laboratory, not mixed vertex, conjugates.

This constructs optical targets only. It is not an acoustic inverse or a
forward trajectory. Existing CartesianPatch supplies the pinned, fixed-volume
non-optical annuli and includes gravity in their mechanical load objective.
"""

import numpy as np

from ..mechanics.surface import CartesianPatch


def pair_prescription(config, indices, stigmatic_z_m, vertex_displacement_m):
    """Return campaign-compatible faces sharing exactly the same middle focus."""
    indices = np.asarray(indices, float)
    z = np.asarray(stigmatic_z_m, float)
    displacement = np.asarray(vertex_displacement_m, float)
    if indices.shape != (3,) or z.shape != (3,) or displacement.shape != (2,):
        raise ValueError("Need three indices, three lab conjugates and two vertex displacements")
    if np.any(~np.isfinite(np.r_[indices, z, displacement])) or np.any(indices <= 0):
        raise ValueError("This prescription requires finite conjugates and positive indices")
    if not z[0] < config.levels_m[0] < config.levels_m[-1] < z[2]:
        raise ValueError("Object and final image must bracket the chamber")
    vertices = np.asarray(config.levels_m[1:3]) + displacement
    if not config.levels_m[0] < vertices[0] < vertices[1] < config.levels_m[-1]:
        raise ValueError("Vertices must be ordered inside the chamber")
    if np.any(z[1] == vertices):
        raise ValueError("The intermediate conjugate cannot coincide with a vertex")
    return [
        {
            "optical": {
                "n_o": float(indices[j]),
                "z_o_m": float(z[j] - vertices[j]),
                "n_i": float(indices[j + 1]),
                "z_i_m": float(z[j + 1] - vertices[j]),
            },
            "vertex_displacement_m": float(displacement[j]),
        }
        for j in range(2)
    ]


def stigmatic_pair(config, indices, stigmatic_z_m, vertex_displacement_m):
    faces = pair_prescription(config, indices, stigmatic_z_m, vertex_displacement_m)
    return [CartesianPatch(config, j, **face) for j, face in enumerate(faces)]


def intermediate_point_from_shape_factor(n0, n1, d0, d2, t, sigma):
    """Intermediate stigmatic point d1 of a singlet from its shape factor sigma.

    Silva-Lora and Torres, JOSA A 37, 1155 (2020), Eqs. (44), (A1)-(A5), for
    n2 = n0; the first vertex is at the origin and the second at t. Of the two
    roots, the one outside the lens (d1 not in [0, t]) is the paper's design
    point; it reproduces their Fig. 7 edge thicknesses.
    """
    c2 = n0 * (n1 - n0) * ((sigma - 1) / d0 - (sigma + 1) / (d2 - t))
    c1 = n0 * (sigma - 1) * (d2 * n1 / (d2 - t) - t * (n1 - n0) / d0) + (
        2 * n1 - n0 * (sigma + 1)
    ) * (n0 * t / (d2 - t) + n1)
    c0 = n1 * t * (sigma - 1) * (n1 - n0)
    disc = np.sqrt(c1**2 - 4 * c2 * c0)
    roots = [(-c1 - disc) / (2 * c2), (-c1 + disc) / (2 * c2)]
    outside = [d for d in roots if not 0 <= d <= t]
    if not outside:
        raise ValueError("No intermediate point outside the lens for this shape factor")
    return float(outside[0])
