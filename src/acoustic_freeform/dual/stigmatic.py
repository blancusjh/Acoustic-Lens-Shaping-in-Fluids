"""Joint Cartesian prescription using laboratory, not mixed vertex, conjugates.

This constructs optical targets only. It is not an acoustic inverse or a
forward trajectory. Existing CartesianPatch supplies the pinned, fixed-volume
non-optical annuli and includes gravity in their mechanical load objective.
"""

import numpy as np

from .surface import CartesianPatch


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
