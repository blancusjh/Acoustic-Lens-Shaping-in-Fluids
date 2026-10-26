"""Aplanatism of a stigmatic two-face lens: generalized Abbe sine condition.

Silva-Lora and Torres, "Aplanatism in stigmatic optical systems" (JOSA A
preprint, 2020): a system of Cartesian surfaces is aplanatic when, for every
ray, M(rho) = prod_k (n_{k+1}/n_k) (1/(d_{k+1}-zeta_k) - 1/V_kC_k) /
(1/(d_k-zeta_k) - 1/V_kC_k) equals one (their Eqs. 15 and 19), where V_kC_k is
the signed distance from the vertex to the axis crossing of the surface normal
at the ray's hit point. The same condition is the Abbe sine condition
n_0 sin u_0 = m n_N sin u_N with constant m; both are evaluated here from an
independent ray trace so that each checks the other.
"""

import numpy as np

from .raytrace import trace_graph_pair


def abbe_sine_condition(cfg, surface, indices, conjugates_m, vertices_m, launch_radius_m, launch_height_m):
    """Per-ray Abbe ratio and the paper's M, both normalized to their axial limit.

    ``surface(face, r, derivative)`` gives the height of each face relative to
    its pinned level; ``conjugates_m`` are the laboratory object, shared and
    image points, ``vertices_m`` the laboratory vertex heights.
    """
    z0, z1, z2 = conjugates_m
    trace = trace_graph_pair(cfg, surface, indices, z0, z2, rays=None,
                             launch_radius_m=launch_radius_m, launch_height_m=launch_height_m)
    hits = trace["intersections_m"]
    first = hits[0] - np.array([0.0, z0])
    sin_in = first[:, 0] / np.linalg.norm(first, axis=1)
    out = trace["final_directions"]
    sin_out = out[:, 0] / np.linalg.norm(out, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        abbe = indices[0] * sin_in / (indices[2] * sin_out)
    conjugate = [(z0, z1), (z1, z2)]
    product = np.ones(len(sin_in))
    for k in range(2):
        r, z = hits[k][:, 0], hits[k][:, 1]
        slope = surface(k, np.abs(r), 1) * np.sign(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = np.where(slope != 0, z + r / slope, np.inf)
        inverse_vc = 1 / (crossing - vertices_m[k])
        before, after = conjugate[k]
        num = 1 / (after - vertices_m[k]) - inverse_vc
        den = 1 / (before - vertices_m[k]) - inverse_vc
        product *= indices[k + 1] / indices[k] * num / den
    return {
        "launch_radius_m": np.asarray(launch_radius_m),
        "abbe_ratio": abbe,
        "paper_m": product,
        "transmitted": trace["transmitted"],
        "max_spot_radius_m": trace["max_radius_m"],
    }


def aplanatism_error(result, reference=None):
    """RMS of (ratio / axial ratio - 1) over transmitted off-axis rays (paper Eq. 31)."""
    ok = result["transmitted"] & (result["launch_radius_m"] > 0)
    abbe = result["abbe_ratio"][ok]
    axial = abbe[0] if reference is None else reference
    return {
        "abbe_rms": float(np.sqrt(np.mean((abbe / axial - 1) ** 2))),
        "paper_rms": float(np.sqrt(np.mean((result["paper_m"][ok] - 1) ** 2))),
        "abbe_max": float(np.max(abs(abbe / axial - 1))),
        "paper_max": float(np.max(abs(result["paper_m"][ok] - 1))),
        "axial_abbe_ratio": float(axial),
    }
