"""Full-domain prescribed mean traction; no emitters or formation simulation."""

import numpy as np
from numpy.polynomial.legendre import leggauss


def traction_components(target, radius, sigma, density_jump, gravity):
    """Upward normal traction in Pa; height is relative to the pinned plane.

    Axisymmetric specialization of the full graph curvature, with the regular
    axis limit. No mechanical discrete gradient is used to generate this load.
    """
    radius = np.asarray(radius, float)
    height = target.evaluate(radius)
    slope = target.evaluate(radius, 1)
    second = target.evaluate(radius, 2)
    stretch = np.sqrt(1 + slope**2)
    radial = np.divide(slope, radius * stretch, out=second.copy(), where=radius != 0)
    return -sigma * (second / stretch**3 + radial), density_jump * gravity * height


def quadrature(edges, order):
    x, w = leggauss(order)
    r = (edges[:-1, None] + np.diff(edges)[:, None] * (x + 1) / 2).ravel()
    area = 2 * np.pi * r * (np.diff(edges)[:, None] * w / 2).ravel()
    return r, area
