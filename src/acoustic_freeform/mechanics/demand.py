"""Load demand: the extra traction a face needs to hold a prescribed shape.

At an equilibrium of the pinned, volume-conserving graph energy (``surface``),
a face h(r) with tension sigma and density step drho = rho_below - rho_above
carries an extra normal traction t (positive upward on the face) with

    -sigma (1/r) (r phi)' + drho g h = t + P,    phi = h' / sqrt(1 + h'^2),

where the constant P is the face's pressure multiplier (the volume-preserving
basis ignores it). The demand of a target face is the left side evaluated on
the target; the acoustic field must supply it up to a constant. Expanding the
sag h = h0 + c r^2/2 + a4 r^4 + a6 r^6 gives t = const + Q2 r^2 + Q4 r^4 + ...,
the low-order load a common source has to produce.
"""

import numpy as np


def demanded_traction(target, sigma, drho_g, r):
    """Exact demand of ``target`` (an object with evaluate(r, derivative)) at r > 0, in Pa.

    Defined up to an additive constant; the volume constraint absorbs it.
    """
    r = np.asarray(r, float)
    h, slope, second = (target.evaluate(r, k) for k in range(3))
    stretch = np.sqrt(1 + slope**2)
    # (1/r)(r phi)' = phi' + phi / r with phi' = h'' / (1 + h'^2)^(3/2).
    curvature = second / stretch**3 + slope / (r * stretch)
    return -sigma * curvature + drho_g * h


def sag_series(target, radius_m, fraction=0.25, order=5):
    """Even-power coefficients c, a4, a6 of the sag near the axis (least squares)."""
    scale = fraction * radius_m
    x = np.linspace(0, 1, 400)
    h = target.evaluate(x * scale)
    # Fit in x = r / scale: raw powers of r in metres are hopelessly conditioned.
    powers = np.vstack([x ** (2 * k) for k in range(order)]).T
    coefficients = np.linalg.lstsq(powers, h, rcond=None)[0] / scale ** (2 * np.arange(order))
    return {"c": 2 * coefficients[1], "a4": coefficients[2], "a6": coefficients[3]}


def series_demand(c, a4, a6, sigma, drho_g):
    """Leading coefficients Q2, Q4 (Pa/m^2, Pa/m^4) of the demanded traction.

    Both vanish for a sphere when drho = 0; Q2 = 0 is the conic matching rule
    drho g / sigma = 4 k c^2 with a4 = (1 + k) c^3 / 8.
    """
    q2 = -sigma * (16 * a4 - 2 * c**3) + drho_g * c / 2
    q4 = -sigma * (36 * a6 - 36 * c**2 * a4 + 9 * c**5 / 4) + drho_g * a4
    return {"Q2_pa_m2": q2, "Q4_pa_m4": q4}


def demanded_force(space, targets):
    """Generalized demand of both targets in the surface space: the mechanical
    gradient at their projection. The space must carry the targets' volumes."""
    q = space.project(targets)
    force, stiffness, _ = space.mechanics(q)
    return q, force, stiffness
