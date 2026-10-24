"""Coupled stationary inverse with fixed pupils and free non-optical annuli."""

import numpy as np
from scipy.linalg import block_diag, null_space


def annulus_spaces(space):
    # More than degree+1 points in every pupil knot interval; a spline that
    # vanishes at these points vanishes identically there, up to roundoff.
    radii = np.linspace(0, space.config.clear_radius_m, 12 * space.count + 1)
    single = null_space(space.basis(radii), rcond=1e-12)
    annulus = block_diag(single, single)
    complement = null_space(annulus.T, rcond=1e-12)
    return annulus, complement


class JointAnnulusObjective:
    """All force rows retained; annulus motion is an explicit unknown."""

    def __init__(self, source, annulus, current, penalty=1e-3, scale_m=1e-6):
        self.source, self.annulus, self.current = source, annulus, current
        self.penalty, self.scale_m = penalty, scale_m
        self.source_size = 2 * source.channels
        self.shape_rows = annulus.shape[0]

    def residual(self, x):
        result = self.source.residual(x[: self.source_size])
        change = x[self.source_size :] * self.scale_m
        result[: self.shape_rows] -= self.source.base.shape_weight * (self.annulus @ change) / 1e-8
        return np.r_[result, self.penalty * (self.current + change) / self.scale_m]

    def jacobian(self, x):
        source = self.source.jacobian(x[: self.source_size])
        result = np.zeros((source.shape[0] + self.annulus.shape[1], len(x)))
        result[: source.shape[0], : self.source_size] = source
        result[: self.shape_rows, self.source_size :] = (
            -self.source.base.shape_weight[:, None] * self.annulus * self.scale_m / 1e-8
        )
        result[source.shape[0] :, self.source_size :] = self.penalty * np.eye(self.annulus.shape[1])
        return result
