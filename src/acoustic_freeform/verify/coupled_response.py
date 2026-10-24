"""Fixed-command low-mode acoustic shape feedback at a declared reference shape.

Central differences recompute the wave problem, not an interpolated field.
This is a truncated static response audit, not a stability certificate.
"""

import numpy as np


def low_rank_compliance(stiffness, directions, force_derivatives):
    """Invert K-F_q V W with the K-orthogonal projector W V=I.

    The returned map is exact for this declared low-rank derivative only.
    """
    inverse = np.linalg.inv(stiffness)
    projection = np.linalg.solve(directions.T @ stiffness @ directions, directions.T @ stiffness)
    feedback = inverse @ force_derivatives
    small = np.eye(directions.shape[1]) - projection @ feedback
    compliance = inverse + feedback @ np.linalg.solve(small, projection @ inverse)
    return compliance, small


def orthogonal_feedback_direction(stiffness, directions, force_derivative):
    """K-orthogonal Arnoldi direction for the static feedback K^{-1} A."""
    vector = np.linalg.solve(stiffness, force_derivative)
    gram = directions.T @ stiffness @ directions
    for _ in range(2):
        vector -= directions @ np.linalg.solve(gram, directions.T @ stiffness @ vector)
    return vector
