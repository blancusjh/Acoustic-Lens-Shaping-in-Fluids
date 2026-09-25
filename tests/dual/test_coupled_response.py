import numpy as np

from acoustic_freeform.dual.coupled_response_audit import (
    low_rank_compliance,
    orthogonal_feedback_direction,
)


def test_full_rank_recovers_independent_coupled_linear_equilibrium():
    stiffness = np.diag([3.0, 5.0, 7.0])
    acoustic = np.array([[0.2, 0.1, 0.3], [-0.1, 0.5, 0.2], [0.4, -0.3, 0.6]])
    directions = np.array([[1.0, 0.2, 0.0], [0.3, 1.0, 0.1], [0.0, 0.4, 1.0]])
    compliance, _ = low_rank_compliance(stiffness, directions, acoustic @ directions)
    load = np.array([0.3, -0.1, 0.4])
    np.testing.assert_allclose(compliance @ load, np.linalg.solve(stiffness - acoustic, load))


def test_zero_feedback_is_mechanical_compliance():
    stiffness = np.array([[3.0, 0.2], [0.2, 5.0]])
    compliance, _ = low_rank_compliance(stiffness, np.ones((2, 1)), np.zeros((2, 1)))
    np.testing.assert_allclose(compliance, np.linalg.inv(stiffness))


def test_feedback_generated_basis_recovers_known_coupled_solution():
    stiffness = np.diag([3.0, 5.0, 7.0])
    acoustic = np.array([[0.2, 0.1, 0.3], [-0.1, 0.5, 0.2], [0.4, -0.3, 0.6]])
    load = np.array([0.3, -0.1, 0.4])
    first = np.linalg.solve(stiffness, load)
    directions = first[:, None] / np.linalg.norm(first)
    for _ in range(2):
        vector = orthogonal_feedback_direction(stiffness, directions, acoustic @ directions[:, -1])
        directions = np.column_stack([directions, vector / np.linalg.norm(vector)])
    gram = directions.T @ stiffness @ directions
    np.testing.assert_allclose(gram - np.diag(np.diag(gram)), 0, atol=1e-14)
    compliance, _ = low_rank_compliance(stiffness, directions, acoustic @ directions)
    np.testing.assert_allclose(compliance @ load, np.linalg.solve(stiffness - acoustic, load))
