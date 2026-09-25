import numpy as np

from acoustic_freeform.inverse.observations import replace_mechanical_compliance


def test_composed_observation_matches_independently_solved_coupled_system():
    stiffness = np.array([[3.0, 0.2], [0.2, 5.0]])
    acoustic = np.array([[0.4, -0.1], [0.7, 0.3]])
    physical_observation = np.array([[1.0, -2.0], [0.3, 0.6], [-0.2, 0.7]])
    load = np.array([0.2, -0.4])
    transformed = replace_mechanical_compliance(
        physical_observation @ np.linalg.inv(stiffness),
        stiffness,
        np.linalg.inv(stiffness - acoustic),
    )
    np.testing.assert_allclose(
        transformed @ load, physical_observation @ np.linalg.solve(stiffness - acoustic, load)
    )
