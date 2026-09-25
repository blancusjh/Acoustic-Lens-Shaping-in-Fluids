import numpy as np
from scipy.optimize import least_squares

from acoustic_freeform.dual.emitter_free_annulus import JointAnnulusObjective
from acoustic_freeform.dual.precision import PrecisionObjective, WhitenedObjective


def test_joint_annulus_closes_all_three_known_force_equations():
    kernels = np.zeros((3, 2, 2), complex)
    kernels[0, 0, 0] = kernels[1, 1, 1] = 1
    kernels[2] = np.eye(2)
    cache = {
        "force_kernels": kernels,
        "required_force_n": np.ones(3),
        "compliance_observation_m_n": 1e-8 * np.eye(3),
        "observation_weights": np.ones(3),
        "carrier_height_response_s": np.zeros((1, 2), complex),
    }
    source = WhitenedObjective(PrecisionObjective(cache, 0, 2 / np.sqrt(2), 0))
    annulus = np.array([[0.0], [0.0], [1.0]])
    objective = JointAnnulusObjective(source, annulus, np.zeros(1), penalty=1e-6, scale_m=1e-8)
    initial = np.r_[source.encode(np.array([0.7, 0.8], complex)), 0.0]
    jacobian = objective.jacobian(initial)
    for j in range(len(initial)):
        perturbation = np.eye(len(initial))[j] * 1e-6
        central = (
            objective.residual(initial + perturbation) - objective.residual(initial - perturbation)
        ) / 2e-6
        np.testing.assert_allclose(jacobian[:, j], central, atol=2e-9)
    fit = least_squares(
        objective.residual, initial, jac=objective.jacobian, ftol=1e-13, xtol=1e-13, gtol=1e-13
    )
    drive = source.unpack(fit.x[: objective.source_size])
    np.testing.assert_allclose(abs(drive) ** 2, [1.0, 1.0], atol=2e-10)
    np.testing.assert_allclose(fit.x[-1], 1.0, atol=2e-10)
    force_residual = np.einsum("a,iab,b->i", drive.conj(), kernels, drive).real - 1
    force_residual -= annulus[:, 0] * fit.x[-1]
    np.testing.assert_allclose(force_residual, 0, atol=2e-10)
