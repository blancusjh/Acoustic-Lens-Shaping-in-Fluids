import pytest

from acoustic_freeform.verify.metrics import geometric_acceptance


def test_mean_specification_does_not_add_carrier_motion():
    mean = [2e-9, 3e-9]
    phase = [20e-9, 30e-9]
    result = geometric_acceptance(mean, phase, "cycle_mean")
    assert result["declared_10nm_sampled_test"]
    assert result["instantaneous_10nm_rejected_by_samples"]
    assert not result["physical_accuracy_certified"]
    assert not geometric_acceptance(mean, phase, "instantaneous")["declared_10nm_sampled_test"]
    assert not geometric_acceptance([2e-9, 11e-9], phase, "cycle_mean")[
        "declared_10nm_sampled_test"
    ]


def test_accuracy_gate_requires_valid_both_face_data():
    with pytest.raises(ValueError):
        geometric_acceptance([0], [0], "cycle_mean")
    with pytest.raises(ValueError):
        geometric_acceptance([0, float("nan")], [0, 0], "cycle_mean")
    with pytest.raises(ValueError):
        geometric_acceptance([0, 0], [0, 0], "unspecified")
