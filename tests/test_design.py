"""Lens description, machine file and settings that need no acoustic solve."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from acoustic_freeform.acoustics.helmholtz import merged_nodes
from acoustic_freeform.design import Machine, Singlet, fill_volumes


def test_singlet_accepts_collimated_and_round_trips():
    lens = Singlet(object_m="-inf", image_m=0.06, thickness_m=0.02, shape_factor=0.0, scale=1 / 14)
    assert math.isinf(lens.object_m) and lens.object_m < 0
    again = Singlet.from_dict(lens.as_dict())
    assert again == lens


@pytest.mark.parametrize("bad", [
    {"object_m": -0.1, "image_m": 0.06, "thickness_m": 0.02},  # no shape given
    {"object_m": -0.1, "image_m": 0.06, "thickness_m": 0.02, "shape_factor": 0, "intermediate_m": 0.01},
    {"object_m": 0.1, "image_m": 0.06, "thickness_m": 0.02, "shape_factor": 0},  # object above the lens
    {"object_m": -0.1, "image_m": 0.06, "thickness_m": -0.02, "shape_factor": 0},
])
def test_singlet_rejects_inconsistent_requests(bad):
    with pytest.raises(ValueError):
        Singlet(**bad)


def test_default_machine_loads_and_builds_synthesis_inputs():
    machine = Machine.load()
    lens = Singlet(object_m=-0.15, image_m=0.09, thickness_m=0.015, shape_factor=0.5, scale=1 / 14)
    inputs = machine.synthesis_inputs(lens, machine.drive_options_hz[0])
    assert inputs["singlet"]["windows_m"] == machine.cell["windows_m"]
    assert inputs["bands_hz"] == [1.5e6]
    assert machine.drive_options_hz[-1] == [0.75e6, 1.5e6]


def test_fill_volumes_conserve_the_cell_volume():
    # Whatever the face shapes, the three layers fill the cylinder between the windows.
    cfg = SimpleNamespace(radius_m=2e-3, levels_m=(-1e-3, 0.0, 1e-3, 2e-3), volume_offsets_m3=(3e-10, -2e-10))
    volumes = fill_volumes(cfg)
    total = math.pi * cfg.radius_m**2 * (cfg.levels_m[3] - cfg.levels_m[0])
    assert sum(volumes) == pytest.approx(total, rel=1e-14)
    flat = fill_volumes(SimpleNamespace(**{**vars(cfg), "volume_offsets_m3": (0.0, 0.0)}))
    assert flat[0] == pytest.approx(math.pi * 4e-6 * 1e-3)
    assert volumes[0] - flat[0] == pytest.approx(3e-10)


def test_merged_nodes_removes_rounding_duplicates_only():
    # The S09 mesh defect: two layer fractions one rounding unit apart made zero-height cells.
    points = np.array([0.0, 0.25, 0.25 + 1e-16, 0.5, 0.5000001, 1.0])
    merged = merged_nodes(points)
    np.testing.assert_allclose(merged, [0.0, 0.25, 0.5, 0.5000001, 1.0])
