import copy
import json

import numpy as np
import pytest

from acoustic_freeform.dual.emitter_force_design import checked_cache


def test_cache_allows_command_bound_but_rejects_physics_changes(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    inputs = {
        "case": {"faces": [{"vertex_displacement_m": 1e-4}]},
        "material_reference_sha256": "material-test-hash",
        "wave_model": "test-model",
        "apparatus": {
            "frequency_hz": 1e6,
            "max_source_speed_m_s": 1.0,
            "densities": (1.0, 2.0, 1.0),
        },
    }
    (source / "config.json").write_text(json.dumps(inputs))
    np.savez(source / "frozen-operator.npz", required_force_n=np.array([1.0, 2.0]))
    changed = copy.deepcopy(inputs)
    changed["apparatus"]["max_source_speed_m_s"] = 2.0
    cache = checked_cache(changed, source, output)
    np.testing.assert_array_equal(cache["required_force_n"], [1.0, 2.0])
    provenance = json.loads((output / "cache-provenance.json").read_text())
    assert len(provenance["sha256"]) == 64
    changed["apparatus"]["frequency_hz"] = 2e6
    with pytest.raises(ValueError, match="apparatus"):
        checked_cache(changed, source, output)
    changed = copy.deepcopy(inputs)
    changed["case"]["faces"][0]["vertex_displacement_m"] = 2e-4
    with pytest.raises(ValueError, match="case"):
        checked_cache(changed, source, output)
