import numpy as np
import pytest

from acoustic_freeform.acoustics.helmholtz import DualAcoustics
from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface


@pytest.mark.parametrize("order", [2, 4])
def test_batched_assembly_preserves_full_driven_field_and_interface_traction(order):
    cfg = DualConfig(
        radial_cells=8,
        cells_per_layer=4,
        radial_ports=4,
        side_ports_per_layer=2,
        acoustic_order=order,
        surface_elements=4,
    )
    space = DualSurface(cfg)
    rng = np.random.default_rng(104)
    q = rng.normal(size=(2, space.count)) * 1e-6
    drive = (rng.normal(size=cfg.channels) + 1j * rng.normal(size=cfg.channels)) * 0.1
    reference = DualAcoustics(cfg, space, volume_assembly_chunk_size=None).solve(q, drive)
    batched = DualAcoustics(cfg, space, volume_assembly_chunk_size=19).solve(q, drive)
    np.testing.assert_allclose(batched.solution, reference.solution, rtol=2e-11, atol=2e-6)
    np.testing.assert_allclose(
        batched.force(np.ones(1, complex)),
        reference.force(np.ones(1, complex)),
        rtol=2e-10,
        atol=1e-16,
    )
    np.testing.assert_array_equal(batched.basis.doflocs, reference.basis.doflocs)
    assert batched.residual < 2e-12
