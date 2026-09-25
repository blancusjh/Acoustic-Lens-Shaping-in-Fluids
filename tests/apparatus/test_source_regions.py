"""Geometric coverage of the declared ideal boundary source regions."""

import numpy as np

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.apparatus.sources import source_regions


def test_regions_cover_both_end_disks_and_cylinder_side():
    cfg = DualConfig(radial_ports=16, side_ports_per_layer=8)
    regions = source_regions(cfg)
    assert [r["channel"] for r in regions] == list(range(cfg.channels))
    for boundary in ("bottom", "top"):
        area = sum(r["area_m2"] for r in regions if r["boundary"] == boundary)
        np.testing.assert_allclose(area, np.pi * cfg.radius_m**2, rtol=1e-14)
    area = sum(r["area_m2"] for r in regions if r["boundary"] == "side")
    np.testing.assert_allclose(
        area, 2 * np.pi * cfg.radius_m * (cfg.levels_m[-1] - cfg.levels_m[0]), rtol=1e-14
    )
