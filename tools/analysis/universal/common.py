"""Shared helpers: published singlets placed in one common cell."""

import json
from dataclasses import replace
from pathlib import Path

from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import pinned_singlet, singlet_geometry

ROOT = Path(__file__).resolve().parents[3]
LENSES = {
    "fig9b": ROOT / "configs/S08-published-singlet/fig9b-1p5mhz-caps-1w-2mpa.json",
    "fig7a": ROOT / "configs/S08-published-singlet/fig7a-1p5mhz-caps-1w-2mpa.json",
}
WATER = ROOT / "configs/shared/noa61-water-pressure-reference.json"
# Common windows for both lenses: 0.8 mm below the front vertex and 0.8 mm above
# the thicker lens's back vertex (Fig. 7a, t = 1.4286 mm).
COMMON_WINDOWS_M = (-0.0008, 0.0022286)


def lens_cell(name, material=WATER, windows=COMMON_WINDOWS_M, overrides=None, singlet=None):
    """Cell (pin ring at the lens rim heights, common windows), targets and spec.

    ``name`` selects a published lens; ``singlet`` replaces its singlet block
    (same keys as the S08 configs) to place any other lens in the same cell.
    """
    spec = json.loads(LENSES[name].read_text())
    if singlet is not None:
        spec["singlet"] = singlet
    singlet = spec["singlet"]
    material = json.loads(Path(material).read_text())
    geometry = singlet_geometry(singlet)
    levels = [windows[0], geometry["levels_m"][1], geometry["levels_m"][2], windows[1]]
    apparatus = {k: v for k, v in spec["apparatus"].items() if k != "volume_offsets_m3"}
    cfg = DualConfig(
        **{
            **material["apparatus"],
            **apparatus,
            "radius_m": singlet["pin_radius_m"],
            "clear_radius_m": singlet["clear_radius_m"],
            "levels_m": tuple(levels),
            **(overrides or {}),
        }
    )
    faces = pinned_singlet(cfg, singlet["indices"], geometry["conjugates_m"], 0.0, geometry["thickness_m"])
    cfg = replace(cfg, volume_offsets_m3=tuple(f.volume_offset() for f in faces))
    faces = pinned_singlet(cfg, singlet["indices"], geometry["conjugates_m"], 0.0, geometry["thickness_m"])
    return cfg, faces, {**spec, "optics_conjugates_m": geometry["conjugates_m"], "thickness_m": geometry["thickness_m"]}
