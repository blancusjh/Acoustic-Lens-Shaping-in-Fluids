"""The fixed machine and the lens a user asks it to form."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.paths import REPOSITORY, data_path

DEFAULT_MACHINE = REPOSITORY / "configs/machine/v1.json"


@dataclass(frozen=True)
class Singlet:
    """A stigmatic singlet (SOSL) in the Silva-Lora and Torres convention.

    Distances are measured along the axis from the front lens vertex, in metres
    after multiplying by ``scale``: ``object_m`` is negative below the lens
    (``-inf`` for collimated light), ``image_m`` is positive above it, and
    ``thickness_m`` is the vertex-to-vertex thickness. The second face is fixed
    either by the paper's ``shape_factor`` or by an ``intermediate_m`` point.
    ``indices`` are the object-side, lens and image-side refractive indices
    used for the optical design (the cured lens in its use medium).
    """

    object_m: float
    image_m: float
    thickness_m: float
    shape_factor: float | None = None
    intermediate_m: float | None = None
    indices: tuple[float, float, float] = (1.0, 1.7, 1.0)
    scale: float = 1.0
    name: str = ""

    def __post_init__(self):
        for key in ("object_m", "image_m", "thickness_m", "scale"):
            object.__setattr__(self, key, float(getattr(self, key)))  # admits "-inf"
        object.__setattr__(self, "indices", tuple(float(n) for n in self.indices))
        if (self.shape_factor is None) == (self.intermediate_m is None):
            raise ValueError("Give exactly one of shape_factor and intermediate_m")
        if not self.thickness_m > 0 or not self.scale > 0:
            raise ValueError("Thickness and scale must be positive")
        if not self.object_m < 0 < self.image_m:
            raise ValueError("The object must lie below the lens (negative) and the image above (positive)")

    @classmethod
    def from_dict(cls, record):
        return cls(**record)

    def as_dict(self):
        record = asdict(self)
        record["object_m"] = str(self.object_m) if math.isinf(self.object_m) else self.object_m
        record["indices"] = list(self.indices)
        return {k: v for k, v in record.items() if v is not None and v != ""}


@dataclass(frozen=True)
class Machine:
    """Fixed hardware: cell, liquids, source layout, drive options and limits."""

    name: str
    material_reference: str
    cell: dict
    apparatus: dict
    modes: dict
    drive_options_hz: list
    limits: dict
    sigmas: list
    acceptance: dict
    source: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path=DEFAULT_MACHINE):
        path = data_path(Path(path))
        record = json.loads(path.read_text())
        keys = {f for f in cls.__dataclass_fields__ if f not in ("source", "extra")}
        extra = {k: v for k, v in record.items() if k not in keys}
        return cls(**{k: record[k] for k in keys}, source=str(path), extra=extra)

    def synthesis_inputs(self, lens, bands_hz):
        """A capability-synthesis configuration for this lens on this machine."""
        singlet = {
            "reference": lens.name or "user lens",
            **lens.as_dict(),
            "pin_radius_m": self.cell["pin_radius_m"],
            "clear_radius_m": self.cell["clear_radius_m"],
            "windows_m": list(self.cell["windows_m"]),
        }
        singlet.pop("name", None)
        return {
            "material_reference": self.material_reference,
            "singlet": singlet,
            "apparatus": dict(self.apparatus),
            "bands_hz": list(bands_hz),
            "modes": self.modes,
            "limits": self.limits,
            "sigmas": self.sigmas,
            "scope": f"{self.name}: lens design request; curing not modelled.",
        }
