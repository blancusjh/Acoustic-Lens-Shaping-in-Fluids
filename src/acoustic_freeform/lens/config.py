"""Finite lens apparatus and explicitly sourced material parameters, all SI."""

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LensConfig:
    name: str = "noa61_asphere"
    radius_m: float = 0.004
    clear_radius_m: float = 0.003
    depth_m: float = 0.006
    wall_thickness_m: float = 0.0008
    base_thickness_m: float = 0.0007
    focal_distance_m: float = 0.020
    refractive_index: float = 1.52
    wavelength_m: float = 589.3e-9
    density_kg_m3: float = 1231.0
    viscosity_pa_s: float = 0.3
    surface_tension_n_m: float = 0.04
    gravity_m_s2: float = 9.81
    sound_speed_m_s: float = 1600.0
    attenuation_np_m: float = 5.0
    frequency_hz: float = 800000.0
    array_rows: int = 16
    array_sectors: int = 16
    element_fill: float = 0.84
    max_wall_speed_m_s: float = 0.08
    surface_modes: int = 14
    mesh_radial: int = 40
    mesh_vertical: int = 48
    acoustic_order: int = 2
    pressure_penalty: float = 2.0
    end_s: float = 1.0
    step_s: float = 0.01
    ramp_s: float = 0.6

    @property
    def bond(self):
        return self.density_kg_m3 * self.gravity_m_s2 * self.radius_m**2 / self.surface_tension_n_m

    @property
    def capillary_time_s(self):
        return self.viscosity_pa_s * self.radius_m / self.surface_tension_n_m

    @property
    def conic_constant(self):
        return -(self.refractive_index**2)

    def as_dict(self):
        return asdict(self)

    def validate(self):
        if not 0 < self.clear_radius_m < self.radius_m or self.depth_m <= 0:
            raise ValueError("A clear aperture must fit inside the finite cylindrical chamber.")
        if self.refractive_index <= 1 or self.focal_distance_m <= 0:
            raise ValueError("Expected a positive focusing liquid/air optical target.")
        if not 0 < self.element_fill < 1 or min(self.array_rows, self.array_sectors) < 1:
            raise ValueError("Invalid cylindrical array.")
        if self.step_s <= 0 or self.ramp_s > self.end_s or self.ramp_s <= 0:
            raise ValueError("Invalid physical timing.")


def load_config(path: str | Path) -> LensConfig:
    data = tomllib.loads(Path(path).read_text())
    c = LensConfig(**data["lens"])
    c.validate()
    return c


MATERIAL_PROVENANCE = {
    "name": "Norland Optical Adhesive 61, liquid at nominal 25 C",
    "source": "https://norlandproducts.com/wp-content/uploads/2025/02/Norland-Products-NOA-61-TDS.pdf",
    "manufacturer_values": {
        "density_kg_m3": 1231,
        "viscosity_pa_s": 0.3,
        "surface_tension_n_m": 0.04,
        "liquid_refractive_index": 1.52,
    },
    "unmeasured_design_assumptions": ["sound speed", "acoustic attenuation", "liquid dispersion"],
    "scope": "A numerical lens in a stated material model; not a calibrated NOA61 acoustic experiment.",
    "curing": "Not simulated. The manufacturer's cured index is 1.56, not the liquid value 1.52.",
}
