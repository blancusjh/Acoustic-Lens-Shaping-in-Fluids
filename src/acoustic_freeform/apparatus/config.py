"""SI apparatus data for three-fluid harmonic transmission experiments."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DualConfig:
    radius_m: float = 0.004
    clear_radius_m: float = 0.002
    levels_m: tuple = (-0.003, -0.001, 0.001, 0.003)
    density_kg_m3: tuple = (1200.0, 1100.0, 1000.0)
    sound_speed_m_s: tuple = (1500.0, 1200.0, 1500.0)
    surface_tension_n_m: tuple = (0.025, 0.025)
    gravity_m_s2: float = 9.81
    frequency_hz: float = 1.0e6
    port_admittance_factor: float = 1.0
    side_admittance_factor: float = 0.5
    max_source_speed_m_s: float = 1.0
    radial_ports: int = 8
    side_ports_per_layer: int = 4
    radial_cells: int = 32
    cells_per_layer: int = 16
    acoustic_order: int = 3
    normal_velocity_trace: str = "weak_flux"
    surface_elements: int = 20
    surface_degree: int = 5

    def validate(self):
        if not 0 < self.clear_radius_m < self.radius_m:
            raise ValueError("The clear aperture must lie inside the cylinder.")
        if len(self.levels_m) != 4 or np.any(np.diff(self.levels_m) <= 0):
            raise ValueError("Four ordered chamber/interface levels are required.")
        for data in (self.density_kg_m3, self.sound_speed_m_s):
            if len(data) != 3 or np.any(np.asarray(data) <= 0):
                raise ValueError("Three positive material constants are required.")
        if len(self.surface_tension_n_m) != 2 or min(self.surface_tension_n_m) <= 0:
            raise ValueError("Both surface tensions must be positive.")
        if min(self.frequency_hz, self.port_admittance_factor, self.max_source_speed_m_s) <= 0:
            raise ValueError("Positive frequency, admittance and source limit required.")
        if self.side_admittance_factor < 0:
            raise ValueError("Side ports must be passive.")
        if min(self.radial_ports, self.side_ports_per_layer) < 1:
            raise ValueError("Each source boundary must have at least one port.")
        if self.acoustic_order not in (2, 3, 4, 5, 6):
            raise ValueError("Supported acoustic orders: 2 through 6.")
        if self.normal_velocity_trace not in ("weak_flux", "averaged_gradient"):
            raise ValueError("Unknown normal-velocity trace recovery.")
        if self.surface_degree != 5 or self.surface_elements < 4:
            raise ValueError("Use quintic surfaces with at least four elements.")

    @property
    def channels(self):
        return 2 * self.radial_ports + 3 * self.side_ports_per_layer

    def as_dict(self):
        return asdict(self)
