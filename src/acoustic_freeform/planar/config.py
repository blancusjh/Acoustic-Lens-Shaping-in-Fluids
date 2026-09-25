"""SI-valued, inspectable experiment definitions. No hidden coefficients."""

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Grid:
    nx: int = 128
    ny: int = 128
    lx_m: float = 0.028
    ly_m: float = 0.028


@dataclass(frozen=True)
class Fluid:
    name: str
    density_kg_m3: float
    sound_speed_m_s: float
    viscosity_pa_s: float


@dataclass(frozen=True)
class Interface:
    surface_tension_n_m: float = 0.072
    gravity_m_s2: float = 9.81
    model: str = "weakly_damped"


@dataclass(frozen=True)
class Array:
    nx: int = 16
    ny: int = 16
    pitch_m: float = 0.0007
    element_width_m: float = 0.0006
    source_distance_m: float = 0.006
    frequency_hz: float = 1_000_000.0
    pressure_amplitude_pa: float = 8_000.0
    retain_evanescent: bool = False


@dataclass(frozen=True)
class Focus:
    x_m: float
    y_m: float
    weight: float = 1.0


@dataclass(frozen=True)
class Beam:
    name: str
    foci: tuple[Focus, ...]


@dataclass(frozen=True)
class Pulse:
    beam: str
    start_s: float
    stop_s: float
    ramp_s: float
    gain: float = 1.0

    def amplitude(self, time_s: float) -> float:
        if time_s <= self.start_s or time_s >= self.stop_s:
            return 0.0
        edge = min(time_s - self.start_s, self.stop_s - time_s)
        if edge >= self.ramp_s or self.ramp_s == 0:
            return self.gain
        return self.gain * 0.5 * (1 - math.cos(math.pi * edge / self.ramp_s))


@dataclass(frozen=True)
class Timing:
    end_s: float = 0.12
    step_s: float = 0.0001
    save_every: int = 5


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    material_status: str
    grid: Grid
    lower: Fluid
    upper: Fluid
    interface: Interface
    array: Array
    beams: tuple[Beam, ...]
    pulses: tuple[Pulse, ...]
    timing: Timing

    def validate(self) -> None:
        for n in (self.grid.nx, self.grid.ny):
            if not isinstance(n, int) or isinstance(n, bool) or n < 16 or n % 2:
                raise ValueError("Grid dimensions must be even integers of at least 16.")
        for value in (
            self.grid.lx_m,
            self.grid.ly_m,
            self.array.pitch_m,
            self.array.element_width_m,
            self.array.source_distance_m,
            self.array.frequency_hz,
            self.interface.surface_tension_n_m,
            self.timing.step_s,
            self.timing.end_s,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Geometric, frequency, capillary and time scales must be positive."
                )
        if (
            not math.isfinite(self.array.pressure_amplitude_pa)
            or self.array.pressure_amplitude_pa < 0
        ):
            raise ValueError("Pressure amplitude must be finite and nonnegative.")
        if any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1
            for n in (self.array.nx, self.array.ny)
        ):
            raise ValueError("Array dimensions must be positive integers.")
        if self.array.element_width_m > self.array.pitch_m:
            raise ValueError("Array elements overlap.")
        if max(self.array.nx, self.array.ny) * self.array.pitch_m >= min(
            self.grid.lx_m, self.grid.ly_m
        ):
            raise ValueError("The array must fit inside the transverse box.")
        for fluid in (self.lower, self.upper):
            values = (fluid.density_kg_m3, fluid.sound_speed_m_s, fluid.viscosity_pa_s)
            if any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError("Fluid properties must be positive and finite.")
        if self.interface.model not in {"stokes", "weakly_damped"}:
            raise ValueError("Interface model must be stokes or weakly_damped.")
        if (
            not math.isfinite(self.interface.gravity_m_s2)
            or self.interface.gravity_m_s2 < 0
            or self.lower.density_kg_m3 < self.upper.density_kg_m3
        ):
            raise ValueError("This prototype supports stable density stratification.")
        names = [b.name for b in self.beams]
        if not names or len(names) != len(set(names)):
            raise ValueError("Provide uniquely named beams.")
        for beam in self.beams:
            if not beam.foci or not any(f.weight > 0 for f in beam.foci):
                raise ValueError("Each beam needs at least one positive focal weight.")
            if any(
                not all(math.isfinite(v) for v in (f.x_m, f.y_m, f.weight)) or f.weight < 0
                for f in beam.foci
            ):
                raise ValueError(
                    "Focal coordinates and weights must be finite; weights nonnegative."
                )
        for pulse in self.pulses:
            if pulse.beam not in names:
                raise ValueError(f"Unknown beam: {pulse.beam}")
            if not all(
                math.isfinite(v) for v in (pulse.start_s, pulse.stop_s, pulse.ramp_s, pulse.gain)
            ):
                raise ValueError("Pulse parameters must be finite.")
            if not 0 <= pulse.start_s < pulse.stop_s <= self.timing.end_s:
                raise ValueError("Pulse times must lie in the simulated interval.")
            if not 0 <= pulse.ramp_s <= (pulse.stop_s - pulse.start_s) / 2 or pulse.gain < 0:
                raise ValueError("Invalid pulse ramp or amplitude.")
        if (
            not isinstance(self.timing.save_every, int)
            or isinstance(self.timing.save_every, bool)
            or self.timing.save_every < 1
        ):
            raise ValueError("save_every must be a positive integer.")
        steps = self.timing.end_s / self.timing.step_s
        if abs(steps - round(steps)) > 1e-8:
            raise ValueError("End time must be an integer number of fixed steps.")

    def as_dict(self) -> dict:
        return asdict(self)


def load_experiment(path: str | Path) -> Experiment:
    with Path(path).open("rb") as stream:
        d = tomllib.load(stream)
    return experiment_from_dict(d)


def experiment_from_dict(d: dict) -> Experiment:
    """Construct the canonical experiment from TOML or a saved JSON report."""
    experiment = Experiment(
        name=d["name"],
        description=d["description"],
        material_status=d["material_status"],
        grid=Grid(**d["grid"]),
        lower=Fluid(**d["lower"]),
        upper=Fluid(**d["upper"]),
        interface=Interface(**d["interface"]),
        array=Array(**d["array"]),
        beams=tuple(Beam(b["name"], tuple(Focus(**f) for f in b["foci"])) for b in d["beams"]),
        pulses=tuple(Pulse(**p) for p in d["pulses"]),
        timing=Timing(**d["timing"]),
    )
    experiment.validate()
    return experiment
