"""Lossless angular-spectrum acoustics and the jump in acoustic momentum flux.

Complex fields use PEAK amplitudes and exp(-i*omega*t). Interface z=0; lower
fluid occupies z<0. Reflected/transmitted waves satisfy pressure and normal
velocity continuity mode by mode. The interface remains flat in this acoustic
reference calculation. No shape-dependent scattering, attenuation or streaming.

Radiation stress: Chesneau et al., PRE 106, 065104 (2022), Eqs. 9--14.
"""

from dataclasses import dataclass

import numpy as np

from .config import Array, Fluid
from .spectral import SpectralGrid


@dataclass
class HarmonicField:
    pressure: np.ndarray
    velocity: np.ndarray  # (3, ny, nx), m/s; FAST acoustic particle velocity

    def intensity(self) -> np.ndarray:
        return 0.5 * np.real(self.pressure[None, ...] * self.velocity.conj())


@dataclass
class PlanarSolution:
    incident: np.ndarray
    reflected: np.ndarray
    transmitted: np.ndarray
    below: HarmonicField
    above: HarmonicField
    radiation_pressure: np.ndarray
    powers_w: dict[str, float]
    source: np.ndarray


def normal_flux_bilinear(a: HarmonicField, b: HarmonicField, fluid: Fluid) -> np.ndarray:
    """Sesquilinear form whose diagonal is the normal acoustic momentum flux."""
    rho, c = fluid.density_kg_m3, fluid.sound_speed_m_s
    return a.pressure * b.pressure.conj() / (4 * rho * c * c) + rho / 4 * (
        a.velocity[2] * b.velocity[2].conj()
        - a.velocity[0] * b.velocity[0].conj()
        - a.velocity[1] * b.velocity[1].conj()
    )


def radiation_pressure(
    below: HarmonicField, above: HarmonicField, lower: Fluid, upper: Fluid
) -> np.ndarray:
    """Positive pressure pushes the interface toward +z. Units Pa."""
    return (
        normal_flux_bilinear(below, below, lower) - normal_flux_bilinear(above, above, upper)
    ).real


class PlanarAcoustics:
    def __init__(self, grid: SpectralGrid, array: Array, lower: Fluid, upper: Fluid):
        self.grid, self.array, self.lower, self.upper = grid, array, lower, upper
        self.omega = 2 * np.pi * array.frequency_hz
        self.kz1 = np.sqrt((self.omega / lower.sound_speed_m_s) ** 2 - grid.k**2 + 0j)
        self.kz2 = np.sqrt((self.omega / upper.sound_speed_m_s) ** 2 - grid.k**2 + 0j)
        self.y1 = self.kz1 / (self.omega * lower.density_kg_m3)
        self.y2 = self.kz2 / (self.omega * upper.density_kg_m3)
        denom = self.y1 + self.y2
        self.reflection = np.divide(
            self.y1 - self.y2, denom, out=np.zeros_like(denom), where=np.abs(denom) > 1e-30
        )
        self.transmission = 1 + self.reflection

    def _field(
        self, forward: np.ndarray, backward: np.ndarray, kz: np.ndarray, fluid: Fluid
    ) -> HarmonicField:
        p = forward + backward
        scale = 1 / (self.omega * fluid.density_kg_m3)
        velocities = [
            self.grid.kxx * p * scale,
            self.grid.kyy * p * scale,
            kz * (forward - backward) * scale,
        ]
        return HarmonicField(
            self.grid.inverse(p), np.stack([self.grid.inverse(v) for v in velocities])
        )

    def solve(self, source_spectrum: np.ndarray) -> PlanarSolution:
        source = np.asarray(source_spectrum, dtype=complex).copy()
        if source.shape != self.grid.k.shape or not np.isfinite(source).all():
            raise ValueError("Source spectrum has invalid shape or entries.")
        if not self.array.retain_evanescent:
            source[self.grid.k >= self.omega / self.lower.sound_speed_m_s * (1 - 1e-12)] = 0
        inc = source * np.exp(1j * self.kz1 * self.array.source_distance_m)
        ref, tra = self.reflection * inc, self.transmission * inc
        below = self._field(inc, ref, self.kz1, self.lower)
        above = self._field(tra, np.zeros_like(tra), self.kz2, self.upper)
        factor = self.grid.area * 0.5
        power_in = factor * np.sum(np.abs(inc) ** 2 * self.y1.real)
        power_ref = factor * np.sum(np.abs(ref) ** 2 * self.y1.real)
        power_tra = factor * np.sum(np.abs(tra) ** 2 * self.y2.real)
        lower_net = factor * np.real(np.sum((inc + ref) * (self.y1 * (inc - ref)).conj()))
        # An incident evanescent mode carries no normal power alone, but its
        # interference with the reflected evanescent field can transfer energy.
        cross_power = lower_net - power_in + power_ref
        powers = {
            "incident": float(power_in),
            "reflected": float(power_ref),
            "transmitted": float(power_tra),
            "lower_net": float(lower_net),
            "evanescent_interference": float(cross_power),
        }
        return PlanarSolution(
            inc,
            ref,
            tra,
            below,
            above,
            radiation_pressure(below, above, self.lower, self.upper),
            powers,
            source,
        )

    def sample(
        self, solution: PlanarSolution, z_m: float, medium: str | None = None, part: str = "total"
    ) -> HarmonicField:
        """Fields at a plane. medium selects either limit when z=0."""
        if medium not in {None, "lower", "upper"} or part not in {"total", "incident"}:
            raise ValueError("Unknown medium or field part.")
        lower = medium == "lower" or (medium is None and z_m <= 0)
        if lower:
            if not -self.array.source_distance_m - 1e-12 <= z_m <= 1e-12:
                raise ValueError("Lower-field samples must lie between source and interface.")
            inc = solution.source * np.exp(1j * self.kz1 * (z_m + self.array.source_distance_m))
            ref = solution.reflected * np.exp(-1j * self.kz1 * z_m)
            if part == "incident":
                ref = np.zeros_like(ref)
            return self._field(inc, ref, self.kz1, self.lower)
        if z_m < 0 or part != "total":
            raise ValueError("Upper-field samples require z>=0 and transmitted total field.")
        tra = solution.transmitted * np.exp(1j * self.kz2 * z_m)
        return self._field(tra, np.zeros_like(tra), self.kz2, self.upper)


class RadiationBasis:
    """Exact coherent quadratic superposition for REAL beam-envelope amplitudes.

    Each beam may have an arbitrary complex element phase/amplitude pattern.
    Cross terms are retained between simultaneous beams at the common carrier.
    """

    def __init__(self, solutions: list[PlanarSolution], lower: Fluid, upper: Fluid):
        count = len(solutions)
        shape = solutions[0].radiation_pressure.shape
        self.kernels = np.empty((count, count, *shape))
        for i, a in enumerate(solutions):
            for j, b in enumerate(solutions):
                self.kernels[i, j] = (
                    normal_flux_bilinear(a.below, b.below, lower)
                    - normal_flux_bilinear(a.above, b.above, upper)
                ).real

    def pressure(self, amplitudes: np.ndarray) -> np.ndarray:
        return np.einsum("i,j,ijyx->yx", amplitudes, amplitudes, self.kernels)
