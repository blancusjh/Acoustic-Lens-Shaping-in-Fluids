"""Several frequency bands driven together through one source layout.

Each band is an independent time-harmonic field with its own command; the
cross terms between bands oscillate at the difference frequencies, which the
creeping-flow interfaces (relaxation rates of order 10 1/s) do not follow, so
the time-averaged generalized forces and their shape Jacobians add. The
wrapper exposes the same ``solve`` / ``shape_force_jacobian`` interface as
``DualAcoustics`` with a drive of shape (bands, channels).
"""

from dataclasses import replace

import numpy as np

from .helmholtz import DualAcoustics


class MultiBandResponse:
    def __init__(self, responses, drives):
        self.responses, self.drives = responses, drives

    def force(self, weights):
        return sum(r.force(weights) for r in self.responses)

    def diagnostics(self, weights):
        bands = [r.diagnostics(weights) for r in self.responses]
        return {
            "bands": bands,
            "source_power_w": float(sum(b["source_power_w"] for b in bands)),
            "boundary_absorption_w": float(sum(b["boundary_absorption_w"] for b in bands)),
            # Worst case of the instantaneous sum: band amplitudes in phase.
            "sampled_pressure_peak_pa": float(sum(b["sampled_pressure_peak_pa"] for b in bands)),
            "band_sampled_pressure_peak_pa": [b["sampled_pressure_peak_pa"] for b in bands],
        }


class MultiBandAcoustics:
    """Sum of single-band solvers; ``frequencies_hz`` replace the config frequency."""

    def __init__(self, config, surface, frequencies_hz, **options):
        self.frequencies_hz = [float(f) for f in frequencies_hz]
        self.bands = [
            DualAcoustics(replace(config, frequency_hz=f), surface, **options)
            for f in self.frequencies_hz
        ]

    @property
    def order(self):
        return self.bands[0].order

    @order.setter
    def order(self, value):
        for band in self.bands:
            band.order = value

    def solve(self, q, drive):
        drive = np.atleast_2d(drive)
        return MultiBandResponse([b.solve(q, d) for b, d in zip(self.bands, drive)], drive)

    def shape_force_jacobian(self, q, drive):
        drive = np.atleast_2d(drive)
        pairs = [b.shape_force_jacobian(q, d) for b, d in zip(self.bands, drive)]
        return (
            MultiBandResponse([p[0] for p in pairs], drive),
            sum(p[1] for p in pairs),
        )


def wave_for(config, surface, bands=None, **options):
    """Single-band solver, or the multi-band sum when ``bands`` (Hz) are given."""
    if bands is None:
        return DualAcoustics(config, surface, **options)
    return MultiBandAcoustics(config, surface, bands, **options)


def acoustics_for_state(config, surface, state, **options):
    """Solver and drive for a saved state: single band, or bands when the state
    records ``band_frequencies_hz`` with a (bands, channels) drive."""
    drive = state["source_velocity_m_s"]
    if "band_frequencies_hz" in state and drive.ndim == 2:
        return MultiBandAcoustics(config, surface, state["band_frequencies_hz"], **options), drive
    return DualAcoustics(config, surface, **options), drive
