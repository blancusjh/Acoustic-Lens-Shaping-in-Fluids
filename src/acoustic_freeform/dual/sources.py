"""Declared axisymmetric source regions, not point-transducer hardware claims."""

import csv
import json
from itertools import pairwise

import numpy as np


def source_regions(config):
    regions = []
    edges = np.linspace(0, config.radius_m, config.radial_ports + 1)
    for boundary, z, normal in (
        ("bottom", config.levels_m[0], -1),
        ("top", config.levels_m[-1], 1),
    ):
        for lo, hi in pairwise(edges):
            regions.append(
                {
                    "channel": len(regions),
                    "boundary": boundary,
                    "r_min_m": float(lo),
                    "r_max_m": float(hi),
                    "z_min_m": z,
                    "z_max_m": z,
                    "outward_normal_r": 0,
                    "outward_normal_z": normal,
                    "area_m2": float(np.pi * (hi**2 - lo**2)),
                }
            )
    for layer in range(3):
        edges = np.linspace(
            config.levels_m[layer], config.levels_m[layer + 1], config.side_ports_per_layer + 1
        )
        for lo, hi in pairwise(edges):
            regions.append(
                {
                    "channel": len(regions),
                    "boundary": "side",
                    "phase": layer,
                    "r_min_m": config.radius_m,
                    "r_max_m": config.radius_m,
                    "z_min_m": float(lo),
                    "z_max_m": float(hi),
                    "outward_normal_r": 1,
                    "outward_normal_z": 0,
                    "area_m2": float(2 * np.pi * config.radius_m * (hi - lo)),
                }
            )
    return regions


def export_sources(directory, config, drive=None):
    (directory / "source-regions.json").write_text(
        json.dumps(
            {
                "frequency_hz": config.frequency_hz,
                "source_condition": "V.n = Y P + g; stored command is g, not total velocity",
                "phasor_convention": "Peak complex amplitude, exp(-i omega t)",
                "apparatus_scope": "Ideal full-azimuth annuli and sidewall bands; not discrete point emitters",
                "electrical_calibration": None,
                "regions": source_regions(config),
            },
            indent=2,
        )
        + "\n"
    )
    if drive is not None:
        if len(drive) != config.channels:
            raise ValueError("One command per declared source region is required.")
        with (directory / "source-commands.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["channel", "real_m_s", "imag_m_s", "peak_amplitude_m_s", "phase_rad"])
            for j, value in enumerate(drive):
                writer.writerow([j, value.real, value.imag, abs(value), np.angle(value)])
