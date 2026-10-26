"""Declared axisymmetric source regions, not point-transducer hardware claims."""

import csv
import json
from itertools import pairwise

import numpy as np
from scipy.special import j0, jn_zeros


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


def modal_source_matrix(config, window_modes, wall_modes, quadrature=64):
    """Region averages of low-order continuous source modes; columns are modes.

    Window modes are the axisymmetric Neumann eigenfunctions J0(j'_{0,k} r / R),
    including the uniform mode; wall modes are cos(k pi (z - z0) / H) over the
    whole wall height, including the uniform mode. Every mode has unit mean
    square on its boundary, so a coefficient is a velocity amplitude in m/s.
    ``window_modes`` maps "bottom"/"top" to a mode count (0 disables a window).
    The declared regions are a quadrature of these continuous sources.
    """
    radius, levels = config.radius_m, config.levels_m
    zeros = np.r_[0.0, jn_zeros(1, max(window_modes.values(), default=1))]
    x, w = np.polynomial.legendre.leggauss(quadrature)
    columns, labels = [], []
    regions = source_regions(config)

    def average(region, profile):
        if region["boundary"] == "side":
            lo, hi = region["z_min_m"], region["z_max_m"]
            z = lo + (hi - lo) * (x + 1) / 2
            return float(np.sum(w * profile(z)) / 2)
        lo, hi = region["r_min_m"], region["r_max_m"]
        r = lo + (hi - lo) * (x + 1) / 2
        weight = w * r * (hi - lo) / 2
        return float(np.sum(weight * profile(r)) / np.sum(weight))

    for boundary in ("bottom", "top"):
        for k in range(window_modes.get(boundary, 0)):
            # Mean square of J0(j r / R) over the disk is J0(j)^2 for j' zeros.
            scale = 1 / abs(j0(zeros[k]))

            def profile(r, k=k, scale=scale):
                return scale * j0(zeros[k] * r / radius)

            columns.append(
                [average(g, profile) if g["boundary"] == boundary else 0.0 for g in regions]
            )
            labels.append({"boundary": boundary, "order": k, "function": "J0(j'_{0,k} r/R)"})
    height = levels[-1] - levels[0]
    for k in range(wall_modes):
        scale = 1.0 if k == 0 else np.sqrt(2.0)

        def profile(z, k=k, scale=scale):
            return scale * np.cos(k * np.pi * (z - levels[0]) / height)

        columns.append([average(g, profile) if g["boundary"] == "side" else 0.0 for g in regions])
        labels.append({"boundary": "side", "order": k, "function": "cos(k pi (z-z0)/H)"})
    return np.array(columns).T, labels


def modal_source_profile(config, labels, coefficients, boundary, positions):
    """Continuous complex source velocity of the modal command on one boundary.

    ``positions`` are radii (m) on a window or heights (m) on the wall. Uses the
    same functions and normalization as ``modal_source_matrix``; the declared
    regions apply the region averages of this field.
    """
    positions = np.asarray(positions, float)
    radius, levels = config.radius_m, config.levels_m
    count = max((g["order"] + 1 for g in labels if g["boundary"] != "side"), default=1)
    zeros = np.r_[0.0, jn_zeros(1, count)]
    height = levels[-1] - levels[0]
    field = np.zeros(positions.shape, complex)
    for label, value in zip(labels, coefficients):
        if label["boundary"] != boundary:
            continue
        k = label["order"]
        if boundary == "side":
            scale = 1.0 if k == 0 else np.sqrt(2.0)
            field += value * scale * np.cos(k * np.pi * (positions - levels[0]) / height)
        else:
            field += value * j0(zeros[k] * positions / radius) / abs(j0(zeros[k]))
    return field
