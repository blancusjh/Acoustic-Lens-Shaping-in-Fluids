"""Render the S09 atlas: which stigmatic singlets one common machine can form.

Reads the atlas and escalation records under
artifacts/studies/S09-universal-source/atlas-2026-10-26/ and draws every lens
of the grid as its cross-section in the common cell, coloured by the least
source freedom whose capability reaches the 10 nm pupil goal within the caps
(1 W unless stated, 2 MPa per band). Capability values are first-order,
frozen-compliance results of the convex program and its one-tone-per-band
refinement; selected lenses are confirmed by the nonlinear chain in the study.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/analysis/universal"))
from common import lens_cell

BASE = ROOT / "artifacts/studies/S09-universal-source/atlas-2026-10-26"
CLASSES = [
    ("one band (1.5 MHz)", "#1b7837", [("atlas-1p5mhz-1w.jsonl", {})]),
    ("two bands (0.75 + 1.5 MHz)", "#5aae61", [("e3-0p75-1p5mhz-26modes.jsonl", {})]),
    ("three bands (0.75, 1.0, 1.5 MHz)", "#a6dba0", [("e5-0p75-1p0-1p5mhz.jsonl", {})]),
    ("three bands, two tones each", "#d9f0d3", [("e6-3band-2tones.jsonl", {})]),
    ("three bands at 3 W", "#fee08b", [("e7-3band-3w.jsonl", {"cap": 3.0})]),
]
NOT_COVERED, NO_FIT = ("not reached", "#d73027"), ("does not fit the common cell", "#bdbdbd")
inputs_used = {}


def records(name):
    path = BASE / name
    if not path.exists():
        return {}
    inputs_used[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {r["index"]: r for r in map(json.loads, path.read_text().splitlines())}


def covered(row, cap):
    return (
        row is not None
        and row.get("status") not in (None, "skipped")
        and row["single_tone_pupil_error_m"] <= 1e-8
        and row["single_tone_peak_pressure_pa"] <= 2e6 * 1.0001
        and row["single_tone_power_w"] <= cap * 1.0001
    )


def classify(index, base, escalations):
    if base[index]["status"] == "skipped":
        return NO_FIT
    for (label, colour, files), tables in zip(CLASSES, escalations):
        for (_, options), table in zip(files, tables):
            if covered(table.get(index), options.get("cap", 1.0)):
                return label, colour
    return NOT_COVERED


def section(ax, row, colour):
    singlet = {
        "indices": [1.0, 1.7, 1.0],
        "object_m": row["object_mm"] if isinstance(row["object_mm"], str) else row["object_mm"] * 1e-3,
        "image_m": row["image_mm"] * 1e-3, "thickness_m": row["thickness_mm"] * 1e-3,
        "shape_factor": row["shape_factor"], "scale": 1 / 14, "pin_radius_m": 0.00202,
        "clear_radius_m": 0.002, "windows_m": [-0.0008, 0.0022286],
    }
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.9, 2.35)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(colour)
        side.set_linewidth(2.2)
    ax.set_facecolor(colour + "33")
    try:
        cfg, faces, _ = lens_cell("fig9b", singlet=singlet)
    except Exception:  # noqa: BLE001 - lenses without a real vertex branch are drawn empty
        ax.text(0, 0.7, "no real\nbranch", ha="center", va="center", fontsize=6, color="#555")
        return
    r = np.linspace(0, cfg.radius_m, 200)
    signed = np.r_[-r[::-1], r] * 1e3
    z = [np.r_[h[::-1], h] * 1e3 for h in ((cfg.levels_m[j + 1] + faces[j].evaluate(r)) for j in range(2))]
    ax.fill_between(signed, z[0], z[1], color="#e3a33b", alpha=0.85, lw=0)
    ax.plot(signed, z[0], color="#7a4a06", lw=0.7)
    ax.plot(signed, z[1], color="#7a4a06", lw=0.7)
    ax.axhline(cfg.levels_m[0] * 1e3, color="#555", lw=0.5)
    ax.axhline(cfg.levels_m[3] * 1e3, color="#555", lw=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = records("atlas-1p5mhz-1w.jsonl")
    escalations = [[records(name) for name, _ in files] for _, _, files in CLASSES]
    objects, images = ["-inf", -300.0, -150.0], [60.0, 90.0, 120.0]
    thicknesses, shapes = [10.0, 15.0, 20.0], [-0.5, 0.0, 0.5]
    fig = plt.figure(figsize=(17, 16.5))
    outer = fig.add_gridspec(3, 3, hspace=0.16, wspace=0.08)
    counts = {}
    for a, d0 in enumerate(objects):
        for b, d2 in enumerate(images):
            inner = outer[a, b].subgridspec(3, 3, hspace=0.06, wspace=0.06)
            for i, t in enumerate(thicknesses):
                for j, s in enumerate(shapes):
                    index = next(k for k, r in base.items() if r["object_mm"] == d0 and r["image_mm"] == d2
                                 and r["thickness_mm"] == t and r["shape_factor"] == s)
                    label, colour = classify(index, base, escalations)
                    counts[label] = counts.get(label, 0) + 1
                    ax = fig.add_subplot(inner[i, j])
                    section(ax, base[index], colour)
                    if i == 0:
                        ax.set_title(f"σ = {s:+.1f}", fontsize=8)
                    if j == 0:
                        ax.set_ylabel(f"t = {t:.0f}", fontsize=8)
                    ax.text(2.1, 2.25, str(index), ha="right", va="top", fontsize=6, color="#333")
            box = outer[a, b].get_position(fig)
            obj = "object at infinity" if d0 == "-inf" else f"object at {d0:.0f} mm"
            fig.text(box.x0, box.y1 + 0.004, f"{obj}, image at {d2:.0f} mm", fontsize=10, weight="bold")
    order = [c[0] for c in CLASSES] + [NOT_COVERED[0], NO_FIT[0]]
    colours = dict([(c[0], c[1]) for c in CLASSES] + [NOT_COVERED, NO_FIT])
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colours[k] + "88", ec=colours[k], lw=2) for k in order]
    fig.legend(handles, [f"{k} ({counts.get(k, 0)})" for k in order], loc="lower center", ncol=4,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("One common machine, many stigmatic singlets: least source freedom that forms each lens",
                 fontsize=15, y=0.925)
    fig.text(0.5, 0.012,
             "Silva-Lora and Torres singlets (n = 1 / 1.7 / 1; distances, thickness t in paper mm; σ shape factor), "
             "scaled 1/14 to a 4 mm aperture, each drawn at equal scale in the common water / NOA 61 / water cell "
             "(windows at -0.8 and +2.23 mm). Same 26 continuous source modes for every lens; 10 nm pupil goal; "
             "≤ 2 MPa per band. First-order convex capability with one-tone-per-band refinement; numerical model, "
             "not an experiment.", ha="center", fontsize=8.5, color="#555", wrap=True)
    fig.savefig(args.out / "s09-atlas.png", dpi=110, bbox_inches="tight")
    (args.out / "atlas-provenance.json").write_text(json.dumps({
        "inputs_sha256": inputs_used,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": counts,
    }, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
