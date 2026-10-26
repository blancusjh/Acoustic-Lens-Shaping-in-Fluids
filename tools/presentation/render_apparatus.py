"""Render a proposed apparatus from its declared component list.

Reads an apparatus-proposal.json (S07, S08) and the design state it names.
Quantities in the labels (volumes, channels, pitch, drive, field) are read
from the saved design, not typed in. Every drawn part is one declared component; modelled
parts use the saved design geometry and drive, proposed parts are drawn in a
separate colour and are not represented in any simulation. Nothing here has
been built. The optical path outside the cell is drawn not to scale.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import colors as mcolors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/S07-faithful-modal-synthesis/apparatus-proposal.json"
WATER, RESIN, FACE, RAY, INK, MUTED = (
    "#4f8fc0",
    "#e3a33b",
    "#9c5a08",
    "#c62828",
    "#20262c",
    "#66727c",
)
MODELLED, PROPOSED = "#1f6fb2", "#d9480f"
GLASS, PIEZO, HOUSING = "#bfe3ee", "#8e8e9e", "#d7dadd"
inputs_used = {}


def read(path):
    path = Path(path)
    inputs_used[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def load(config_path):
    from acoustic_freeform.apparatus.config import DualConfig
    from acoustic_freeform.core.paths import data_path
    from acoustic_freeform.mechanics.surface import DualSurface

    proposal = json.loads(read(config_path).read_text())
    design = data_path(proposal["design_state"])
    cfg = DualConfig(**json.loads(read(design / "config.json").read_text()))
    space = DualSurface(cfg, allow_volume_offsets=True)
    state = np.load(read(design / "state.npz"))
    regions = json.loads(read(design / "source-regions.json").read_text())["regions"]
    record = json.loads(read(design.parent / "config.json").read_text())
    record["summary"] = json.loads(read(design / "summary.json").read_text())
    components = {c["id"]: c for c in proposal["components"]}
    return (
        proposal,
        components,
        cfg,
        space,
        state["coefficients_m"],
        state["source_velocity_m_s"],
        regions,
        record,
    )


def facts(cfg, space, q, drive, regions, record):
    """Numbers quoted in the drawing, all derived from the saved design."""
    R, L = cfg.radius_m, np.asarray(cfg.levels_m)
    r = np.linspace(0, R, 4001)
    v = [float(np.trapezoid(2 * np.pi * r * space.height(j, q[j], r), r)) for j in range(2)]
    area = np.pi * R**2
    volumes = [area * (L[1] - L[0]) + v[0], area * (L[2] - L[1]) - v[0] + v[1], area * (L[3] - L[2]) - v[1]]
    count = {b: sum(1 for g in regions if g["boundary"] == b) for b in ("bottom", "top", "side")}
    window = [g for g in regions if g["boundary"] == "bottom"]
    pitch = float(np.median([g["r_max_m"] - g["r_min_m"] for g in window]))
    summary = record["summary"]
    return {
        "volumes_ul": [x * 1e9 for x in volumes],
        "counts": count,
        "channels": len(drive),
        "pitch_mm": pitch * 1e3,
        "frequency_mhz": cfg.frequency_hz / 1e6,
        "power_w": summary["source_power_w"],
        "pressure_mpa": summary["peak_pressure_pa"] / 1e6,
    }


def revolve(radius, z, first=0.0, last=np.pi, sectors=120):
    theta = np.linspace(first, last, sectors)
    x, y = np.outer(radius, np.cos(theta)), np.outer(radius, np.sin(theta))
    return pv.StructuredGrid(x, y, np.repeat(np.asarray(z)[:, None], len(theta), 1))


def tube(r_in, r_out, z0, z1, first=0.0, last=np.pi, sectors=120):
    """Closed annular solid between two radii, as a revolved rectangle outline."""
    r = np.array([r_in, r_out, r_out, r_in, r_in])
    z = np.array([z0, z0, z1, z1, z0])
    return revolve(r, z, first, last, sectors)


def render_3d(cfg, space, q, drive, regions, path, size=(1600, 1500)):
    mm = 1e3
    R, L = cfg.radius_m * mm, np.asarray(cfg.levels_m) * mm
    amp = np.abs(drive)
    norm = mcolors.Normalize(0, amp.max())
    cmap = plt.get_cmap("viridis")
    u = R / 4  # hardware drawn in proportion to the cell (laid out for the 4 mm S07 bore)
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    p.enable_anti_aliasing("ssaa")
    p.enable_lightkit()
    # Proposed housing / thermal jacket, cut open.
    p.add_mesh(tube(R + 1.6 * u, R + 2.4 * u, L[0] - 2.2 * u, L[3] + 2.2 * u), color=HOUSING, opacity=0.35)
    # Proposed piezoelectric wall tube; its electrode bands carry the modelled wall command.
    p.add_mesh(tube(R, R + u, L[0], L[3]), color=PIEZO, opacity=0.25)
    for reg in regions:
        colour = cmap(norm(amp[reg["channel"]]))[:3]
        if reg["boundary"] == "side":
            band = revolve(np.full(2, R + 0.005 * u), np.array([reg["z_min_m"], reg["z_max_m"]]) * mm)
            p.add_mesh(band, color=colour)
        else:
            z = reg["z_min_m"] * mm + (-0.005 if reg["boundary"] == "bottom" else 0.005) * u
            ring = revolve(np.array([reg["r_min_m"], reg["r_max_m"]]) * mm, np.full(2, z))
            p.add_mesh(ring, color=colour, opacity=0.8)
    # Proposed transparent transducer windows outside the modelled source planes.
    for z0, z1 in ((L[0] - u, L[0]), (L[3], L[3] + u)):
        p.add_mesh(tube(0.0, R + u, z0, z1), color=GLASS, opacity=0.35)
    # Modelled fluids and both faces from the saved design state.
    r = np.linspace(0, cfg.radius_m, 220)
    faces = [(cfg.levels_m[j + 1] + space.height(j, q[j], r)) * mm for j in range(2)]
    signed = np.r_[-r[::-1] * 0.985, r * 0.985] * mm
    bounds = [np.full_like(r, L[0]), faces[0], faces[1], np.full_like(r, L[3])]
    for j, colour in enumerate([WATER, RESIN, WATER]):
        low, high = bounds[j], bounds[j + 1]
        strip = np.stack([np.r_[low[::-1], low], np.r_[high[::-1], high]])
        x = np.broadcast_to(signed, strip.shape)
        cap = pv.StructuredGrid(
            x[..., None].copy(), np.zeros_like(x)[..., None], strip[..., None].copy()
        )
        p.add_mesh(cap, color=colour, opacity=0.85)
    for z in faces:
        p.add_mesh(
            revolve(r * 0.985 * mm, z, sectors=180),
            color="#e8b25c",
            opacity=0.5,
            smooth_shading=True,
        )
    # Pinning edges (modelled contact lines; hardware edge proposed).
    theta = np.linspace(0, np.pi, 180)
    for z in L[1:3]:
        edge = np.c_[(R - 0.03 * u) * np.cos(theta), (R - 0.03 * u) * np.sin(theta), np.full_like(theta, z)]
        p.add_mesh(pv.lines_from_points(edge).tube(radius=0.05 * u), color="black")
    # Proposed fill and vent ports, one per layer, through the wall and housing.
    for z in (L[0] + L[1]) / 2, (L[1] + L[2]) / 2, (L[2] + L[3]) / 2:
        p.add_mesh(
            pv.Cylinder(center=(R + 1.7 * u, 0.6 * u, z), direction=(1, 0, 0), radius=0.18 * u, height=3.4 * u),
            color=PROPOSED,
        )
    centre = (L[0] + L[3]) / 2
    p.camera_position = [(17 * u, -27 * u, centre + 15 * u), (0, 0, centre - 0.5 * u), (0, 0, 1)]
    p.camera.view_angle = 30
    p.screenshot(path)
    p.close()


def autocrop(image, margin=10):
    mask = np.any(image[..., :3] < 0.98, axis=-1)
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return image[
        max(rows[0] - margin, 0) : rows[-1] + margin, max(cols[0] - margin, 0) : cols[-1] + margin
    ]


def section(ax, cfg, space, q, drive, regions, known):
    mm = 1e3
    R, L = cfg.radius_m * mm, np.asarray(cfg.levels_m) * mm
    u, centre = R / 4, (L[0] + L[3]) / 2
    r = np.linspace(0, cfg.radius_m, 801)
    signed = np.r_[-r[:0:-1], r] * mm
    mirror = lambda h: np.r_[h[:0:-1], h]
    faces = [(cfg.levels_m[j + 1] + space.height(j, q[j], r)) * mm for j in range(2)]
    layers = [
        np.full_like(signed, L[0]),
        mirror(faces[0]),
        mirror(faces[1]),
        np.full_like(signed, L[3]),
    ]
    for j, colour in enumerate([WATER, RESIN, WATER]):
        ax.fill_between(
            signed, layers[j], layers[j + 1], color=colour, alpha=0.3 if j != 1 else 0.45, lw=0
        )
    for j in range(2):
        ax.plot(signed, mirror(faces[j]), color=FACE, lw=1.8)
    # Proposed hardware around the modelled domain.
    for sgn in (1, -1):
        x0 = R if sgn > 0 else -R - u
        ax.add_patch(Rectangle((x0, L[0]), u, L[3] - L[0], color=PIEZO, alpha=0.5, lw=0))
        hx = R + 1.6 * u if sgn > 0 else -R - 2.4 * u
        ax.add_patch(Rectangle((hx, L[0] - 2.2 * u), 0.8 * u, L[3] - L[0] + 4.4 * u, color=HOUSING, lw=0))
    for z0 in (L[0] - u, L[3]):
        ax.add_patch(Rectangle((-R - u, z0), 2 * R + 2 * u, u, color=GLASS, alpha=0.8, lw=0))
    amp = np.abs(drive)
    norm = mcolors.Normalize(0, amp.max())
    cmap = plt.get_cmap("viridis")
    for reg in regions:
        c = cmap(norm(amp[reg["channel"]]))
        if reg["boundary"] == "side":
            for x in (R - 0.08 * u, -R):
                ax.add_patch(
                    Rectangle(
                        (x, reg["z_min_m"] * mm),
                        0.08 * u,
                        (reg["z_max_m"] - reg["z_min_m"]) * mm,
                        color=c,
                        lw=0,
                    )
                )
        else:
            z = reg["z_min_m"] * mm - (0.08 * u if reg["boundary"] == "bottom" else 0)
            for a, b in ((reg["r_min_m"], reg["r_max_m"]), (-reg["r_max_m"], -reg["r_min_m"])):
                ax.add_patch(Rectangle((a * mm, z), (b - a) * mm, 0.08 * u, color=c, lw=0))
    for z in L[1:3]:
        ax.plot([-R, R], [z, z], "k^", ms=6)
    ports = (L[0] + L[1]) / 2, (L[1] + L[2]) / 2, (L[2] + L[3]) / 2
    for z in ports:
        ax.add_patch(Rectangle((R + 0.1 * u, z - 0.15 * u), 2.3 * u, 0.3 * u, color=PROPOSED, lw=0))
    n, vol = known["counts"], known["volumes_ul"]
    # Text columns sit beside the drawing; rows are spread over its height.
    left, right = -9.6 * u, 6.6 * u
    rows = np.linspace(L[3] + 2.3 * u, L[0] - 2.3 * u, 7)
    callouts = [
        ((-R - 0.5 * u, L[3] + 0.5 * u), "transparent ring-electrode\ntransducer window (proposed)", PROPOSED,
         (left, rows[0])),
        ((R + 0.5 * u, L[3] - 0.3 * u), f"segmented piezo tube,\n{n['side']} electrode bands (proposed)", PROPOSED,
         (right, rows[0])),
        ((R + 2.0 * u, L[0] - 1.6 * u), "thermal enclosure\n(proposed)", PROPOSED, (right, rows[6])),
        ((R + 2.0 * u, ports[1]), "fill / vent ports\n(proposed)", PROPOSED, (right, rows[3])),
        ((R, L[2]), "pinned contact lines\n(modelled; edge proposed)", MODELLED, (right, rows[1])),
        ((0, L[0] - 0.05 * u), f"{n['bottom']} source rings per window,\n{known['pitch_mm']:.3f} mm pitch (modelled)",
         MODELLED, (left, rows[6])),
        ((-0.4 * R, ports[1]), f"NOA 61, {vol[1]:.2f} µL (modelled)", MODELLED, (left, rows[3])),
        ((-0.6 * R, ports[0]), f"water, {vol[0]:.2f} µL (modelled)", MODELLED, (left, rows[5])),
        ((-0.6 * R, ports[2]), f"water, {vol[2]:.2f} µL (modelled)", MODELLED, (left, rows[1])),
    ]
    for xy, text, colour, at in callouts:
        ax.annotate(
            text,
            xy,
            at,
            fontsize=8.5,
            color=colour,
            va="center",
            arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.7},
        )
    half = max(6.5 * u, (L[3] - L[0]) / 2 + 3.0 * u)
    ax.set(xlim=(-10 * u, 10 * u), ylim=(centre - half, centre + half), aspect="equal", xlabel="x (mm)",
           ylabel="z (mm)")
    ax.set_title(
        "(b) Section at equal scale: modelled (blue labels) and proposed (orange) parts",
        loc="left",
        fontsize=11,
    )


def optical_path(ax, cfg, record):
    z0, z1, z2 = np.array([float(z) for z in record["optics"]["stigmatic_z_m"]]) * 1e3
    L = np.asarray(cfg.levels_m) * 1e3
    collimated = np.isinf(z0)
    source = ("collimated beam: LED + pinhole\n+ collimating lens (proposed)" if collimated
              else f"point source: pinhole + LED\non axis at z = {z0:.0f} mm\n(proposed)")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.5, 10.5)
    ax.axis("off")
    stations = [
        (0.8, source, PROPOSED),
        (5.0, f"cell, z = {L[0]:.2f} … {L[3]:.2f} mm\n(modelled interior)", MODELLED),
        (9.2, f"camera at image plane z = {z2:.2f} mm\n+ wavefront sensor (proposed)", PROPOSED),
    ]
    for y, text, colour in stations:
        ax.add_patch(
            FancyBboxPatch(
                (-4.6, y - 0.7), 9.2, 1.4, boxstyle="round,pad=0.1", fc="white", ec=colour, lw=1.4
            )
        )
        ax.text(0, y, text, ha="center", va="center", fontsize=8.5, color=colour)
    for a, b in ((1.6, 4.2), (5.8, 8.4)):
        ax.add_patch(
            FancyArrowPatch((0, a), (0, b), arrowstyle="-|>", mutation_scale=12, color=RAY)
        )
    ax.text(0.3, 2.9, "parallel to the axis" if collimated else f"{L[0] - z0:.0f} mm, object side",
            fontsize=8, color=MUTED, va="center")
    ax.text(
        0.3,
        7.1,
        f"{z2 - L[3]:.2f} mm to the image\n(intermediate conjugate {z1:.1f} mm)",
        fontsize=8,
        color=MUTED,
        va="center",
    )
    ax.set_title("(c) Optical axis, not to scale", loc="left", fontsize=11)


def block_diagram(ax, known):
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    w, h = 3.6, 1.2
    boxes = {
        "computer": (8.0, 7.0, "control computer\n(feedback law not designed)", PROPOSED),
        "rf": (3.0, 4.3, (f"{known['channels']}-channel CW transmitter\n"
                          f"{known['frequency_mhz']:.2f} MHz, amplitude + phase"), PROPOSED),
        "temp": (8.0, 4.3, "temperature controller\nPeltier + thermistor", PROPOSED),
        "pumps": (13.0, 4.3, "syringe pumps\nfill and trim", PROPOSED),
        "cell": (
            8.0,
            1.2,
            (f"cell: {known['channels']} source surfaces, three liquids\n"
             f"(modelled; {known['power_w']:.2f} W, {known['pressure_mpa']:.2f} MPa)"),
            MODELLED,
        ),
        "sensor": (14.0, 1.2, "camera +\nwavefront sensor", PROPOSED),
    }
    for x, y, text, colour in boxes.values():
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.08",
                fc="white",
                ec=colour,
                lw=1.4,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color=colour)

    def arrow(start, end, **style):
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=0.9, **style
            )
        )

    top = lambda k: (boxes[k][0], boxes[k][1] + h / 2 + 0.08)
    bottom = lambda k: (boxes[k][0], boxes[k][1] - h / 2 - 0.08)
    for k in ("rf", "temp", "pumps"):
        arrow(bottom("computer"), top(k))
        arrow(bottom(k), top("cell"))
    cx, cy = boxes["cell"][:2]
    sx, sy = boxes["sensor"][:2]
    arrow((cx + w / 2 + 0.08, cy), (sx - w / 2 - 0.08, sy))
    ax.text((cx + sx) / 2, cy + 0.2, "light", ha="center", fontsize=7.5, color=RAY)
    arrow(
        top("sensor"),
        (boxes["computer"][0] + w / 2 + 0.08, boxes["computer"][1]),
        connectionstyle="angle,angleA=90,angleB=0,rad=0",
    )
    ax.text(15.4, 5.6, "measured spot\nand wavefront", fontsize=7.5, color=MUTED, ha="center")
    ax.set_title("(d) Signal and control chain", loc="left", fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/studies/S07-faithful-modal-synthesis/meniscus-2026-10-25/figures",
    )
    parser.add_argument("--prefix", default="s07", help="file name prefix")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    proposal, components, cfg, space, q, drive, regions, record = load(args.config.resolve())
    known = facts(cfg, space, q, drive, regions, record)
    render = args.out / "_apparatus.png"
    render_3d(cfg, space, q, drive, regions, render)
    fig = plt.figure(figsize=(22, 12))
    g = fig.add_gridspec(
        2, 3, width_ratios=[1.1, 1.25, 0.8], height_ratios=[1, 0.85], wspace=0.15, hspace=0.12
    )
    ax = fig.add_subplot(g[:, 0])
    ax.imshow(autocrop(plt.imread(render)))
    ax.axis("off")
    ax.set_title(
        "(a) Proposed cell, half cut away, equal scale\nsource rings and wall bands coloured by commanded amplitude",
        loc="left",
        fontsize=11,
    )
    section(fig.add_subplot(g[0, 1]), cfg, space, q, drive, regions, known)
    optical_path(fig.add_subplot(g[0, 2]), cfg, record)
    block_diagram(fig.add_subplot(g[1, 1:]), known)
    amp = np.abs(drive)
    bar = fig.colorbar(
        plt.cm.ScalarMappable(mcolors.Normalize(0, amp.max()), "viridis"),
        ax=fig.axes[0],
        orientation="horizontal",
        fraction=0.03,
        pad=0.02,
        shrink=0.7,
    )
    bar.set_label("commanded source velocity amplitude (m/s)", fontsize=9)
    fig.text(
        0.01,
        0.01,
        f"Proposal only: no part has been built or measured. Blue: in the {proposal['study']} numerical model "
        "with these values. "
        "Orange: required hardware not yet simulated (window optics, transducer coupling, thermal expansion, feedback).",
        fontsize=9.5,
        color=MUTED,
    )
    fig.savefig(args.out / f"{args.prefix}-apparatus.png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    render.unlink()
    (args.out / "apparatus-provenance.json").write_text(
        json.dumps(
            {
                "inputs_sha256": inputs_used,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "components": {k: v["status"] for k, v in components.items()},
                "quoted_from_design": known,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
