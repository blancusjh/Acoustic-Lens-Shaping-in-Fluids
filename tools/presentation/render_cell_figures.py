"""Render a modelled two-face cell, its design and formation from saved states.

Reads a faithful-synthesis run (S07 shared-conjugate patches or an S08
prescribed singlet) and its forward start-up trajectory, recomputes every
displayed field, ray and spot from the saved geometry and command, and writes
figures, videos and input hashes to the output directory. Every drawn part
is a declared component of the numerical model; nothing here is an
experiment. Geometric exaggeration and playback speed are labelled.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyvista as pv
from matplotlib import colors as mcolors

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/studies/S07-faithful-modal-synthesis/meniscus-2026-10-25"
WATER, RESIN, FACE, RAY, INK, MUTED = "#4f8fc0", "#e3a33b", "#9c5a08", "#c62828", "#20262c", "#66727c"
SOURCE_MAP = "viridis"
inputs_used = {}


def read(path):
    path = Path(path)
    inputs_used[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def load(run, formation):
    from acoustic_freeform.acoustics.helmholtz import DualAcoustics
    from acoustic_freeform.apparatus.config import DualConfig
    from acoustic_freeform.core.paths import data_path
    from acoustic_freeform.mechanics.surface import DualSurface, targets_from_record

    record = json.loads(read(run / "config.json").read_text())
    cfg = DualConfig(**json.loads(read(run / "design/config.json").read_text()))
    space = DualSurface(cfg, allow_volume_offsets=True)
    material = json.loads(read(data_path(record["material_reference"])).read_text())
    targets = targets_from_record(cfg, record)
    state = np.load(read(run / "design/state.npz"))
    q, drive = state["coefficients_m"], state["source_velocity_m_s"]
    modal = np.load(read(run / "best-source.npz"))["modal_coefficients_m_s"]
    labels = json.loads(read(run / "modes.json").read_text())
    summary = json.loads(read(run / "design/summary.json").read_text())
    trajectory = np.load(read(formation / "trajectory.npz"))
    formation_config = json.loads(read(formation / "config.json").read_text())
    start = space.unforced_equilibrium()
    if np.max(abs(trajectory["coefficients_m"][0] - start)) > 1e-12:
        raise ValueError("The saved trajectory does not start at the acoustics-off meniscus")
    wave = DualAcoustics(cfg, space, linear_solver="static_condensed")
    response = wave.solve(q, drive)
    xy = response.basis.mapping.geometry(wave.flat.p)[0] * cfg.radius_m
    field = {
        "xy": xy,
        "triangles": wave.flat.t.T,
        "pressure": response.solution[response.basis.nodal_dofs[0], 0],
        "diagnostics": response.diagnostics(np.ones(1, complex)),
    }
    return {
        "record": record, "cfg": cfg, "space": space, "material": material,
        "targets": targets, "q": q, "drive": drive, "modal": modal, "labels": labels,
        "summary": summary, "trajectory": trajectory, "formation": formation_config,
        "start": start, "field": field, "optics": record["optics"],
        "conjugates": [float(z) for z in record["optics"]["stigmatic_z_m"]],
    }


def trace(data, q, rays=401):
    from acoustic_freeform.optics.raytrace import trace_pair

    cfg, optics = data["cfg"], data["optics"]
    r = np.linspace(0, cfg.clear_radius_m, rays)
    return trace_pair(
        data["space"], q, optics["indices"], float(optics["stigmatic_z_m"][0]),
        float(optics["stigmatic_z_m"][2]), launch_radius_m=r,
        launch_height_m=cfg.levels_m[1] + data["targets"][0].evaluate(r),
    )


def ray_paths(data, q, launch):
    from acoustic_freeform.optics.raytrace import trace_pair

    cfg, optics = data["cfg"], data["optics"]
    z0, _, z2 = data["conjugates"]
    paths = []
    for rho in launch:
        point = np.array([abs(rho)])
        tr = trace_pair(data["space"], q, optics["indices"], z0, z2, launch_radius_m=point,
                        launch_height_m=cfg.levels_m[1] + data["targets"][0].evaluate(point))
        hits = tr["intersections_m"][:, 0, :]
        if np.isinf(z0):
            entry = np.array([hits[0][0], cfg.levels_m[0]])
        else:
            d = hits[0] - np.array([0.0, z0])
            entry = np.array([0.0, z0]) + (cfg.levels_m[0] - z0) / d[1] * d
        out = tr["spots_m"][0, 0]
        top = image_top(data)
        direction = np.array([out, z2]) - hits[1]
        end = hits[1] + (top - hits[1][1]) / direction[1] * direction
        path = np.vstack([entry, hits[0], hits[1], end])
        path[:, 0] *= np.sign(rho) if rho else 1
        paths.append(path)
    return paths


def image_top(data):
    """Rays are drawn to a near image plane, or 0.2 cell heights past the cell."""
    levels, z2 = data["cfg"].levels_m, data["conjugates"][2]
    height = levels[3] - levels[0]
    return z2 + 0.05 * height if z2 < levels[3] + 3 * height else levels[3] + 0.2 * height


def faces_mm(data, q, r):
    cfg = data["cfg"]
    return [(cfg.levels_m[j + 1] + data["space"].height(j, q[j], r)) * 1e3 for j in range(2)]


def source_amplitudes(data):
    from acoustic_freeform.apparatus.sources import modal_source_profile

    cfg = data["cfg"]
    r = np.linspace(0, cfg.radius_m, 200)
    z = np.linspace(cfg.levels_m[0], cfg.levels_m[-1], 200)
    return {
        "r": r, "z": z,
        "bottom": modal_source_profile(cfg, data["labels"], data["modal"], "bottom", r),
        "top": modal_source_profile(cfg, data["labels"], data["modal"], "top", r),
        "side": modal_source_profile(cfg, data["labels"], data["modal"], "side", z),
    }


def revolve(radius, z, first=0.0, last=np.pi, sectors=120):
    theta = np.linspace(first, last, sectors)
    x, y = np.outer(radius, np.cos(theta)), np.outer(radius, np.sin(theta))
    return pv.StructuredGrid(x, y, np.repeat(np.asarray(z)[:, None], len(theta), 1))


def render_3d(data, sources, paths, path, size=(1500, 1350)):
    """Half cell (y >= 0) so the meridional section faces the camera; equal scale."""
    cfg, mm = data["cfg"], 1e3
    levels = np.asarray(cfg.levels_m)
    r = np.linspace(0, cfg.radius_m, 220)
    inner = r * 0.985
    faces = faces_mm(data, data["q"], r)
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    p.enable_anti_aliasing("ssaa")
    p.enable_lightkit()
    peak = max(np.max(abs(sources[k])) for k in ("bottom", "top", "side"))
    cmap = plt.get_cmap(SOURCE_MAP)
    for boundary, z in (("bottom", levels[0]), ("top", levels[3])):
        amplitude = abs(sources[boundary])
        for i in range(len(sources["r"]) - 1):
            colour = cmap(amplitude[i] / peak)[:3]
            ring = revolve(sources["r"][i:i + 2] * mm, np.full(2, z * mm))
            p.add_mesh(ring, color=colour, opacity=0.55 if boundary == "top" else 1.0)
    amplitude = abs(sources["side"])
    for i in range(len(sources["z"]) - 1):
        band = revolve(np.full(2, cfg.radius_m * mm), sources["z"][i:i + 2] * mm)
        p.add_mesh(band, color=cmap(amplitude[i] / peak)[:3], opacity=0.9)
    signed = np.r_[-inner[::-1], inner] * mm
    bounds = [np.full_like(r, levels[0] * mm), faces[0], faces[1], np.full_like(r, levels[3] * mm)]
    for j, colour in enumerate([WATER, RESIN, WATER]):
        low, high = bounds[j], bounds[j + 1]
        strip = np.stack([np.r_[low[::-1], low], np.r_[high[::-1], high]])
        x = np.broadcast_to(signed, strip.shape)
        cap = pv.StructuredGrid(x[..., None].copy(), np.zeros_like(x)[..., None], strip[..., None].copy())
        p.add_mesh(cap, color=colour, opacity=0.9, ambient=0.5, diffuse=0.6)
    for z in faces:
        p.add_mesh(revolve(inner * mm, z, sectors=180), color="#e8b25c", opacity=0.55,
                   smooth_shading=True, specular=0.8, specular_power=60)
        line = np.c_[signed, np.full_like(signed, -0.01), np.r_[z[::-1], z]]
        p.add_mesh(pv.lines_from_points(line).tube(radius=0.005 * cfg.radius_m * mm), color=FACE)
    for ray in paths:
        xyz = np.c_[ray[:, 0] * mm, np.full(len(ray), -0.0075 * cfg.radius_m * mm), ray[:, 1] * mm]
        p.add_mesh(pv.lines_from_points(xyz).tube(radius=0.003 * cfg.radius_m * mm, n_sides=10), color=RAY,
                   ambient=0.7, diffuse=0.4)
    u, centre = cfg.radius_m * mm / 4, np.mean(levels[[0, 3]]) * mm
    p.camera_position = [(10 * u, -19 * u, centre + 14 * u), (0, 0, centre - 0.3 * u), (0, 0, 1)]
    p.camera.view_angle = 26
    p.screenshot(path)
    p.close()


def autocrop(image, margin=12):
    mask = np.any(image[..., :3] < 0.98, axis=-1)
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return image[max(rows[0] - margin, 0):rows[-1] + margin, max(cols[0] - margin, 0):cols[-1] + margin]


def setup_figure(data, out, prefix):
    cfg, mm = data["cfg"], 1e3
    sources = source_amplitudes(data)
    launch = np.linspace(-cfg.clear_radius_m, cfg.clear_radius_m, 9)
    paths = ray_paths(data, data["q"], launch)
    render = out / "_render.png"
    render_3d(data, sources, paths, render)
    levels = np.asarray(cfg.levels_m) * mm
    R = cfg.radius_m * mm
    r = np.linspace(0, cfg.radius_m, 1601)
    signed = np.r_[-r[:0:-1], r] * mm
    mirror = lambda h: np.r_[h[:0:-1], h]
    faces = faces_mm(data, data["q"], r)
    peak = max(np.max(abs(sources[k])) for k in ("bottom", "top", "side"))
    norm = mcolors.Normalize(0, peak)
    cmap = plt.get_cmap(SOURCE_MAP)
    n, rho = data["optics"]["indices"], cfg.density_kg_m3
    volumes = np.array(cfg.volume_offsets_m3) * 1e9  # µL
    u = R / 4  # drawing unit: offsets below were laid out for the 4 mm S07 cell
    w = 0.22 * u  # source strip width

    fig = plt.figure(figsize=(22, 9))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.0, 0.75, 0.8], wspace=0.28)
    ax = fig.add_subplot(grid[0])
    ax.imshow(autocrop(plt.imread(render)))
    ax.set_axis_off()
    ax.set_title("(a) Modelled cell, front half removed, equal scale", loc="left", fontsize=12)

    ax = fig.add_subplot(grid[1])
    layers = [np.full_like(signed, levels[0]), mirror(faces[0]), mirror(faces[1]),
              np.full_like(signed, levels[3])]
    for j, colour in enumerate([WATER, RESIN, WATER]):
        ax.fill_between(signed, layers[j], layers[j + 1], color=colour, alpha=0.25 if j != 1 else 0.4, lw=0)
    for boundary, z, dz in (("bottom", levels[0], -w), ("top", levels[3], 0.0)):
        for i in range(len(sources["r"]) - 1):
            c = cmap(norm(abs(sources[boundary][i])))
            for sgn in (1, -1):
                x0, x1 = sorted((sgn * sources["r"][i] * mm, sgn * sources["r"][i + 1] * mm))
                ax.add_patch(plt.Rectangle((x0, z + dz), x1 - x0, w, color=c, lw=0))
    for i in range(len(sources["z"]) - 1):
        c = cmap(norm(abs(sources["side"][i])))
        z0, z1 = sources["z"][i] * mm, sources["z"][i + 1] * mm
        for x in (R, -R - w):
            ax.add_patch(plt.Rectangle((x, z0), w, z1 - z0, color=c, lw=0))
    for j in range(2):
        ax.plot(signed, mirror(faces[j]), color=FACE, lw=2)
        ax.plot([-R, R], [levels[j + 1]] * 2, color=MUTED, lw=0.6, ls=":")
        ax.plot([-R, R], [levels[j + 1]] * 2, "o", color="black", ms=5)
    for ray in paths:
        ax.plot(ray[:, 0] * mm, ray[:, 1] * mm, color=RAY, lw=0.8)
    note = {"fontsize": 9, "color": INK}
    # Optical indices are the lens prescription's; for a published singlet they
    # are the cured lens in air, not the liquids being shaped.
    singlet = "singlet" in data["record"]
    names = ("water", "NOA 61 (uncured)", "water")
    middle = (faces[0][-1] + faces[1][-1]) / 2
    top = image_top(data) * mm
    for j, z in enumerate((levels[0] + 0.12 * u, middle, levels[3] - 0.45 * u)):
        index = f"  n={n[j]}" if not singlet else ""
        ax.text(-R + 0.3 * u, z, f"{names[j]}{index}  ρ={rho[j]:.0f} kg/m³", va="center", **note)
    ax.text(R + 0.3 * u, levels[1] + 0.15 * u, "pinned\nrims", ha="left", fontsize=8, color=MUTED)
    ax.text(0.15 * u, faces[0][0] - 0.25 * u, f"front face, displaced volume {volumes[0]:+.3f} µL",
            fontsize=8.5, color=FACE)
    ax.text(0.15 * u, faces[1][0] + 0.12 * u, f"back face, displaced volume {volumes[1]:+.3f} µL",
            fontsize=8.5, color=FACE)
    z0, z1, z2 = np.array(data["conjugates"]) * mm
    source = "collimated input (object at infinity)" if np.isinf(z0) else f"rays from on-axis object at z = {z0:.1f} mm"
    ax.text(0, levels[0] - 0.55 * u, source, ha="center", va="top", fontsize=8.5, color=RAY)
    ax.text(0, top + 0.2 * u, f"to image at z = {z2:.2f} mm; intermediate conjugate {z1:.1f} mm",
            ha="center", fontsize=8.5, color=RAY)
    ax.text(0, levels[0] - 1.25 * u, f"cylinder Ø {2 * R:.2f} mm, height {levels[3] - levels[0]:.2f} mm, "
            f"clear apertures Ø {2 * cfg.clear_radius_m * mm:.1f} mm", ha="center", fontsize=8.5, color=MUTED)
    if singlet:
        ax.text(0, levels[0] - 1.6 * u, f"rays traced with the published lens indices {n[0]:g} / {n[1]:g} / {n[2]:g}",
                ha="center", fontsize=8.5, color=RAY)
    ax.set(xlim=(-R - 0.5 * u, R + 1.0 * u), ylim=(levels[0] - 1.9 * u, max(levels[3] + 2.0 * u, top + 0.6 * u)),
           aspect="equal", xlabel="x (mm)", ylabel="z (mm)")
    ax.set_title("(b) Meridional section, equal scale", loc="left", fontsize=12)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=ax, orientation="horizontal",
                       fraction=0.035, pad=0.09, shrink=0.85)
    bar.set_label("source velocity amplitude |u| on windows and wall (m/s)", fontsize=9)

    sub = grid[2].subgridspec(3, 1, hspace=0.55)
    for k, (boundary, axis, label) in enumerate((("bottom", "r", "lower window"), ("top", "r", "upper window"),
                                                 ("side", "z", "cylinder wall"))):
        a = fig.add_subplot(sub[k])
        x = sources[axis] * mm
        a.plot(x, abs(sources[boundary]) * 1e3, color=INK, lw=1.4)
        a.set_ylabel("|u| (mm/s)", fontsize=8.5)
        b = a.twinx()
        b.plot(x, np.angle(sources[boundary]), color=MUTED, lw=0.8, ls="--")
        b.set_ylim(-np.pi, np.pi)
        b.set_ylabel("phase (rad)", fontsize=8, color=MUTED)
        a.set_xlabel("radius (mm)" if axis == "r" else "z (mm)", fontsize=8.5)
        a.set_title(f"({'cde'[k]}) {label}", loc="left", fontsize=10)
    fig.text(0.585, 0.95, f"Continuous source: {len(data['labels'])} low-order modes", fontsize=11)

    ax = fig.add_subplot(grid[3])
    field = data["field"]
    xy = field["xy"] * mm
    tri = mtri.Triangulation(xy[0], xy[1], field["triangles"])
    pressure = field["pressure"].real / 1e6
    limit = float(np.max(abs(field["pressure"]))) / 1e6
    image = ax.tripcolor(tri, pressure, shading="gouraud", cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
    for j in range(2):
        ax.plot(r * mm, faces[j], color="black", lw=1)
    ax.set(xlim=(0, R), ylim=(levels[0], levels[3]), aspect="equal", xlabel="radius (mm)", ylabel="z (mm)")
    ax.set_title(f"(f) Pressure at one instant, {cfg.frequency_hz / 1e6:.2f} MHz", loc="left", fontsize=12)
    cb = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.035, pad=0.09, shrink=0.9)
    cb.set_label("acoustic pressure (MPa)", fontsize=9)
    d = field["diagnostics"]
    fig.text(0.01, 0.015,
             "Numerical model, not an experiment: axisymmetric three-fluid cell, lossless longitudinal acoustics "
             "with passive ideal source boundaries, capillarity and gravity. Source power "
             f"{d['source_power_w']:.2f} W, sampled peak pressure {d['sampled_pressure_peak_pa'] / 1e6:.2f} MPa. "
             "No apparatus has been built.", fontsize=9, color=MUTED)
    fig.savefig(out / f"{prefix}-setup.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    render.unlink()


def faces_figure(data, out, prefix):
    cfg, mm = data["cfg"], 1e3
    r = np.linspace(0, cfg.clear_radius_m, 801)
    states = {"acoustics-off meniscus": data["start"], "acoustic design": data["q"]}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for j, name in enumerate(("front", "back")):
        target = data["targets"][j].evaluate(r)
        a, b = axes[j, 0], axes[j, 1]
        a.plot(r * mm, target * 1e6, color=INK, ls="--", lw=1.4, label="Cartesian target")
        for (label, q), colour in zip(states.items(), (MUTED, FACE)):
            h = data["space"].height(j, q[j], r)
            a.plot(r * mm, h * 1e6, color=colour, lw=1.2, label=label)
            b.plot(r * mm, (h - target) * 1e9, color=colour, lw=1.4, label=label)
        b.axhspan(-10, 10, color="#2e8b57", alpha=0.12, label="±10 nm goal")
        a.set(xlabel="radius (mm)", ylabel="sag from pinned plane (µm)")
        b.set(xlabel="radius (mm)", ylabel="height error (nm)")
        a.set_title(f"{name} face: sag over the clear aperture", loc="left")
        b.set_title(f"{name} face: error to target", loc="left")
        a.legend(fontsize=8, frameon=False)
        b.legend(fontsize=8, frameon=False)
    for k, (label, q) in enumerate(states.items()):
        tr = trace(data, q)
        s = tr["spots_m"] * 1e6
        a = axes[k, 2]
        a.scatter(s[:, 0], s[:, 1], s=4, color=RAY, alpha=0.5, lw=0)
        a.add_patch(plt.Circle((0, 0), 1, fill=False, color="#2e8b57", ls="--"))
        lim = max(1.3, 1.1 * tr["max_radius_m"] * 1e6)
        a.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="detector x (µm)", ylabel="detector y (µm)")
        a.set_title(f"spot, {label}\nmax {tr['max_radius_m'] * 1e6:.2f} µm; "
                    f"{int(tr['transmitted'].sum())}/{len(tr['transmitted'])} rays", loc="left", fontsize=10)
    fig.text(0.01, 0.005, "Green circle: 1 µm. Numerical model; errors are cycle-mean surface heights, no piston removed.",
             fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}-faces.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def formation_video(data, out, prefix, label, fps=10):
    from acoustic_freeform.forward.formation import envelope

    cfg, mm = data["cfg"], 1e3
    tr = data["trajectory"]
    times, states = tr["time_s"], tr["coefficients_m"]
    ramp = data["formation"]["ramp_s"]
    r_full = np.linspace(0, cfg.radius_m, 801)
    r = np.linspace(0, cfg.clear_radius_m, 401)
    targets = [t.evaluate(r) for t in data["targets"]]
    history = {"t": [], "front": [], "back": [], "spot": []}
    traces = [trace(data, q) for q in states]
    levels = np.asarray(cfg.levels_m) * 1e3
    R = cfg.radius_m * 1e3
    change = max(np.max(abs(data["space"].height(j, states[-1][j], r_full) - data["space"].height(j, states[0][j], r_full)))
                 for j in range(2))
    for t, q, tt in zip(times, states, traces):
        history["t"].append(t)
        for j, name in enumerate(("front", "back")):
            history[name].append(np.max(abs(data["space"].height(j, q[j], r) - targets[j])) * 1e9)
        history["spot"].append(tt["max_radius_m"] * 1e6)
    error_limit = 1.1 * max(max(history["front"]), max(history["back"]))
    spot_limit = 1.1 * max(1.3, max(history["spot"]))
    frames = []
    for i, (t, q) in enumerate(zip(times, states)):
        fig = plt.figure(figsize=(16, 8.5))
        g = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 0.8], hspace=0.38, wspace=0.3)
        ax = fig.add_subplot(g[:, 0])
        faces = faces_mm(data, q, r_full)
        signed = np.r_[-r_full[:0:-1], r_full] * mm
        mirror = lambda h: np.r_[h[:0:-1], h]
        layers = [np.full_like(signed, levels[0]), mirror(faces[0]), mirror(faces[1]), np.full_like(signed, levels[3])]
        for j, colour in enumerate([WATER, RESIN, WATER]):
            ax.fill_between(signed, layers[j], layers[j + 1], color=colour, alpha=0.25 if j != 1 else 0.4, lw=0)
        for j in range(2):
            ax.plot(signed, mirror(faces[j]), color=FACE, lw=2)
        ax.set(xlim=(-1.05 * R, 1.05 * R), ylim=(levels[0] - 0.05 * R, levels[3] + 0.05 * R), aspect="equal",
               xlabel="x (mm)", ylabel="z (mm)")
        ax.set_title(f"section, equal scale\n(changes during formation are ≤ {change * 1e6:.2f} µm: see right)",
                     loc="left", fontsize=10)
        a = fig.add_subplot(g[0, 1])
        for j, (name, colour) in enumerate((("front", FACE), ("back", WATER))):
            a.plot(r * mm, (data["space"].height(j, q[j], r) - targets[j]) * 1e9, color=colour, lw=1.5, label=f"{name} face")
        a.axhspan(-10, 10, color="#2e8b57", alpha=0.12)
        a.set(ylim=(-error_limit, error_limit), xlabel="radius (mm)", ylabel="height error to target (nm)")
        a.set_title("height error over the clear aperture (magnified view)", loc="left", fontsize=10)
        a.legend(fontsize=8, frameon=False, loc="upper left")
        a = fig.add_subplot(g[1, 1])
        a.semilogy(history["t"], history["front"], color=FACE, lw=1, alpha=0.4)
        a.semilogy(history["t"], history["back"], color=WATER, lw=1, alpha=0.4)
        a.semilogy(history["t"][: i + 1], history["front"][: i + 1], color=FACE, lw=2, label="front max error")
        a.semilogy(history["t"][: i + 1], history["back"][: i + 1], color=WATER, lw=2, label="back max error")
        a.axhline(10, color="#2e8b57", ls="--", lw=1)
        e = a.twinx()
        e.plot(times, [envelope(x, ramp) for x in times], color=MUTED, lw=1, ls=":")
        e.set_ylabel("command envelope", color=MUTED, fontsize=8)
        a.set(xlabel="physical time (s)", ylabel="max height error (nm)")
        a.legend(fontsize=8, frameon=False, loc="lower left")
        a = fig.add_subplot(g[:, 2])
        s = traces[i]["spots_m"] * 1e6
        a.scatter(s[:, 0], s[:, 1], s=5, color=RAY, alpha=0.5, lw=0)
        a.add_patch(plt.Circle((0, 0), 1, fill=False, color="#2e8b57", ls="--"))
        a.set(xlim=(-spot_limit, spot_limit), ylim=(-spot_limit, spot_limit), aspect="equal",
              xlabel="detector x (µm)", ylabel="detector y (µm)")
        a.set_title(f"spot at the image plane\nmax {history['spot'][i]:.2f} µm "
                    f"({int(traces[i]['transmitted'].sum())}/{len(traces[i]['transmitted'])} rays)", loc="left", fontsize=10)
        fig.suptitle(f"Start-up from the acoustics-off meniscus under the held {label} command    t = {t:.3f} s",
                     x=0.01, ha="left", fontsize=13)
        fig.text(0.01, 0.01, f"Saved creeping-flow trajectory ({len(times)} states, 0–{times[-1]:.1f} s), "
                 f"recomputed rays. Playback {1 / (fps * (times[1] - times[0])):.0f}× slower than real time. "
                 "Numerical model, not an experiment.", fontsize=9, color=MUTED)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out / f"{prefix}-formation.mp4", frames, fps=fps, quality=8, macro_block_size=1)
    imageio.mimsave(out / f"{prefix}-formation.gif", [f[::2, ::2] for f in frames], duration=1 / fps, loop=0)
    return history


def pressure_cycle_video(data, out, prefix, frames_per_cycle=24, fps=12):
    cfg, mm = data["cfg"], 1e3
    field = data["field"]
    xy = field["xy"] * mm
    tri = mtri.Triangulation(xy[0], xy[1], field["triangles"])
    limit = float(np.max(abs(field["pressure"]))) / 1e6
    r = np.linspace(0, cfg.radius_m, 401)
    faces = faces_mm(data, data["q"], r)
    period = 1 / cfg.frequency_hz
    images = []
    for k in range(frames_per_cycle):
        t = k * period / frames_per_cycle
        value = (field["pressure"] * np.exp(-2j * np.pi * cfg.frequency_hz * t)).real / 1e6
        fig, ax = plt.subplots(figsize=(6.5, 5.6))
        ax.tripcolor(tri, value, shading="gouraud", cmap="RdBu_r", vmin=-limit, vmax=limit)
        for j in range(2):
            ax.plot(r * mm, faces[j], color="black", lw=1)
        ax.set(xlim=(0, cfg.radius_m * mm), ylim=(cfg.levels_m[0] * mm, cfg.levels_m[3] * mm), aspect="equal",
               xlabel="radius (mm)", ylabel="z (mm)")
        ax.set_title(f"acoustic pressure, t = {t * 1e6:.3f} µs of {period * 1e6:.3f} µs\n"
                     f"colour range ±{limit:.2f} MPa; playback {1 / (fps * period / frames_per_cycle):.1e}× slower",
                     fontsize=10)
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out / f"{prefix}-pressure-cycle.gif", images * 3, duration=1 / fps, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=BASE / "meniscus-750khz")
    parser.add_argument("--formation", type=Path, default=BASE / "formation-750khz-dt0025")
    parser.add_argument("--out", type=Path, default=BASE / "figures")
    parser.add_argument("--prefix", default="s07", help="file name prefix")
    parser.add_argument("--label", default="S07", help="design name in titles")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load(args.run.resolve(), args.formation.resolve())
    setup_figure(data, args.out, args.prefix)
    faces_figure(data, args.out, args.prefix)
    history = formation_video(data, args.out, args.prefix, args.label)
    pressure_cycle_video(data, args.out, args.prefix)
    (args.out / "provenance.json").write_text(json.dumps({
        "inputs_sha256": inputs_used,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "final_max_error_nm": [history["front"][-1], history["back"][-1]],
        "final_max_spot_um": history["spot"][-1],
        "scope": "Renderings of saved numerical states; not experimental images.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
