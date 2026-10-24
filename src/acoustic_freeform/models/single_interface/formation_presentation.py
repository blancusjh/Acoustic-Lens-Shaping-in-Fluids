"""Read-only formation analysis and visualization; never synthesizes a trajectory."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from .config import LensConfig
from .optics import trace_surface
from .surface import SurfaceSpace


def load_formation(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    with np.load(directory / "trajectory.npz", allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in saved.files}
    cfg = LensConfig(**report["configuration"])
    space = SurfaceSpace(cfg)
    t = data["times_s"]
    if len(t) != len(data["coefficients"]) or np.any(np.diff(t) <= 0):
        raise ValueError("A strictly increasing saved physical timeline is required.")
    if report["initial_condition"] != "rest_fixed":
        raise ValueError("This presentation expects fixed-drive formation from rest.")
    np.testing.assert_array_equal(data["drive_m_s"], np.broadcast_to(
        data["drive_m_s"][0], data["drive_m_s"].shape))
    return {"directory": directory, "report": report, "data": data, "cfg": cfg, "space": space}


def exact_height_error(space, coefficients, radial_m):
    """Absolute lab-height error versus exact Cartesian geometry, without refitting."""
    return (space.config.radius_m * space.evaluate(coefficients, radial_m / space.config.radius_m)
            - space.optical_vertex_m - space.config.diopter.sag(radial_m))


def height_maximum(space, coefficients, count=1601):
    """Derivative bracketing and endpoints; floating-point search, not interval proof."""
    cfg = space.config

    def derivative(r):
        z = cfg.diopter.sag(np.asarray(r))
        fr, fz = cfg.diopter.fermat_gradient(np.asarray(r), z)
        return space.evaluate(coefficients, np.asarray(r) / cfg.radius_m, 1) + fr / fz

    grid = np.linspace(0, cfg.clear_radius_m, count)
    d = derivative(grid)
    brackets = np.flatnonzero(d[:-1] * d[1:] < 0)
    # Refine all sign-changing brackets together. The optical sag routine is
    # vectorized; this avoids hundreds of redundant scalar optical continuations.
    lower, upper = grid[brackets].copy(), grid[brackets+1].copy()
    sign_lower = d[brackets].copy()
    for _ in range(32):
        if not len(lower):
            break
        mid = (lower+upper)/2
        sign_mid = derivative(mid)
        left = sign_lower * sign_mid <= 0
        upper = np.where(left, mid, upper)
        lower = np.where(left, lower, mid)
        sign_lower = np.where(left, sign_lower, sign_mid)
    points = np.r_[grid, (lower+upper)/2]
    return float(np.max(abs(exact_height_error(space, coefficients, points))))


def analyze_formation(run, ray_count=401):
    cfg, space, data = run["cfg"], run["space"], run["data"]
    r = np.linspace(0, cfg.radius_m, 801)
    pupil = np.linspace(0, cfg.clear_radius_m, 801)
    surface, error, maxima, spots, rms = [], [], [], [], []
    plane = space.optical_vertex_m + cfg.focal_distance_m
    for c in data["coefficients"]:
        surface.append(cfg.radius_m * space.evaluate(c, r / cfg.radius_m))
        error.append(exact_height_error(space, c, pupil))
        maxima.append(height_maximum(space, c))
        ray = trace_surface(space, c, count=ray_count, target_plane_m=plane)
        spots.append(ray["target_spot_r_m"])
        rms.append(ray["rms_spot_at_target_m"])
    theta = np.arange(ray_count) * np.pi * (3 - np.sqrt(5))
    spots = np.asarray(spots)
    # Axisymmetry gives the same meridional intercept at every azimuth.
    xy = spots[:, :, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1)[None, :, :]
    np.testing.assert_allclose(np.sqrt(np.mean(np.sum(xy**2, axis=2), axis=1)), rms,
                               rtol=1e-12, atol=1e-15)
    result = {"radius_m": r, "pupil_radius_m": pupil, "time_s": data["times_s"],
              "surface_height_m": np.asarray(surface), "height_error_m": np.asarray(error),
              "maximum_height_error_m": np.asarray(maxima), "spot_xy_m": xy,
              "spot_rms_m": np.asarray(rms), "detector_z_m": plane,
              "target_height_m": space.optical_vertex_m + cfg.diopter.sag(r)}
    if not all(np.all(np.isfinite(v)) for v in result.values()):
        raise ValueError("Nonfinite reconstructed geometry or optics.")
    return result


def apparatus_figure(run):
    """Declared wall/base and actual liquid; source regions are boundary markings."""
    cfg, space = run["cfg"], run["space"]
    mm = 1e3
    radius, outer = cfg.radius_m * mm, (cfg.radius_m + cfg.wall_thickness_m) * mm
    depth, base = cfg.depth_m * mm, cfg.base_thickness_m * mm
    fig = plt.figure(figsize=(10, 8), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    theta = np.linspace(0, 2*np.pi, 97)
    rad = np.linspace(0, radius, 45)
    rr, tt = np.meshgrid(rad, theta)
    hh = cfg.radius_m * space.evaluate(run["data"]["coefficients"][-1], rr / radius) * mm
    ax.plot_surface(rr*np.cos(tt), rr*np.sin(tt), hh, color="#26a69a", alpha=.85,
                    linewidth=0, rcount=70, ccount=45)
    # Rear half of the container is rendered; the cutaway is a visibility choice.
    rear = np.linspace(0, np.pi, 65)
    tt, zz = np.meshgrid(rear, [-depth, 0])
    for rad_wall in (radius, outer):
        ax.plot_surface(rad_wall*np.cos(tt), rad_wall*np.sin(tt), zz,
                        color="#8da0b2", alpha=.17, shade=False)
    tt, rr = np.meshgrid(rear, [radius, outer])
    ax.plot_surface(rr*np.cos(tt), rr*np.sin(tt), np.zeros_like(rr),
                    color="#8da0b2", alpha=.4)
    tt, rr = np.meshgrid(theta, np.linspace(0, outer, 20))
    ax.plot_surface(rr*np.cos(tt), rr*np.sin(tt), np.full_like(rr, -depth),
                    color="#7b9fba", alpha=.5)
    tt, zz = np.meshgrid(theta, [-depth-base, -depth])
    ax.plot_surface(outer*np.cos(tt), outer*np.sin(tt), zz, color="#7b9fba", alpha=.5)
    # Colored inner-wall bands have zero thickness: exactly the imposed source supports.
    # No unmodeled piezo backing, electrode thickness or extra retaining ring is invented.
    for row in range(cfg.array_rows):
        pitch = depth / cfg.array_rows
        center = -depth + (row + .5) * pitch
        half = pitch * cfg.element_fill / 2
        tt, zz = np.meshgrid(rear, [center-half, center+half])
        ax.plot_surface(radius*np.cos(tt), radius*np.sin(tt), zz,
                        color="#e69f00", alpha=.65, shade=False)
    ax.plot(radius*np.cos(theta), radius*np.sin(theta), np.zeros_like(theta), color=".3", lw=1)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z above rim (mm)",
           xlim=(-outer, outer), ylim=(-outer, outer), zlim=(-depth-base, 2),
           title="Single-face apparatus · rear-half wall cutaway\nEqual geometric scale; final computed liquid surface")
    ax.set_box_aspect((2*outer, 2*outer, depth+base+2))
    ax.view_init(elev=24, azim=-65)
    ax.legend(handles=[Line2D([], [], color=c, lw=5, label=l) for c, l in (
        ("#26a69a", "Computed liquid–air surface"), ("#8da0b2", "Container wall"),
        ("#7b9fba", "Base"), ("#e69f00", "Ideal acoustic source bands (not hardware)"))],
        loc="upper left", fontsize=8)
    return fig


def formation_animation(run, analysis, frame_stride=2):
    """Animate only saved states with synchronized surface, errors and ray spots."""
    a = analysis
    t, radius, pupil = a["time_s"], a["radius_m"], a["pupil_radius_m"]
    indices = np.unique(np.r_[np.arange(0, len(t), frame_stride), len(t)-1])
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    surface_ax, error_ax, full_ax, height_ax, rms_ax, zoom_ax = axs.ravel()
    sr = np.r_[-radius[:0:-1], radius] * 1e3
    sym = lambda v: np.r_[v[:0:-1], v]
    surface_ax.plot(sr, sym(a["target_height_m"]) * 1e3, "--", color=".35", label="Cartesian target")
    surface_line, = surface_ax.plot(sr, sym(a["surface_height_m"][0]) * 1e3,
                                    color="#009688", lw=2, label="Forward surface")
    surface_ax.set(xlabel="Signed radius (mm)", ylabel="Height above rim (mm)",
                   title="Forming surface · equal scale", aspect="equal", adjustable="box")
    all_h = a["surface_height_m"] * 1e3
    surface_ax.set_ylim(min(0, all_h.min())-.08, max(all_h.max(), a["target_height_m"].max()*1e3)+.08)
    surface_ax.legend(fontsize=8)
    error_ax.axhspan(-10, 10, color="#b8e0d2", alpha=.5)
    error_line, = error_ax.plot(pupil*1e3, a["height_error_m"][0]*1e9, color="#d6604d")
    error_ax.set_yscale("symlog", linthresh=10)
    error_ax.set(xlabel="Pupil radius (mm)", ylabel="Signed height error (nm)",
                 title="Exact target · shaded ±10 nm")
    ee = a["height_error_m"]*1e9
    error_ax.set_ylim(min(-20, ee.min()*1.15), max(20, ee.max()*1.15))
    xy = a["spot_xy_m"]*1e6
    extent = max(1, np.max(np.linalg.norm(xy, axis=2))*1.1)
    zoom = max(.1, np.max(np.linalg.norm(xy[-1], axis=1))*1.3)
    dots = []
    for ax, bound, title in ((full_ax, extent, "Spot diagram · fixed full-run scale"),
                             (zoom_ax, zoom, "Spot diagram · fixed final-state zoom")):
        dots.append(ax.scatter(xy[0, :, 0], xy[0, :, 1], s=4, color="#2166ac", alpha=.55))
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound), aspect="equal",
               xlabel="Detector x (µm)", ylabel="Detector y (µm)", title=title)
    zoom_note = zoom_ax.text(.02, .97, "", transform=zoom_ax.transAxes, va="top", fontsize=8)
    cursors = []
    for ax, values, label in ((height_ax, a["maximum_height_error_m"]*1e9, "Max height error (nm)"),
                              (rms_ax, a["spot_rms_m"]*1e6, "Geometric spot RMS (µm)")):
        ax.semilogy(t*1e3, values, color=".65", lw=1)
        cursors.append(ax.axvline(0, color="#009688", lw=1.5))
        ax.set(xlabel="Physical time (ms)", ylabel=label, xlim=(0, t[-1]*1e3))
    height_ax.axhline(10, color="k", ls="--", lw=1)
    title = fig.suptitle("")
    for ax in axs.ravel():
        ax.grid(alpha=.2)

    def update(frame):
        k = int(indices[frame])
        surface_line.set_ydata(sym(a["surface_height_m"][k])*1e3)
        error_line.set_ydata(a["height_error_m"][k]*1e9)
        for scatter in dots:
            scatter.set_offsets(xy[k])
        for cursor in cursors:
            cursor.set_xdata([t[k]*1e3]*2)
        outside = int(np.count_nonzero(np.any(abs(xy[k]) > zoom, axis=1)))
        zoom_note.set_text(f"Outside zoom: {outside}/{xy.shape[1]} rays")
        title.set_text(f"Single-face forward formation · physical t = {t[k]*1e3:.2f} ms\n"
                       f"Max height error = {a['maximum_height_error_m'][k]*1e9:.1f} nm · "
                       f"spot RMS = {a['spot_rms_m'][k]*1e6:.3f} µm · fixed detector")
        return surface_line, error_line, *dots, *cursors, zoom_note, title

    update(0)
    fig.canvas.draw()
    fig.set_layout_engine(None)  # Fixed panel geometry throughout playback.
    animation = FuncAnimation(fig, update, frames=len(indices), interval=110, blit=False)
    with plt.rc_context({"animation.embed_limit": 100}):
        html = animation.to_jshtml(default_mode="once")
    html = re.sub(r"<link[^>]*>", "", html)
    html = re.sub(r'(<button[^>]*title="([^"]+)"[^>]*>)\s*<i[^>]*></i>', r"\1\2", html)
    html += "<style>.anim-buttons button {width:auto;padding:5px;margin:2px}</style>"
    # Static snapshots are exactly selected animation frames.
    snapshots = []
    for frame in (0, len(indices)//2, len(indices)-1):
        update(frame)
        snapshots.append((int(indices[frame]), fig_to_png(fig)))
    plt.close(fig)
    return html, indices, snapshots


def fig_to_png(fig):
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    return buffer.getvalue()
