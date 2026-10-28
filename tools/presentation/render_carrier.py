"""The faces as they are, not their cycle mean: interface vibration at the drive frequency.

The saved designs give the cycle-mean faces. The harmonic solve also gives the
normal velocity v of each interface; with exp(-i omega t) its height
oscillation is dh = i v / (omega n_z), and the instantaneous face is

    h(r, t) = h_mean(r) + sum over bands Re(dh_b(r) exp(-i omega_b t))

to first order in the field (the second-order mean shift is in h_mean). Rays
are traced through the instantaneous faces at each phase of the slowest band's
cycle. Writes a figure, a slowed animation and a JSON summary.
Usage: render_carrier.py RUN FORMATION OUT_PREFIX TITLE [--lens-only]
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/presentation"))
import render_cell_figures as cell
import render_showcase as show

from acoustic_freeform.acoustics.helmholtz import DualAcoustics
from acoustic_freeform.optics.raytrace import trace_graph_pair

PHASES, CYCLES, FPS = 48, 3, 24


def vibration(data, q):
    """Per band and face: (frequency, sample radii, complex height oscillation)."""
    cfg, space = data["cfg"], data["space"]
    bands = [f for f, _ in data["field"]["bands"]]
    drives = np.atleast_2d(data["drive"])
    out = []
    for f, d in zip(bands, drives):
        response = DualAcoustics(replace(cfg, frequency_hz=f), space, linear_solver="static_condensed").solve(q, d)
        omega = 2 * np.pi * f
        faces = []
        for j in range(2):
            r = response.radial_m[j]
            dh = 1j * (response.velocity[j] @ np.ones(1)) / (omega * response.normals_z[j])
            order = np.argsort(r)
            faces.append((r[order], dh[order]))
        out.append((f, faces))
    return out


def splines(r, dh):
    # Even in r (zero slope on the axis): fit the mirrored samples.
    rr = np.r_[-r[::-1], r]
    return [CubicSpline(rr, np.r_[part[::-1], part]) for part in (dh.real, dh.imag)]


def main():
    run, formation, prefix, title = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    global PHASES, CYCLES
    if "--preview" in sys.argv:
        PHASES, CYCLES = 4, 1
    prefix.parent.mkdir(parents=True, exist_ok=True)
    data = cell.load(run.resolve(), formation.resolve())
    cfg, space, targets = data["cfg"], data["space"], data["targets"]
    q = data["trajectory"]["coefficients_m"][-1]
    n = data["optics"]["indices"]
    z0, _, z2 = data["conjugates"]
    bands = vibration(data, q)
    fits = [[splines(r, dh) for r, dh in faces] for _, faces in bands]
    slowest = min(f for f, _ in bands)
    period = 1 / slowest

    def oscillation(face, r, t, derivative=0):
        total = 0.0
        for (f, _), fit in zip(bands, fits):
            re, im = fit[face]
            phasor = re(r, derivative) + 1j * im(r, derivative)
            total = total + np.real(phasor * np.exp(-2j * np.pi * f * t))
        return total

    pupil = np.linspace(0, cfg.clear_radius_m, 201)
    mean_error = [space.height(j, q[j], pupil) - targets[j].evaluate(pupil) for j in range(2)]
    envelope = [sum(np.abs(fit[j][0](pupil) + 1j * fit[j][1](pupil)) for fit in fits) for j in range(2)]
    launch = np.linspace(0, cfg.clear_radius_m * 0.999, 401)
    f_number = (z2 - cfg.levels_m[2]) / (2 * cfg.clear_radius_m)
    airy = 1.22 * 587.6e-9 * f_number
    times = np.arange(PHASES) / PHASES * period
    records = []
    for t in list(times) + [None]:
        def evaluate(face, r, derivative=0, t=t):
            base = space.height(face, q[face], r, derivative)
            return base if t is None else base + oscillation(face, np.asarray(r, float), t, derivative)
        tr = trace_graph_pair(cfg, evaluate, n, z0, z2, launch_radius_m=launch,
                              launch_height_m=cfg.levels_m[1] + targets[0].evaluate(launch))
        errors = [None if t is None else mean_error[j] + oscillation(j, pupil, t) for j in range(2)]
        records.append({"t": t, "spots": tr["spots_m"], "max": tr["max_radius_m"], "rms": tr["rms_radius_m"],
                        "errors": errors})
    mean_record, cycle = records[-1], records[:-1]
    inst_max = np.array([[np.max(abs(rec["errors"][j])) for j in range(2)] for rec in cycle])
    spot_max = np.array([rec["max"] for rec in cycle])
    spot_rms = np.array([rec["rms"] for rec in cycle])
    all_spots = np.vstack([rec["spots"] for rec in cycle])
    summary = {
        "bands_hz": [f for f, _ in bands],
        "vibration_peak_on_pupil_nm": [float(np.max(e) * 1e9) for e in envelope],
        "mean_face_error_nm": [float(np.max(abs(e)) * 1e9) for e in mean_error],
        "instantaneous_face_error_max_over_cycle_nm": [float(x) for x in inst_max.max(axis=0) * 1e9],
        "spot_max_um": {"mean_faces": float(mean_record["max"] * 1e6), "worst_instant": float(spot_max.max() * 1e6)},
        "spot_rms_um": {"mean_faces": float(mean_record["rms"] * 1e6), "worst_instant": float(spot_rms.max() * 1e6),
                        "all_phases_together": float(np.sqrt(np.nanmean(np.sum(all_spots**2, axis=1))) * 1e6)},
        "airy_radius_um": float(airy * 1e6),
        "period_ns": float(period * 1e9),
        "scope": "First-order interface vibration from the harmonic solve of the saved design; numerical model, "
                 "not an experiment. Rays use the published indices.",
    }
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    lim = max(3 * airy, 1.2 * spot_max.max()) * 1e6
    ymax = 1.15 * max(max(np.max(e) for e in envelope) + max(np.max(abs(m)) for m in mean_error), 10e-9) * 1e9
    colours = (show.FACE, show.BACK)

    def frame(k):
        rec = cycle[k % PHASES]
        fig = plt.figure(figsize=(16, 9), dpi=120)
        g = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1.35, 1], left=0.06, right=0.93,
                             top=0.86, bottom=0.1, wspace=0.2, hspace=0.5)
        fig.text(0.06, 0.95, title, fontsize=20, weight="bold", color=show.INK)
        bands_text = " + ".join(f"{f / 1e6:g} MHz" for f, _ in bands)
        fig.text(0.06, 0.912, f"The faces as they are during one acoustic cycle ({bands_text}), not their average",
                 fontsize=12.5, color=show.MUTED)
        fig.text(0.97, 0.95, f"t = {rec['t'] * 1e9:6.1f} ns", fontsize=20, ha="right", family="monospace",
                 color=show.INK)
        fig.text(0.97, 0.912, f"of a {period * 1e9:.0f} ns cycle; shown about {1 / (FPS * period / PHASES):.0e}× slower",
                 fontsize=11, ha="right", color=show.MUTED)
        a = fig.add_subplot(g[0, 0])
        for j, (name, colour) in enumerate((("front face", colours[0]), ("back face", colours[1]))):
            a.fill_between(pupil * 1e3, (mean_error[j] - envelope[j]) * 1e9, (mean_error[j] + envelope[j]) * 1e9,
                           color=colour, alpha=0.12, lw=0)
            a.plot(pupil * 1e3, mean_error[j] * 1e9, color=colour, lw=1, ls=(0, (4, 3)))
            a.plot(pupil * 1e3, rec["errors"][j] * 1e9, color=colour, lw=2.4, label=f"{name}, now")
        for s in (-10, 10):
            a.axhline(s, color=show.GOOD, lw=1, ls="--")
        a.text(cfg.clear_radius_m * 1e3, 10.5, "±10 nm goal", color=show.GOOD, ha="right", va="bottom", fontsize=9)
        a.set_ylim(-ymax, ymax)
        a.set_xlim(0, cfg.clear_radius_m * 1e3)
        a.set_xlabel("radius on the pupil (mm)")
        a.set_ylabel("height minus target (nm)")
        a.set_title("Face height error now (solid), its cycle mean (dashed) and the range over the cycle (shaded)",
                    loc="left", fontsize=11)
        a.legend(loc="lower left", fontsize=9, frameon=False)
        b = fig.add_subplot(g[0, 1])
        b.add_patch(plt.Circle((0, 0), airy * 1e6, color=show.GOOD, alpha=0.28, lw=0))
        b.scatter(all_spots[:, 0] * 1e6, all_spots[:, 1] * 1e6, s=2, color="#c9ced6", lw=0)
        b.scatter(rec["spots"][:, 0] * 1e6, rec["spots"][:, 1] * 1e6, s=6, color=show.RAY, lw=0)
        b.set_xlim(-lim, lim)
        b.set_ylim(-lim, lim)
        b.set_aspect("equal")
        b.set_xlabel("screen x (µm)")
        b.set_ylabel("y (µm)")
        b.set_title(f"Spot now (red), all phases (grey), Airy disc (green, {airy * 1e6:.2f} µm)", loc="left",
                    fontsize=11)
        c = fig.add_subplot(g[1, :])
        tt = times * 1e9
        c.plot(tt, inst_max[:, 0] * 1e9, color=colours[0], lw=2, label="front face max error (nm)")
        c.plot(tt, inst_max[:, 1] * 1e9, color=colours[1], lw=2, label="back face max error (nm)")
        c.axhline(10, color=show.GOOD, lw=1, ls="--")
        c.axvline(rec["t"] * 1e9, color=show.MUTED, lw=0.8)
        c.set_xlabel("time within the cycle (ns)")
        c.set_ylabel("max error (nm)")
        c2 = c.twinx()
        c2.plot(tt, spot_max * 1e6, color=show.RAY, lw=1.5, ls=":", label="spot max radius (µm)")
        c2.axhline(airy * 1e6, color=show.GOOD, lw=0.8, ls=":")
        c2.set_ylabel("spot max radius (µm)", color=show.RAY)
        c2.spines["right"].set_visible(True)
        lines = c.get_legend_handles_labels()
        lines2 = c2.get_legend_handles_labels()
        c.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="lower left", bbox_to_anchor=(0.0, 1.02),
                 fontsize=9, frameon=False, ncol=3)
        c.set_title("Over the cycle", loc="right", fontsize=11)
        fig.text(0.06, 0.02, "Numerical model (axisymmetric, lossless acoustics, ideal sources); first-order interface "
                 "vibration from the harmonic solve of the saved design. Not an experiment. Rays use the published "
                 f"indices {n[0]:g}/{n[1]:g}/{n[2]:g}.", fontsize=9.5, color=show.MUTED)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return image

    # The lens itself: true-scale section, the same section with only the vibration
    # magnified, both faces in 3D coloured by their displacement, and the spot.
    mm, R = 1e3, cfg.radius_m
    r_full = np.linspace(0, R, 300)
    mean_faces = [cfg.levels_m[j + 1] + space.height(j, q[j], r_full) for j in range(2)]
    peak = max(float(np.max(e)) for e in envelope)
    magnify = show.nice_factor(0.25e-3 / peak)
    launch_rays = np.linspace(-0.92, 0.92, 17) * cfg.clear_radius_m
    paths_mean = cell.ray_paths(data, q, launch_rays)
    theta = np.linspace(0, 1.5 * np.pi, 60)  # three quarters of the revolution, cut open
    rr, th = np.meshgrid(r_full[::6], theta)
    colour_limit = peak * 1e9
    levels_mm = np.asarray(cfg.levels_m) * mm
    signed = np.r_[-r_full[::-1], r_full] * mm

    def mirror(h):
        return np.r_[h[::-1], h] * mm

    def section(ax, faces, title_text, dashed=None):
        ax.fill_between(signed, levels_mm[0], mirror(faces[0]), color=show.WATER, lw=0)
        ax.fill_between(signed, mirror(faces[0]), mirror(faces[1]), color=show.RESIN, lw=0)
        ax.fill_between(signed, mirror(faces[1]), levels_mm[3], color=show.WATER, lw=0)
        if dashed is not None:
            for h in dashed:
                ax.plot(signed, mirror(h), color=show.INK, lw=0.9, ls=(0, (4, 3)))
        for h, colour in zip(faces, colours):
            ax.plot(signed, mirror(h), color=colour, lw=2.4)
        ax.plot([-R * mm, -R * mm, R * mm, R * mm, -R * mm],
                [levels_mm[0], levels_mm[3], levels_mm[3], levels_mm[0], levels_mm[0]], color=show.INK, lw=2.5)
        ax.set_aspect("equal")
        ax.set_xlim(-R * mm - 0.2, R * mm + 0.2)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title(title_text, loc="left", fontsize=11)

    def frame_lens(k):
        rec = cycle[k % PHASES]
        t = rec["t"]
        moving = [oscillation(j, r_full, t) for j in range(2)]
        fig = plt.figure(figsize=(16, 9), dpi=120)
        g = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.3, 0.75], left=0.05, right=0.97, top=0.86, bottom=0.1,
                             wspace=0.2, hspace=0.42)
        fig.text(0.05, 0.95, title, fontsize=20, weight="bold", color=show.INK)
        bands_text = " + ".join(f"{f / 1e6:g} MHz" for f, _ in bands)
        fig.text(0.05, 0.912, f"The liquid lens during one acoustic cycle ({bands_text}); peak vibration of the faces "
                 f"{peak * 1e9:.0f} nm", fontsize=12.5, color=show.MUTED)
        fig.text(0.97, 0.95, f"t = {t * 1e9:6.1f} ns", fontsize=20, ha="right", family="monospace", color=show.INK)
        fig.text(0.97, 0.912, f"of a {period * 1e9:.0f} ns cycle, slowed about {PHASES * period ** -1 / FPS:.0e}×",
                 fontsize=11, ha="right", color=show.MUTED)
        a = fig.add_subplot(g[:, 0])
        section(a, mean_faces, "The lens at true scale, with the light it focuses\n(the vibration is invisible here)")
        for path in paths_mean:
            a.plot(path[:, 0] * mm, path[:, 1] * mm, color=show.RAY, lw=0.8, alpha=0.85)
        a.set_ylim(levels_mm[0] - 0.3, max(levels_mm[3], z2 * mm) + 0.3)
        b = fig.add_subplot(g[0, 1])
        shown = [mean_faces[j] + magnify * moving[j] for j in range(2)]
        section(b, shown, f"Same section, vibration magnified ×{magnify:,}\n(dashed: the mean faces, which hold the lens)",
                dashed=mean_faces)
        b.set_ylim(levels_mm[0] - 0.3, levels_mm[3] + 0.3)
        c = fig.add_subplot(g[1, 1], projection="3d")
        cmap = plt.get_cmap("RdBu_r")
        for j in range(2):
            z_mean = np.interp(rr, r_full, mean_faces[j])
            dz = np.interp(rr, r_full, moving[j])
            c.plot_surface(rr * np.cos(th) * mm, rr * np.sin(th) * mm, (z_mean + magnify * dz) * mm,
                           facecolors=cmap(0.5 + 0.5 * dz * 1e9 / colour_limit), linewidth=0, antialiased=False,
                           shade=False, rstride=1, cstride=1, alpha=0.95 if j == 0 else 0.8)
            ring = np.linspace(0, 1.5 * np.pi, 120)
            c.plot(R * np.cos(ring) * mm, R * np.sin(ring) * mm, np.full_like(ring, mean_faces[j][-1] * mm),
                   color=colours[j], lw=1.5)
        lens_low = float(np.min(mean_faces[0]) * mm) - 0.15
        lens_high = float(np.max(mean_faces[1]) * mm) + 0.15
        c.set_box_aspect((1, 1, 0.55), zoom=1.3)
        c.view_init(elev=14, azim=-58)
        c.set_xlim(-R * mm, R * mm)
        c.set_ylim(-R * mm, R * mm)
        c.set_zlim(lens_low, lens_high)
        c.set_axis_off()
        c.set_title(f"Both faces in 3D, vibration ×{magnify:,}\ncolour: displacement now, ±{colour_limit:.0f} nm",
                    fontsize=11, loc="left")
        d = fig.add_subplot(g[:, 2])
        d.add_patch(plt.Circle((0, 0), airy * 1e6, color=show.GOOD, alpha=0.28, lw=0))
        d.scatter(all_spots[:, 0] * 1e6, all_spots[:, 1] * 1e6, s=2, color="#c9ced6", lw=0)
        d.scatter(rec["spots"][:, 0] * 1e6, rec["spots"][:, 1] * 1e6, s=6, color=show.RAY, lw=0)
        d.set_xlim(-lim, lim)
        d.set_ylim(-lim, lim)
        d.set_aspect("equal")
        d.set_xlabel("screen x (µm)")
        d.set_title(f"Focus now: spot (red), all instants (grey),\nAiry disc (green, {airy * 1e6:.2f} µm)",
                    loc="left", fontsize=11)
        fig.text(0.05, 0.02, "Numerical model (axisymmetric, lossless acoustics, ideal sources); first-order "
                 "vibration from the harmonic solve of the saved design. Not an experiment. Rays traced through the "
                 f"actual faces with the published indices {n[0]:g}/{n[1]:g}/{n[2]:g}.", fontsize=9.5, color=show.MUTED)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return image

    if "--lens-only" not in sys.argv:
        frames = [frame(k) for k in range(PHASES * CYCLES)]
        imageio.mimsave(prefix.with_suffix(".mp4"), frames, fps=FPS, quality=9, macro_block_size=1)
        imageio.imwrite(prefix.with_suffix(".png"), frames[PHASES // 4])
    lens_frames = [frame_lens(k) for k in range(PHASES * CYCLES)]
    lens_path = prefix.with_name(prefix.name + "-lens")
    imageio.mimsave(lens_path.with_suffix(".mp4"), lens_frames, fps=FPS, quality=9, macro_block_size=1)
    imageio.imwrite(lens_path.with_suffix(".png"), lens_frames[PHASES // 4])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
