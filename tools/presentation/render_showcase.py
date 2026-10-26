"""Showcase animations of lens formation, made so the surface change is visible.

The interfaces move by micrometres on a millimetre cell and finish within a
nanometre of the target, so an equal-scale drawing shows nothing. Here each
face is drawn as the target plus its deviation magnified by a stated factor;
the dashed line is the target. Rays are traced through the actual (unmagnified)
state at every frame; inside the cell they are drawn through the magnified
faces (schematic), and the focus zoom, spot and error curves are exact.
Only saved states of the forward trajectory are shown, with their physical
time; nothing is interpolated.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/presentation"))
import render_cell_figures as cell

WATER, RESIN, FACE, TARGET, RAY, INK, MUTED, GOOD, BACK = (
    "#d6e9f5", "#f3c77a", "#8a4b08", "#1f2933", "#c0392b", "#1f2933", "#6b7785", "#2e8b57", "#2c7fb8")
for name in ("Inter", "Helvetica", "Arial", "DejaVu Sans"):
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "xtick.direction": "out", "ytick.direction": "out", "font.size": 11})


def nice_factor(value):
    for step in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000):
        if step >= value:
            return step
    return 10000


def write_gif(frames, path, width, fps, max_frames=110):
    """Compact README copy: every n-th frame, Lanczos-resampled, one shared palette."""
    from PIL import Image

    step = max(1, -(-len(frames) // max_frames))
    resized = []
    for f in frames[::step]:
        image = Image.fromarray(f)
        resized.append(image.resize((width, round(image.height * width / image.width)), Image.LANCZOS))
    # One palette for all frames, from the first, middle and last frames side by side.
    sample = Image.new("RGB", (3 * width, resized[0].height + 60))
    for i, image in enumerate((resized[0], resized[len(resized) // 2], resized[-1])):
        sample.paste(image, (i * width, 0))
    # Large swatches of the line colours keep them exact in the palette.
    for i, colour in enumerate((RAY, FACE, BACK, GOOD, MUTED, INK, WATER, RESIN, "#9aa5b1")):
        sample.paste(colour, (i * 3 * width // 9, resized[0].height, (i + 1) * 3 * width // 9, resized[0].height + 60))
    palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    images = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in resized]
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 * step / fps), loop=0,
                   optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--formation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--hold-s", type=float, default=2.5)
    parser.add_argument("--gif-width", type=int, default=1280)
    parser.add_argument("--gif-from-mp4", action="store_true", help="only rebuild the GIF from the saved MP4")
    parser.add_argument("--preview", type=int, nargs="*", help="only write PNGs of these saved states")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.gif_from_mp4:
        write_gif(list(imageio.mimread(args.out.with_suffix(".mp4"), memtest=False)), args.out.with_suffix(".gif"),
                  args.gif_width, args.fps)
        return
    data = cell.load(args.run.resolve(), args.formation.resolve())
    cfg, space, targets = data["cfg"], data["space"], data["targets"]
    traj = data["trajectory"]
    times, states = traj["time_s"], traj["coefficients_m"]
    mm, R = 1e3, cfg.radius_m * 1e3
    L = np.asarray(cfg.levels_m) * mm
    r = np.linspace(0, cfg.radius_m, 400)
    pupil = np.linspace(0, cfg.clear_radius_m, 401)
    target_h = [cfg.levels_m[j + 1] + targets[j].evaluate(r) for j in range(2)]
    target_p = [targets[j].evaluate(pupil) for j in range(2)]
    z0, _, z2 = data["conjugates"]
    n = data["optics"]["indices"]
    # Magnification: the initial (acoustics-off) deviation is drawn as about 0.3 mm.
    dev0 = max(np.max(abs(cfg.levels_m[j + 1] + space.height(j, states[0][j], r) - target_h[j])) for j in range(2))
    factor = nice_factor(0.3e-3 / dev0)
    errors = np.array([[np.max(abs(space.height(j, q[j], pupil) - target_p[j])) for j in range(2)] for q in states])
    traces = [cell.trace(data, q) for q in states]
    spots = np.array([t["max_radius_m"] for t in traces])
    f_number = (z2 - cfg.levels_m[2]) / (2 * cfg.clear_radius_m)
    airy = 1.22 * 587.6e-9 * f_number
    launch = np.linspace(-0.92 * cfg.clear_radius_m, 0.92 * cfg.clear_radius_m, 15)
    envelope_ramp = data["formation"]["ramp_s"]
    from acoustic_freeform.forward.formation import envelope

    rms = np.array([t["rms_radius_m"] for t in traces])
    full = 1.15 * spots[0]  # spot panel at the starting scale
    zoom = 3 * airy  # focus inset and spot panel at the diffraction scale
    frames = []
    # Each saved state is shown for 1-5 frames, more where the error changes fastest.
    change = np.r_[0, np.abs(np.diff(np.log10(errors.max(axis=1))))]
    repeats = np.clip(np.round(change / 0.03), 1, 5).astype(int)
    order = [k for k in range(len(states)) for _ in range(repeats[k])]
    order += [len(states) - 1] * int(args.hold_s * args.fps)
    dt = float(np.median(np.diff(times)))
    slow = (1 / (args.fps * dt), repeats.max() / (args.fps * dt))
    if args.preview:
        order = args.preview
    for frame, k in enumerate(order):
        q, t = states[k], times[k]
        shown = [target_h[j] + factor * (cfg.levels_m[j + 1] + space.height(j, q[j], r) - target_h[j]) for j in range(2)]
        fig = plt.figure(figsize=(16, 9), dpi=120)
        g = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], height_ratios=[1.15, 1, 0.55],
                             left=0.05, right=0.97, top=0.83, bottom=0.11, wspace=0.18, hspace=0.45)
        fig.text(0.05, 0.955, args.title, fontsize=20, weight="bold", color=INK)
        fig.text(0.05, 0.915, args.subtitle, fontsize=12, color=MUTED)
        fig.text(0.97, 0.955, f"t = {t:5.3f} s", fontsize=20, ha="right", color=INK, family="monospace")
        # --- Section with magnified deviation and rays.
        ax = fig.add_subplot(g[:, 0])
        signed = np.r_[-r[::-1], r] * mm
        mirror = lambda h: np.r_[h[::-1], h] * mm
        bottom, front, back, top = (np.full_like(signed, L[0]), mirror(shown[0]), mirror(shown[1]),
                                    np.full_like(signed, L[3]))
        ax.fill_between(signed, bottom, front, color=WATER, lw=0)
        ax.fill_between(signed, front, back, color=RESIN, lw=0)
        ax.fill_between(signed, back, top, color=WATER, lw=0)
        for j in range(2):
            ax.plot(signed, mirror(target_h[j]), color=TARGET, lw=1.0, ls=(0, (4, 3)))
            ax.plot(signed, mirror(shown[j]), color=FACE, lw=2.2)
        ax.plot([-R, -R], [L[0], L[3]], color=INK, lw=2.5)
        ax.plot([R, R], [L[0], L[3]], color=INK, lw=2.5)
        ax.plot([-R, R], [L[0], L[0]], color=INK, lw=2.5)
        ax.plot([-R, R], [L[3], L[3]], color=INK, lw=2.5)
        for path in cell.ray_paths(data, q, launch):
            pts = path.copy()
            for j in (1, 2):  # move the refraction points onto the magnified faces (schematic)
                rr = abs(pts[j, 0])
                pts[j, 1] = np.interp(rr, r, shown[j - 1])
            ax.plot(pts[:, 0] * mm, pts[:, 1] * mm, color=RAY, lw=0.9, alpha=0.85)
        ax.set_xlim(-R - 0.3, R + 0.3)
        ax.set_ylim(L[0] - 0.25, max(L[3], z2 * mm) + 0.45)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        source = "collimated light" if np.isinf(z0) else f"point source at z = {z0 * mm:.1f} mm"
        ax.set_title(f"Section: each face drawn as target + deviation × {factor:,} (dashed: target)\n"
                     f"{source}; rays traced through the actual faces, drawn on the magnified ones",
                     loc="left", fontsize=11, color=INK)
        # Focus zoom (exact, in µm).
        inset = ax.inset_axes([0.62, 0.70, 0.36, 0.28])
        exit_rays = cell.ray_paths(data, q, np.linspace(-cfg.clear_radius_m * 0.98, cfg.clear_radius_m * 0.98, 41))
        for path in exit_rays:
            a, b = path[2], path[3]
            zz = np.linspace(z2 - 0.4 * (b[1] - z2) - 2e-4, b[1], 50)
            xx = a[0] + (zz - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            inset.plot(xx * 1e6, (zz - z2) * 1e6, color=RAY, lw=0.5, alpha=0.6)
        inset.add_patch(plt.Circle((0, 0), airy * 1e6, color=GOOD, alpha=0.25, lw=0))
        inset.set_xlim(-zoom * 1e6, zoom * 1e6)
        inset.set_ylim(-15 * zoom * 1e6, 15 * zoom * 1e6)
        inset.set_title("focus (µm; axis stretched 15×)", fontsize=9, color=MUTED)
        inset.tick_params(labelsize=8)
        inset.plot(0, 0, marker="+", color=INK, ms=8)
        # --- Spot.
        s = traces[k]["spots_m"] * 1e6
        pair = g[0, 1].subgridspec(1, 2, wspace=0.35)
        for column, (limit, label) in enumerate(((full, "starting scale"), (zoom, "diffraction scale"))):
            a = fig.add_subplot(pair[0, column])
            a.add_patch(plt.Circle((0, 0), airy * 1e6, color=GOOD, alpha=0.3, lw=0))
            a.scatter(s[:, 0], s[:, 1], s=5, color=RAY, alpha=0.6, lw=0, clip_on=True)
            a.set_xlim(-limit * 1e6, limit * 1e6)
            a.set_ylim(-limit * 1e6, limit * 1e6)
            a.set_aspect("equal")
            a.set_xlabel(f"x (µm), {label}", fontsize=10)
            a.tick_params(labelsize=8)
            if column == 0:
                a.set_ylabel("y (µm)")
        fig.text(0.585, 0.862, f"Spot at the image plane (full clear aperture): max {spots[k] * 1e6:6.2f} µm, "
                 f"rms {rms[k] * 1e6:6.3f} µm\ngreen: Airy disc, radius {airy * 1e6:.2f} µm at 587.6 nm",
                 fontsize=11, va="bottom", color=INK)
        # --- Height error.
        b = fig.add_subplot(g[1, 1])
        for j, (label, colour) in enumerate((("front face", FACE), ("back face", BACK))):
            b.semilogy(times, errors[:, j] * 1e9, color=colour, lw=0.8, alpha=0.25)
            b.semilogy(times[: k + 1], errors[: k + 1, j] * 1e9, color=colour, lw=2.0, label=label)
        b.axhline(10, color=GOOD, ls="--", lw=1)
        b.text(times[-1], 12, "10 nm goal", color=GOOD, ha="right", va="bottom", fontsize=9)
        b.axvline(t, color=MUTED, lw=0.8)
        b.set_xlabel("physical time (s)")
        b.set_ylabel("max height error (nm)")
        b.set_title("Pupil height error to the Cartesian target", loc="left", fontsize=11)
        b.legend(loc="upper right", fontsize=9, frameon=False)
        # --- Drive envelope.
        c = fig.add_subplot(g[2, 1])
        c.plot(times, [envelope(x, envelope_ramp) for x in times], color=MUTED, lw=1)
        c.fill_between(times[: k + 1], [envelope(x, envelope_ramp) for x in times[: k + 1]], color="#9aa5b1", alpha=0.5)
        c.set_ylim(0, 1.1)
        c.set_xlabel("physical time (s)")
        c.set_ylabel("drive")
        c.set_title("Acoustic command, switched on with a 1 s ramp", loc="left", fontsize=11)
        fig.text(0.05, 0.015, "Numerical model (axisymmetric, lossless acoustics, creeping flow), not an experiment. "
                 f"Rays use the published indices {n[0]:g}/{n[1]:g}/{n[2]:g}.\nOnly saved states of the forward "
                 f"trajectory are shown (no interpolation); playback {slow[0]:.0f}-{slow[1]:.0f}× slower than "
                 "real time, slowest where the faces move fastest.", fontsize=9.5, color=MUTED)
        if args.preview:
            fig.savefig(args.out.parent / f"{args.out.stem}-preview-{k}.png")
            plt.close(fig)
            continue
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        if frame % 40 == 0:
            print(f"frame {frame}/{len(order)}", flush=True)
    if args.preview:
        return
    mp4 = args.out.with_suffix(".mp4")
    imageio.mimsave(mp4, frames, fps=args.fps, quality=9, macro_block_size=1)
    write_gif(frames, args.out.with_suffix(".gif"), args.gif_width, args.fps)
    args.out.with_suffix(".json").write_text(json.dumps({
        "inputs_sha256": cell.inputs_used,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "deviation_magnification": factor, "frames": len(frames), "states": len(states),
        "final_max_error_nm": (errors[-1] * 1e9).tolist(), "final_spot_um": float(spots[-1] * 1e6),
        "scope": "Rendering of a saved numerical trajectory; not an experiment.",
    }, indent=2) + "\n")
    print(json.dumps({"mp4": str(mp4), "factor": factor}))


if __name__ == "__main__":
    main()
