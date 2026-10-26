# ruff: noqa: C408, ISC004
"""Story video with sound: meniscus, sound on, lens forms, light, a second lens, moving source.

Scenes, all from saved states of the declared numerical model:
1. Lens A's acoustics-off meniscus, with light traced through it.
2. Lens A's command switched on: its saved forward formation trajectory.
3. The formed lens A focusing light.
4. Sound off; pins and fill set for lens B. This transition is not simulated
   and is shown as a card, not as a motion.
5. Lens B's saved formation from its own meniscus under its command.
6. Lens B formed; the point source is moved along the axis (exact ray traces).

Each face is drawn as target plus deviation magnified by a stated factor. The
pressure map is the harmonic field computed at the formed state, scaled by the
drive envelope and shown at a slowed phase. The audio is the drive itself with
every frequency divided by a stated factor (1.5 MHz -> 750 Hz), amplitude
following the drive envelope.
Usage: render_story.py OUT_DIR
"""

import json
import multiprocessing
import subprocess
import sys
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio_ffmpeg
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/presentation"))
import render_cell_figures as cell
import render_showcase as show

from acoustic_freeform.forward.formation import envelope
from acoustic_freeform.optics.raytrace import trace_pair

S = ROOT / "artifacts/studies/S09-universal-source"
LENSES = {
    "A": dict(run=S / "synthesis-2026-10-26/fig9b", formation=S / "fig9b-verification-2026-10-26/formation-dt00125",
              name="Lens A: Fig. 9b of Silva-Lora & Torres (2020), ×1/14", factor=200),
    "B": dict(run=S / "synthesis-2026-10-26/atlas7-2band",
              formation=S / "atlas7-2band-verification-2026-10-26/formation-dt00125",
              name="Lens B: family lens 7, f/0.79, ×1/14", factor=10),
}
FPS, DIVIDER, RATE, CYCLE_FRAMES = 24, 2000.0, 44100, 16
INK, MUTED, RAY, FACE, GOOD = show.INK, show.MUTED, show.RAY, show.FACE, show.GOOD
mm = 1e3


def ray_fan(data, q, launch, z0, z_top):
    """Meridional rays through the actual faces, vectorized; returns (n, 4, 2) paths in metres."""
    cfg, n = data["cfg"], data["optics"]["indices"]
    z2 = data["conjugates"][2]
    radius = np.abs(launch)
    tr = trace_pair(data["space"], q, n, z0, z2, launch_radius_m=radius)
    hits = tr["intersections_m"]  # (2, n, 2)
    if np.isinf(z0):
        entry = np.c_[radius, np.full(len(radius), cfg.levels_m[0] - 0.4e-3)]
    else:
        d = hits[0] - np.array([0.0, z0])
        entry = np.array([0.0, z0]) + ((cfg.levels_m[0] - 0.4e-3) - z0) / d[:, 1:2] * d
    direction = tr["final_directions"]
    end = hits[1] + ((z_top - hits[1][:, 1]) / direction[:, 1])[:, None] * direction
    paths = np.stack([entry, hits[0], hits[1], end], axis=1)
    paths[:, :, 0] *= np.where(launch < 0, -1, 1)[:, None]
    return paths


def spot(data, q, z0):
    cfg, n = data["cfg"], data["optics"]["indices"]
    r = np.linspace(0, cfg.clear_radius_m * 0.999, 401)
    tr = trace_pair(data["space"], q, n, z0, data["conjugates"][2], launch_radius_m=r)
    return tr["spots_m"], tr["max_radius_m"], tr["rms_radius_m"]


def prepare():
    lenses = {}
    for key, spec in LENSES.items():
        data = cell.load(spec["run"].resolve(), spec["formation"].resolve())
        cfg, space, targets = data["cfg"], data["space"], data["targets"]
        traj = data["trajectory"]
        r = np.linspace(0, cfg.radius_m, 300)
        pupil = np.linspace(0, cfg.clear_radius_m, 201)
        target = [cfg.levels_m[j + 1] + targets[j].evaluate(r) for j in range(2)]
        errors = np.array([[np.max(abs(space.height(j, q[j], pupil) - targets[j].evaluate(pupil))) for j in range(2)]
                           for q in traj["coefficients_m"]])
        field = data["field"]
        xy = field["xy"]
        npts = xy.shape[1]
        tri = np.r_[field["triangles"], field["triangles"] + npts]
        weights = np.array(field["diagnostics"]["band_peaks_pa"])
        lenses[key] = dict(
            data=data, r=r, target=target, errors=errors, times=traj["time_s"], states=traj["coefficients_m"],
            ramp=data["formation"]["ramp_s"], tri=(np.r_[-xy[0], xy[0]] * mm, np.r_[xy[1], xy[1]] * mm, tri),
            bands=[(f, np.r_[p, p]) for f, p in field["bands"]], audio_weights=weights / weights.max(),
            peak=field["diagnostics"]["sampled_pressure_peak_pa"], airy=1.22 * 587.6e-9 *
            (data["conjugates"][2] - cfg.levels_m[2]) / (2 * cfg.clear_radius_m), **spec)
    for key, L in lenses.items():
        start = spot(L["data"], L["states"][0], -np.inf)[1]
        moved = spot(L["data"], L["states"][-1], -12e-3)[1] if key == "B" else 0
        L["wide"] = 1.15 * max(start, moved)
    return lenses


def plan(lenses):
    """Frame list. Each frame names the lens, saved state, drive envelope, source and light progress."""
    frames = []

    def add(count, **kw):
        for i in range(count):
            frames.append(dict(kw, sub=i / max(1, count - 1)))

    B = lenses["B"]
    add(int(1.5 * FPS), lens="A", k=0, drive=0.0, light=None, z0=-np.inf, caption="meniscus")
    add(int(2.0 * FPS), lens="A", k=0, drive=0.0, light="grow", z0=-np.inf, caption="meniscus-light")
    add(int(1.0 * FPS), lens="A", k=0, drive=0.0, light="full", z0=-np.inf, caption="meniscus-light")
    for key in ("A", "B"):
        L = lenses[key]
        change = np.r_[0, np.abs(np.diff(np.log10(L["errors"].max(axis=1))))]
        repeats = np.clip(np.round(change / 0.03), 1, 5).astype(int)
        for k, t in enumerate(L["times"]):
            add(int(repeats[k]), lens=key, k=k, drive=envelope(t, L["ramp"]), light="full", z0=-np.inf,
                caption="forming")
        last = len(L["times"]) - 1
        add(int(1.0 * FPS), lens=key, k=last, drive=1.0, light="full", z0=-np.inf, caption="formed")
        add(int(2.0 * FPS), lens=key, k=last, drive=1.0, light="grow", z0=-np.inf, caption="formed-light")
        add(int(1.0 * FPS), lens=key, k=last, drive=1.0, light="full", z0=-np.inf, caption="formed-light")
        if key == "A":
            add(int(1.0 * FPS), lens="A", k=last, drive=None, light="full", z0=-np.inf, caption="fade")
            add(int(2.5 * FPS), lens=None, k=None, drive=0.0, light=None, z0=None, caption="card")
            add(int(1.5 * FPS), lens="B", k=0, drive=0.0, light="full", z0=-np.inf, caption="meniscus-B")
    # Moving source: inverse object distance swept from 0 (collimated) to 1/12 mm^-1 and back.
    last = len(B["times"]) - 1
    sweep = np.r_[np.linspace(0, 1, int(4 * FPS)), np.ones(int(1.0 * FPS)), np.linspace(1, 0, int(3 * FPS))]
    for s in sweep:
        inverse = (np.sin(np.pi / 2 * s) ** 2) / 12e-3
        frames.append(dict(lens="B", k=last, drive=1.0, light="full", sub=0.0, caption="source",
                           z0=-np.inf if inverse == 0 else -1 / inverse))
    add(int(1.5 * FPS), lens="B", k=last, drive=1.0, light="full", z0=-np.inf, caption="end")
    for i, f in enumerate(frames):
        f["index"] = i
    return frames


CAPTIONS = {
    "meniscus": ("1  The unforced liquid lens",
                 "NOA 61 between two water layers, pinned at two edges.\nWithout sound, tension and gravity give a "
                 "meniscus\nthat misses the stigmatic target by micrometres."),
    "meniscus-light": ("1  Light through the unforced lens",
                       "Collimated light is not brought to a point:\nthe spot is many times the diffraction limit."),
    "forming": ("2  Sound on",
                "The transducers are ramped up over 1 s. The radiation\nstress pushes both faces onto the target; "
                "the height\nerror falls by four orders of magnitude."),
    "formed": ("3  The lens is formed", "Both faces within 1 nm of the Cartesian target,\nheld by the sound."),
    "formed-light": ("3  A stigmatic lens", "All rays meet at one point: the geometric spot sits\ninside the Airy disc."),
    "fade": ("4  Sound off", "The command for lens A is switched off."),
    "card": ("", ""),
    "meniscus-B": ("4  A new lens, same machine", "Same cell, liquids and transducers. Only the pin heights,\n"
                   "the fill volumes and the drive command change."),
    "source": ("5  Moving the light source",
               "A stigmatic lens images one conjugate pair perfectly.\nMoving the point source along the axis "
               "blurs the spot\non the fixed screen: exact ray traces, no animation."),
    "end": ("5  Back to the design conjugate", "Collimated light again focuses to a point."),
}


STATE = {}  # filled before the worker pool forks


def draw(frame):
    lenses, out = STATE["lenses"], STATE["out"]
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    if frame["caption"] == "card":
        fig.text(0.5, 0.58, "Sound off. Pins and fill set for lens B.", ha="center", fontsize=30, color=INK,
                 weight="bold")
        fig.text(0.5, 0.48, "(this change of machine settings is not simulated; lens A would relax back to its "
                 "meniscus)", ha="center", fontsize=15, color=MUTED)
        fig.savefig(out / f"{frame['index']:05d}.png")
        plt.close(fig)
        return
    L = lenses[frame["lens"]]
    data, cfg, space = L["data"], L["data"]["cfg"], L["data"]["space"]
    q, t = L["states"][frame["k"]], L["times"][frame["k"]]
    drive = frame["drive"] if frame["drive"] is not None else 1 - frame["sub"]
    z0 = frame["z0"]
    g = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.0], height_ratios=[0.9, 1.15, 1.0], left=0.04, right=0.97,
                         top=0.84, bottom=0.13, wspace=0.12, hspace=0.45)
    fig.text(0.04, 0.945, "Shaping a liquid lens with sound", fontsize=21, weight="bold", color=INK)
    fig.text(0.04, 0.915, L["name"], fontsize=12.5, color=MUTED)
    tunes = " + ".join(f"{f / DIVIDER:.0f} Hz" for f, _ in L["bands"])
    real = " + ".join(f"{f / 1e6:g} MHz" for f, _ in L["bands"])
    fig.text(0.97, 0.945, f"t = {t:5.3f} s", fontsize=21, ha="right", color=INK, family="monospace")
    fig.text(0.97, 0.915, f"sound {'ON ' if drive > 0 else 'off'} {drive:4.0%}  |  {real}, heard as {tunes}",
             fontsize=11.5, ha="right", color=INK if drive > 0 else MUTED)
    # Section with pressure, faces and rays.
    ax = fig.add_subplot(g[:, 0])
    X, Y, tri = L["tri"]
    phase = 2 * np.pi * frame["index"] / CYCLE_FRAMES
    f0 = L["bands"][-1][0]
    pressure = sum(np.real(p * np.exp(-1j * phase * f / f0)) for f, p in L["bands"]) * drive
    limit = 0.8 * L["peak"]
    ax.tripcolor(mtri.Triangulation(X, Y, tri), pressure, shading="gouraud", cmap="RdBu_r", vmin=-limit,
                 vmax=limit, rasterized=True, zorder=0)
    r, factor = L["r"], L["factor"]
    shown = [L["target"][j] + factor * (cfg.levels_m[j + 1] + space.height(j, q[j], r) - L["target"][j])
             for j in range(2)]
    signed = np.r_[-r[::-1], r] * mm

    def mirror(h):
        return np.r_[h[::-1], h] * mm

    ax.fill_between(signed, mirror(shown[0]), mirror(shown[1]), color="#f3c77a", alpha=0.55, lw=0, zorder=1)
    for j in range(2):
        ax.plot(signed, mirror(L["target"][j]), color=INK, lw=0.9, ls=(0, (4, 3)), zorder=3)
        ax.plot(signed, mirror(shown[j]), color=FACE, lw=2.4, zorder=3)
    R, lv = cfg.radius_m * mm, np.asarray(cfg.levels_m) * mm
    ax.plot([-R, -R, R, R, -R], [lv[0], lv[3], lv[3], lv[0], lv[0]], color=INK, lw=3, zorder=4)
    for j in (1, 2):  # pinning edges
        ax.plot([-R, -R + 0.08], [lv[j]] * 2, color=INK, lw=3, zorder=4)
        ax.plot([R, R - 0.08], [lv[j]] * 2, color=INK, lw=3, zorder=4)
    z_top = 7.2e-3
    if frame["light"] is not None:
        launch = np.linspace(-0.92, 0.92, 21) * cfg.clear_radius_m
        paths = ray_fan(data, q, launch, z0, z_top)
        for j in (1, 2):  # refraction points drawn on the magnified faces
            paths[:, j, 1] = np.interp(np.abs(paths[:, j, 0]), r, shown[j - 1])
        progress = 1.0 if frame["light"] == "full" else frame["sub"]
        for path in paths * mm:
            seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
            cum = np.r_[0, np.cumsum(seg)]
            reach = progress * cum[-1]
            keep = cum <= reach
            pts = path[keep]
            if reach < cum[-1]:
                i = np.searchsorted(cum, reach) - 1
                tip = path[i] + (reach - cum[i]) / seg[i] * (path[i + 1] - path[i])
                pts = np.vstack([pts, tip])
                ax.plot(*tip, "o", color="#ffd400", ms=4, mec=RAY, mew=0.6, zorder=6)
            ax.plot(pts[:, 0], pts[:, 1], color=RAY, lw=1.0, alpha=0.9, zorder=5)
    if not np.isinf(z0):
        ax.annotate(f"point source at z = {z0 * mm:.1f} mm", xy=(0, lv[0] - 0.42), xytext=(0, lv[0] - 0.9),
                    ha="center", fontsize=11, color=RAY, arrowprops=dict(arrowstyle="->", color=RAY))
    else:
        ax.text(0, lv[0] - 0.75, "collimated light", ha="center", fontsize=11, color=RAY)
    ax.plot(0, data["conjugates"][2] * mm, marker="_", ms=26, color=INK, mew=1.5, zorder=7)
    ax.text(0.25, data["conjugates"][2] * mm, "screen (design image plane)", va="center", fontsize=10, color=MUTED)
    ax.set_xlim(-R - 0.35, R + 0.35)
    ax.set_ylim(lv[0] - 1.0, z_top * mm)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(f"Faces: target (dashed) + deviation × {factor}.  Colour: acoustic pressure, ±{limit / 1e6:.1f} MPa,"
                 f" oscillation slowed ~10⁶×", loc="left", fontsize=10.5, color=INK)
    # Caption.
    c = fig.add_subplot(g[0, 1])
    c.axis("off")
    title, body = CAPTIONS[frame["caption"]]
    c.text(0, 0.95, title, fontsize=18, weight="bold", color=INK, va="top")
    c.text(0, 0.62, body, fontsize=13, color=INK, va="top", linespacing=1.5)
    # Spot at two scales.
    spots, smax, srms = spot(data, q, z0)
    pair = g[1, 1].subgridspec(1, 2, wspace=0.35)
    for column, (lim, label) in enumerate(((L["wide"], "wide"), (3 * L["airy"], "diffraction scale"))):
        a = fig.add_subplot(pair[0, column])
        a.add_patch(plt.Circle((0, 0), L["airy"] * 1e6, color=GOOD, alpha=0.3, lw=0))
        a.scatter(spots[:, 0] * 1e6, spots[:, 1] * 1e6, s=4, color=RAY, alpha=0.6, lw=0)
        a.set_xlim(-lim * 1e6, lim * 1e6)
        a.set_ylim(-lim * 1e6, lim * 1e6)
        a.set_aspect("equal")
        a.set_xlabel(f"screen x (µm), {label}", fontsize=10)
        a.tick_params(labelsize=8)
    c.text(0, -0.08, f"Spot on the screen: max {smax * 1e6:7.2f} µm, rms {srms * 1e6:6.3f} µm\n"
           f"green: Airy disc, radius {L['airy'] * 1e6:.2f} µm at 587.6 nm", fontsize=11.5, color=INK, va="top")
    # Height error.
    b = fig.add_subplot(g[2, 1])
    k = frame["k"]
    for j, (label, colour) in enumerate((("front face", FACE), ("back face", show.BACK))):
        b.semilogy(L["times"], L["errors"][:, j] * 1e9, color=colour, lw=0.8, alpha=0.25)
        b.semilogy(L["times"][: k + 1], L["errors"][: k + 1, j] * 1e9, color=colour, lw=2, label=label)
    b.axhline(10, color=GOOD, ls="--", lw=1)
    b.axvline(t, color=MUTED, lw=0.8)
    b.set_xlabel("physical time (s)")
    b.set_ylabel("max height error (nm)")
    b.legend(fontsize=9, frameon=False, loc="upper right")
    fig.text(0.04, 0.02, "Numerical model (axisymmetric, lossless acoustics, creeping flow), not an experiment. "
             "Faces and their motion are saved states of the forward trajectories (no interpolation). Rays use "
             "the published indices 1/1.7/1.\nPressure: harmonic field at the formed state × drive envelope. Audio: "
             f"the drive with frequencies divided by {DIVIDER:.0f}, loudness following the envelope.",
             fontsize=9, color=MUTED)
    fig.savefig(out / f"{frame['index']:05d}.png")
    plt.close(fig)


def audio(frames, lenses, path):
    samples_per_frame = RATE // FPS
    signal, phases = [], {}
    for frame in frames:
        chunk = np.zeros(samples_per_frame)
        if frame["lens"] is not None:
            L = lenses[frame["lens"]]
            drive = frame["drive"] if frame["drive"] is not None else 1 - frame["sub"]
            for (f, _), w in zip(L["bands"], L["audio_weights"]):
                tone = f / DIVIDER
                start = phases.get(tone, 0.0)
                ph = start + 2 * np.pi * tone * np.arange(1, samples_per_frame + 1) / RATE
                chunk += drive * w * np.sin(ph)
                phases[tone] = ph[-1]
        signal.append(chunk)
    signal = np.concatenate(signal)
    signal = 0.35 * signal / max(1e-9, np.max(np.abs(signal)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((signal * 32767).astype(np.int16).tobytes())


def main():
    out = Path(sys.argv[1]).resolve()
    pngs = out / "frames"
    pngs.mkdir(parents=True, exist_ok=True)
    lenses = prepare()
    frames = plan(lenses)
    print(f"{len(frames)} frames, {len(frames) / FPS:.1f} s", flush=True)
    only = [int(x) for x in sys.argv[2:]]
    STATE.update(lenses=lenses, out=pngs)
    jobs = [f for f in frames if not only or f["index"] in only]
    with multiprocessing.get_context("fork").Pool(4) as pool:
        for i, _ in enumerate(pool.imap_unordered(draw, jobs, chunksize=4)):
            if i % 50 == 0:
                print(f"frame {i}/{len(jobs)}", flush=True)
    if only:
        return
    wav = out / "story-sonified.wav"
    audio(frames, lenses, wav)
    mp4 = out / "story-with-sound.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(pngs / "%05d.png"), "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-shortest", str(mp4)], check=True)
    (out / "story.json").write_text(json.dumps({
        "inputs_sha256": cell.inputs_used, "frames": len(frames), "fps": FPS, "audio_divider": DIVIDER,
        "magnification": {k: v["factor"] for k, v in LENSES.items()},
        "not_simulated": "the change of pins and fill between lens A and lens B",
        "scope": "Rendering of saved numerical trajectories; not an experiment.",
    }, indent=2) + "\n")
    print(mp4)


if __name__ == "__main__":
    main()
