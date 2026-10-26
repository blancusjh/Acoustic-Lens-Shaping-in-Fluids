"""Animate how the sources emit sound into the cell, with a sonified soundtrack.

The solver is time-harmonic, so the switch-on is built by linear Fourier
synthesis: the saved design command is solved at frequencies f0 + k df around
the drive frequency (same geometry, same commanded source velocities), and a
smoothly ramped tone burst A(t) cos(2 pi f0 t) is synthesized as
p(x, t) = Re sum_k c_k P_k(x) exp(-i 2 pi (f0 + k df) t), with c_k the Fourier
coefficients of the envelope A. This is the exact linear response of the
declared lossless interior with its absorbing source boundaries to that burst,
periodic in T = 1/df; only the first half of the period is shown.

The soundtrack is a sonification, not a recording: the real field at f0 is
ultrasonic and inaudible. The pressure at a "microphone" point inside the
resin is frequency-divided (f0 -> f0 / DIVIDER) while its envelope is played
on the video's slowed time axis.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import wave
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/presentation"))
import render_cell_figures as cell

WATER, RESIN, FACE, INK, MUTED = "#4f8fc0", "#e3a33b", "#7a4a06", "#20262c", "#66727c"


def envelope(t, rise_s, hold_s):
    """Raised-cosine rise, hold, raised-cosine fall; periodic burst for the synthesis."""
    up = np.clip(t / rise_s, 0, 1)
    down = np.clip((rise_s + hold_s + rise_s - t) / rise_s, 0, 1)
    return np.minimum(0.5 - 0.5 * np.cos(np.pi * up), 0.5 - 0.5 * np.cos(np.pi * down))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--formation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", default="emission")
    parser.add_argument("--period-us", type=float, default=80.0, help="synthesis period T = 1/df")
    parser.add_argument("--band", type=float, default=0.2, help="half-band as a fraction of f0")
    parser.add_argument("--rise-cycles", type=float, default=8.0)
    parser.add_argument("--frames-per-cycle", type=int, default=6)
    parser.add_argument("--show-us", type=float, default=36.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--divider", type=float, default=2000.0)
    parser.add_argument("--exaggeration", type=float, default=15000.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from acoustic_freeform.acoustics.helmholtz import DualAcoustics
    from acoustic_freeform.apparatus.sources import source_regions

    data = cell.load(args.run.resolve(), args.formation.resolve())
    cfg, space, q, drive = data["cfg"], data["space"], data["q"], data["drive"]
    f0, mm = cfg.frequency_hz, 1e3
    period = args.period_us * 1e-6
    df = 1 / period
    ks = np.arange(-round(args.band * f0 / df), round(args.band * f0 / df) + 1)
    print(f"{len(ks)} frequencies, df = {df / 1e3:.1f} kHz", flush=True)
    fields = []
    for k in ks:
        wave_k = DualAcoustics(replace(cfg, frequency_hz=f0 + k * df), space, linear_solver="static_condensed")
        response = wave_k.solve(q, drive)
        fields.append(response.solution[response.basis.nodal_dofs[0], 0])
    fields = np.array(fields)
    xy = data["field"]["xy"] * mm
    triangles = data["field"]["triangles"]
    # Envelope Fourier coefficients on the synthesis period.
    rise = args.rise_cycles / f0
    hold = period / 2 - 2 * rise
    samples = 1 << 16
    t_grid = np.arange(samples) * period / samples
    spectrum = np.fft.fft(envelope(t_grid, rise, hold)) / samples
    coefficients = spectrum[ks % samples] * np.exp(0j)

    def baseband(t):
        """Complex envelope E(x, t) with p = Re[E exp(-i w0 t)], all nodes."""
        phase = np.exp(-2j * np.pi * np.outer(t, ks * df))
        return (phase * coefficients) @ fields

    # Microphone: on the axis, half-way between the two vertices (inside the resin).
    z_front = cfg.levels_m[1] + space.height(0, q[0], np.array([0.0]))[0]
    z_back = cfg.levels_m[2] + space.height(1, q[1], np.array([0.0]))[0]
    mic_z = 0.5 * (z_front + z_back) * mm
    mic = int(np.argmin((xy[0]) ** 2 + (xy[1] - mic_z) ** 2))
    shown = args.show_us * 1e-6
    t_frames = np.arange(0, shown, 1 / (f0 * args.frames_per_cycle))
    t_trace = np.linspace(0, shown, 4000)
    trace_env = baseband(t_trace)[:, mic]
    trace = np.real(trace_env * np.exp(-2j * np.pi * f0 * t_trace))
    steady = np.max(abs(fields[ks == 0][0]))
    limit = 1.05 * steady / 1e6
    # Source geometry and commanded velocities (displacement u = Re[i g / w e^{-iwt}]).
    regions = source_regions(cfg)
    omega = 2 * np.pi * f0
    r = np.linspace(0, cfg.radius_m, 400)
    faces = [(cfg.levels_m[j + 1] + space.height(j, q[j], r)) * mm for j in range(2)]
    signed = np.r_[-r[::-1], r] * mm
    tri = mtri.Triangulation(np.r_[xy[0], -xy[0]], np.r_[xy[1], xy[1]],
                             np.r_[triangles.T, triangles.T + xy.shape[1]].reshape(-1, 3))
    R, L = cfg.radius_m * mm, np.asarray(cfg.levels_m) * mm
    frames = []
    for n, t in enumerate(t_frames):
        E = baseband(np.array([t]))[0]
        p = np.real(E * np.exp(-2j * np.pi * f0 * t)) / 1e6
        amplitude = envelope(np.array([t]), rise, hold)[0]
        carrier = np.exp(-1j * omega * t)
        fig = plt.figure(figsize=(14, 7.4))
        g = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1], wspace=0.22, hspace=0.42)
        ax = fig.add_subplot(g[:, 0])
        ax.tripcolor(tri, np.r_[p, p], shading="gouraud", cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
        for z in faces:
            ax.plot(signed, np.r_[z[::-1], z], color=FACE, lw=1.6)
        # Moving source surfaces: windows (lines) and wall (both sides), exaggerated.
        for reg in regions:
            u = args.exaggeration * amplitude * np.real(1j * drive[reg["channel"]] / omega * carrier) * mm
            if reg["boundary"] == "side":
                z0, z1 = reg["z_min_m"] * mm, reg["z_max_m"] * mm
                for sign in (1, -1):
                    ax.plot([sign * (R - u)] * 2, [z0, z1], color="#111", lw=2.4, solid_capstyle="butt")
            else:
                base = L[0] if reg["boundary"] == "bottom" else L[3]
                into = 1 if reg["boundary"] == "bottom" else -1
                for a, b in ((reg["r_min_m"], reg["r_max_m"]), (-reg["r_max_m"], -reg["r_min_m"])):
                    ax.plot([a * mm, b * mm], [base + into * u] * 2, color="#111", lw=2.4, solid_capstyle="butt")
        ax.plot(0, mic_z, marker="o", ms=7, mfc="white", mec="black")
        ax.annotate("microphone", (0, mic_z), (0.45 * R, mic_z + 0.25), fontsize=9,
                    arrowprops={"arrowstyle": "-", "lw": 0.7})
        ax.set(xlim=(-R - 0.25, R + 0.25), ylim=(L[0] - 0.25, L[3] + 0.25), aspect="equal",
               xlabel="x (mm)", ylabel="z (mm)")
        ax.set_title(f"acoustic pressure, colour range ±{limit:.2f} MPa\nblack: transducer surfaces, motion ×{args.exaggeration:,.0f}",
                     loc="left", fontsize=10)
        a = fig.add_subplot(g[0, 1])
        upto = t_trace <= t
        a.plot(t_trace * 1e6, trace / 1e6, color="#bbb", lw=0.5)
        a.plot(t_trace[upto] * 1e6, trace[upto] / 1e6, color="#c62828", lw=0.8)
        a.plot(t_trace * 1e6, envelope(t_trace, rise, hold) * np.max(abs(trace_env)) / 1e6, color=MUTED, lw=0.8, ls="--")
        a.axvline(t * 1e6, color=INK, lw=0.8)
        a.set(xlabel="physical time (µs)", ylabel="pressure (MPa)", xlim=(0, shown * 1e6))
        a.set_title("pressure at the microphone (inside the resin)", loc="left", fontsize=10)
        b = fig.add_subplot(g[1, 1])
        b.axis("off")
        b.text(0, 0.95,
               f"Drive: {f0 / 1e6:.2f} MHz, one tone, {len(drive)} source regions.\n"
               f"Switch-on: {args.rise_cycles:.0f}-cycle smooth ramp of the same command.\n\n"
               f"Real sound: ultrasonic ({f0 / 1e6:.2f} MHz is {f0 / 20e3:.0f}× above the\n"
               "~20 kHz limit of human hearing): the cell is silent.\n\n"
               f"Soundtrack: the microphone signal divided by {args.divider:.0f} in pitch\n"
               f"({f0 / 1e6:.2f} MHz → {f0 / args.divider:.0f} Hz), loudness following the video.\n\n"
               "Linear Fourier synthesis of the time-harmonic model\n"
               f"({len(ks)} frequencies); interfaces held at the design shape.\nNumerical model, not an experiment.",
               va="top", fontsize=9.5, color=INK)
        slow = (len(t_frames) / args.fps) / shown
        fig.suptitle(f"How the sources emit: t = {t * 1e6:6.2f} µs    (shown {slow:.1e}× slower than real time)",
                     x=0.02, ha="left", fontsize=13)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        if n % 40 == 0:
            print(f"frame {n}/{len(t_frames)}", flush=True)
    silent = args.out / f"{args.prefix}-silent.mp4"
    imageio.mimsave(silent, frames, fps=args.fps, quality=8, macro_block_size=1)
    # Sonification on the video's time axis: tau (s of video) <-> t = tau / slow.
    rate, duration = 44100, len(frames) / args.fps
    tau = np.arange(int(rate * duration)) / rate
    env_audio = np.interp(tau / slow, t_trace, trace_env.real) + 1j * np.interp(tau / slow, t_trace, trace_env.imag)
    audio = np.real(env_audio * np.exp(-2j * np.pi * (f0 / args.divider) * tau))
    audio = audio / max(np.max(abs(audio)), 1e-30) * 0.6
    wav = args.out / f"{args.prefix}-sonified.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((audio * 32767).astype("<i2").tobytes())
    final = args.out / f"{args.prefix}.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(final)], check=True)
    silent.unlink()
    (args.out / f"{args.prefix}-provenance.json").write_text(json.dumps({
        "inputs_sha256": cell.inputs_used,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frequencies_hz": (f0 + ks * df).tolist(),
        "microphone_m": [0.0, mic_z / mm],
        "slow_down": slow, "audio_divider": args.divider,
        "scope": "Linear synthesis of the time-harmonic model; sonification, not a recording.",
    }, indent=2) + "\n")
    print(json.dumps({"video": str(final), "frames": len(frames), "slow_down": slow}))


if __name__ == "__main__":
    main()
