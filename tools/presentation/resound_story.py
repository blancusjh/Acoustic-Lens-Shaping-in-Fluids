"""Replace the story soundtrack with what an outside listener would hear (sonified).

Like a Chladni plate heard from a distance, every point of the cell boundary
contributes with its own amplitude and phase. For each drive band the saved
harmonic field gives p and dp/dn on the outer boundary of the meridional mesh
(both windows and the wall); the Kirchhoff-Helmholtz integral

    p(x) = sum over boundary of [p dG/dn - G dp/dn] dS,  G = exp(ikR) / (4 pi R),

carries it to a listener on the axis above the cell. The model has no medium
outside the cell, so the cell is assumed to sit in a water bath with the outer
layer's sound speed (an added assumption). For an on-axis listener the azimuthal
integral is exact: R and n.(x - y) do not depend on the azimuth, so dS = 2 pi r ds.
dp/dn is the gradient of the linear interpolant on the boundary triangle.

The soundtrack keeps the story's frame plan: each band is played at its
frequency divided by 2000 with its listener amplitude and phase, times the
drive envelope. Loudness is comparable between lens A and lens B.
Usage: resound_story.py STORY_DIR [LISTENER_Z_M]
"""

import json
import subprocess
import sys
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_story as story

from acoustic_freeform.acoustics.radiation import listener_pressure


def main():
    out = Path(sys.argv[1]).resolve()
    z_listener = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    lenses = story.prepare()
    frames = story.plan(lenses)
    report = {}
    for key, L in lenses.items():
        cfg, field = L["data"]["cfg"], L["data"]["field"]
        c = np.atleast_1d(cfg.sound_speed_m_s)[0]
        L["listener"] = []
        for f, pressure in field["bands"]:
            k = 2 * np.pi * f / c
            near = listener_pressure(field["xy"], field["triangles"], pressure, k, z_listener)
            far = listener_pressure(field["xy"], field["triangles"], pressure, k, 2 * z_listener)
            L["listener"].append(near)
            report.setdefault(key, []).append({
                "band_hz": f, "listener_pressure_pa": abs(near),
                # Far-field check: doubling the distance should halve the amplitude.
                "distance_doubling_ratio": abs(far) / abs(near),
            })
    peak = max(abs(sum(abs(p) for p in L["listener"])) for L in lenses.values())
    rate, per = story.RATE, story.RATE // story.FPS
    signal, clock = [], 0
    for frame in frames:
        time = (clock + np.arange(per)) / rate
        clock += per
        chunk = np.zeros(per)
        if frame["lens"] is not None:
            L = lenses[frame["lens"]]
            drive = frame["drive"] if frame["drive"] is not None else 1 - frame["sub"]
            for (f, _), p in zip(L["bands"], L["listener"]):
                chunk += drive * abs(p) / peak * np.cos(2 * np.pi * f / story.DIVIDER * time - np.angle(p))
        signal.append(chunk)
    signal = 0.5 * np.concatenate(signal)
    wav = out / "story-listener.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((signal * 32767).astype(np.int16).tobytes())
    mp4 = out / "story-with-listener-sound.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(out / "story-with-sound.mp4"),
                    "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(mp4)], check=True)
    (out / "story-listener.json").write_text(json.dumps({
        "listener": {"r_m": 0.0, "z_m": z_listener, "medium": "assumed water bath, outer-layer sound speed"},
        "bands": report, "audio_divider": story.DIVIDER,
        "scope": "Sonification of the time-harmonic model through a Kirchhoff-Helmholtz integral; not a recording.",
    }, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    print(mp4)


if __name__ == "__main__":
    main()
