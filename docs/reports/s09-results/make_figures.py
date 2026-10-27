"""Figures for the S09 results report, read from the saved S09 records.

Reads (read-only) the JSON records under artifacts/studies/S09-universal-source/
and writes PNGs to artifacts/reports/s09-results/figures/ (private, not
committed). Plots only; no physics is recomputed. Runs in seconds.

    uv run python docs/reports/s09-results/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
S09 = ROOT / "artifacts/studies/S09-universal-source"
OUT = ROOT / "artifacts/reports/s09-results/figures"
NM = 1e9

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
ONE, TWO, FAIL, GREY = "#1b7837", "#762a83", "#b2182b", "#777777"


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atlas():
    d = S09 / "atlas-nonlinear-2026-10-27"
    rows = [r for r in jsonl(d / "atlas-nonlinear.jsonl") if r["status"] == "checked"]
    follow = {r["index"]: r for r in jsonl(d / "followup-two-bands.jsonl")}
    rows.sort(key=lambda r: r["index"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    for r in rows:
        err = max(r["nonlinear_pupil_error_m"]) * NM
        two = len(r["bands_hz"]) == 2
        failed = not r["branch_reached_end"]
        ax.plot(r["index"], err, "x" if failed else ("s" if two else "o"), ms=4,
                color=FAIL if failed else (TWO if two else ONE), mfc="none" if two else None)
        if r["index"] in follow:
            f = follow[r["index"]]
            ferr = max(f["nonlinear_pupil_error_m"]) * NM
            ax.annotate("", xy=(r["index"], ferr), xytext=(r["index"], err),
                        arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8})
            ax.plot(r["index"], ferr, "D", ms=4, color=FAIL if ferr > 10 else TWO)
            ax.text(r["index"] + 0.8, ferr, str(r["index"]), fontsize=7, va="center", color=GREY)
        if failed:
            ax.text(r["index"] + 0.8, err, str(r["index"]), fontsize=7, va="center", color=GREY)
    for level, label in ((1, "1 nm"), (10, "10 nm goal")):
        ax.axhline(level, color=GREY, lw=0.6, ls="--")
        ax.text(28, level, label, fontsize=7, va="bottom", ha="center", color=GREY)
    ax.set_yscale("log")
    ax.set_xlim(-1, 82)
    ax.set_xlabel("atlas lens index (Silva-Lora and Torres family, 68 that fit the cell)")
    ax.set_ylabel("start-up end, max pupil height error (nm)")
    ax.set_title("(a) Start-up branch from the acoustics-off meniscus, per lens", loc="left", fontsize=9)
    handles = [
        plt.Line2D([], [], ls="", marker="o", color=ONE, label="one tone, 1.5 MHz"),
        plt.Line2D([], [], ls="", marker="s", mfc="none", color=TWO, label="two tones, 0.75 + 1.5 MHz"),
        plt.Line2D([], [], ls="", marker="x", color=FAIL, label="fold before full amplitude (lens 57)"),
        plt.Line2D([], [], ls="", marker="D", color=TWO, label="follow-up with two tones"),
        plt.Line2D([], [], ls="", marker="D", color=FAIL, label="follow-up ends on another equilibrium"),
    ]
    ax.set_ylim(1e-2, 1e9)
    ax.legend(handles=handles, fontsize=7, loc="upper left", ncol=2, frameon=False)
    frozen = np.array([r["refined_frozen_pupil_error_m"] for r in rows]) * NM
    nonlin = np.array([max(r["nonlinear_pupil_error_m"]) for r in rows]) * NM
    ok = np.array([r["branch_reached_end"] for r in rows])
    two = np.array([len(r["bands_hz"]) == 2 for r in rows])
    bx.loglog(frozen[ok & ~two], nonlin[ok & ~two], "o", ms=3, color=ONE)
    bx.loglog(frozen[ok & two], nonlin[ok & two], "s", ms=3, mfc="none", color=TWO)
    bx.loglog(frozen[~ok], nonlin[~ok], "x", ms=5, color=FAIL)
    lim = [5e-4, 1e2]
    bx.plot(lim, lim, color=GREY, lw=0.6)
    bx.set_xlim(lim)
    bx.set_ylim(lim)
    bx.set_xlabel("frozen (first-order) error of the refined command (nm)")
    bx.set_ylabel("start-up end error (nm)")
    bx.set_title("(b) First-order vs. nonlinear, same command", loc="left", fontsize=9)
    fig.text(0.0, -0.04, "Numerical model only (axisymmetric, lossless acoustics, creeping flow); log scales. "
             "Source: atlas-nonlinear.jsonl and followup-two-bands.jsonl.", fontsize=7, color=GREY)
    fig.savefig(OUT / "atlas-nonlinear.png")
    plt.close(fig)


def channels():
    rows = jsonl(S09 / "channels-2026-10-27/channel-study.jsonl")
    names = {"fig9b": "Fig. 9b", "fig7a": "Fig. 7a", "atlas22": "lens 22", "atlas45": "lens 45",
             "atlas7": "lens 7 (two bands)"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for i, (key, label) in enumerate(names.items()):
        sel = sorted((r for r in rows if r["lens"] == key), key=lambda r: r["channels"])
        n = [r["channels"] for r in sel]
        color = plt.cm.tab10(i)
        ax.plot(n, [r["convex_pupil_error_m"] * NM for r in sel], "o-", ms=3, color=color, label=label)
        bx.plot(n, [r["single_tone_pupil_error_m"] * NM for r in sel], "o-", ms=3, color=color, label=label)
    for a, title in ((ax, "(a) Convex bound (any drive of the bands)"),
                     (bx, "(b) One tone per band, refined")):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xticks([14, 21, 28, 42, 56, 112])
        a.set_xticklabels(["14", "21", "28", "42", "56", "112"])
        a.minorticks_off()
        a.axhline(10, color=GREY, lw=0.6, ls="--")
        a.axhline(1, color=GREY, lw=0.6, ls=":")
        a.set_xlabel("source channels")
        a.set_title(title, loc="left", fontsize=9)
    ax.set_ylabel("frozen max pupil height error (nm)")
    bx.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.text(0.0, -0.13, "Channels = 2 x rings per window + 3 x wall bands per layer. First-order (frozen-compliance) "
             "results at 1 W and 2 MPa per band;\nbelow 26 channels the mode count equals the channel count. "
             "Dashed: 10 nm goal; dotted: 1 nm. Source: channel-study.jsonl.", fontsize=7, color=GREY)
    fig.savefig(OUT / "channel-study.png")
    plt.close(fig)


def formation():
    runs = {
        "Fig. 9b": "fig9b-verification-2026-10-26",
        "Fig. 7a, unconstrained": "fig7a-verification-2026-10-26",
        "Fig. 7a, fold margin 0.7": "fig7a-margin0p7-verification-2026-10-26",
        "lens 7, two bands": "atlas7-2band-verification-2026-10-26",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for i, (label, d) in enumerate(runs.items()):
        color = plt.cm.tab10(i)
        for dt, ls in (("formation-dt0025", "-"), ("formation-dt00125", "--")):
            obs = json.loads((S09 / d / dt / "observations.json").read_text())
            t = np.array([o["time_s"] for o in obs])
            e = np.array([o["sampled_pupil_error_m"] for o in obs]) * NM
            step = "25 ms" if dt.endswith("0025") else "12.5 ms"
            axes[0].plot(t, e[:, 0], ls, color=color, lw=1, label=f"{label}, dt {step}")
            axes[1].plot(t, e[:, 1], ls, color=color, lw=1)
    for a, face in zip(axes, ("front", "back")):
        a.set_yscale("log")
        a.axvline(1.0, color=GREY, lw=0.6, ls=":")
        a.axhline(10, color=GREY, lw=0.6, ls="--")
        a.set_xlabel("physical time (s); drive ramp ends at 1 s")
        a.set_title(f"{face} face", loc="left", fontsize=9)
    axes[0].set_ylabel("max pupil height error (nm)")
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.text(0.0, -0.06, "Forward creeping-flow formation from the acoustics-off meniscus under the held command "
             "(linearized acoustic load, 64x12 fluid mesh); saved states only, nothing interpolated. "
             "Dashed: 10 nm goal.", fontsize=7, color=GREY)
    fig.savefig(OUT / "formation-histories.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    atlas()
    channels()
    formation()
    print(OUT)
