# S09 — One common machine for many stigmatic singlets

**Question.** With the cell, liquids and source layout fixed, and the lens chosen
only by the drive command (and the fill pumps), which stigmatic singlets
(SOSLs) can be formed faithfully, and which extra source freedom extends the
set most cheaply?

**Method.** Theory section `docs/theory/sections/10f_universal_source.tex`.
(1) *Load demand*: the extra traction a lens face needs, exact
(`mechanics/demand.py`) and as a series with the conic matching rule
Δρ g/σ = 4 k c². (2) *Capability*: time-averaged forces of tones at distinct
frequencies add, so the forces a fixed source can deliver form a convex cone
and the best pupil tolerance within a power cap is a semidefinite program
(`inverse/capability.py`); the fill pumps enter as free volume columns, and a
static fold margin sym(K − A(X)) ⪰ μK is a linear matrix inequality. The
program's leading eigenvectors are refined to one tone per band. (3) Every
design reported below is then accepted only through the nonlinear chain:
start-up branch from the acoustics-off meniscus, fixed-command refinement,
coupled stability, formation at two time steps.

**Common machine.** Water / NOA 61 / water; bore radius 2.02 mm (clear aperture
4 mm); windows at z = −0.8 and +2.2286 mm for every lens; 112 source regions
driven through the same 26 continuous modes (8 Bessel modes per window, 10
wall cosines); caps 1 W and 2 MPa per band unless stated. One mechanical
setting still follows the lens: the heights of the two pinning edges (the lens
rim heights). The fill pumps set each layer to the lens's own volume.

## Checks of the theory

- Applying the exact demanded traction to the finite-element meniscus
  reproduces both faces of Fig. 9b and Fig. 7a to 3e-11 m; the two
  discretizations of the demand agree to 2e-3 (`demand_check.py`).
- The series Q2 r² + Q4 r⁴ matches the exact demand to 6e-4 for r ≤ 0.15 R.
- The matching rule predicts the S08 brine / oil choice for Fig. 9b
  (−8.3e4 vs −7.9e4 m⁻², 1.0e6 vs 7.5e5 m⁻²) and explains why Fig. 7a fails
  in water with weak fields (its back face needs the opposite sign).
- Finite-difference acoustic shape stiffness agrees with the solver's analytic
  shape Jacobian to 1.7e-5.

## The published lenses in the plain water cell, one tone at 1.5 MHz

| | Fig. 9b | Fig. 7a, unconstrained | Fig. 7a, fold margin ≥ 0.7 |
|---|---:|---:|---:|
| Acoustics-off meniscus error (front / back) | 0.77 / 2.07 µm | 19.4 / 23.1 µm | same |
| Source power, sampled peak pressure | 0.17 W, 0.55 MPa | 0.91 W, 1.86 MPa | 1.12 W, 2.10 MPa |
| Fold margin along the start-up (minimum) | 0.96 | 0.068 | 0.94 |
| Frozen-to-coupled amplification | 1.05 | 18.5 | 1.09 |
| Slowest creeping-flow decay (128×24 mesh) | −41.4 1/s | −7.4 1/s | −96.9 1/s |
| Fixed-command refinement, max change | 0.18 nm | 11 nm (P3) | 2.6 nm (P3), else ≤ 0.5 nm |
| Formed, pupil max height error (front / back) | 0.05 / 0.04 nm | 14.2 / 0.4 nm (dt 25 ms), 10.7 / 0.4 nm (12.5 ms) | 0.53 / 0.41 nm (both steps) |
| Spot RMS / max, published indices | 0.004 / 0.071 µm | 0.033 / 0.37 µm | 0.027 / 0.45 µm |
| Airy radius at 587.6 nm | 1.02 µm | 0.68 µm | 0.68 µm |

Heights below about 0.2 nm (9b) and 0.5 nm (7a) are below the model's own
discretization resolution; the geometric spots are far inside the Airy disc,
i.e. the formed lenses are diffraction-limited in this model.
Spots here launch rays through points on the pin plane, a pupil slightly
inside the 2 mm clear aperture. Launched through the front face out to the full
clear radius (as in the showcase videos), the Fig. 7a margin-0.7 maximum is
0.86 µm (RMS 0.03 µm, wavefront PV about λ/560 at 1 nm), slightly beyond the
Airy radius at the extreme edge ray; the 9b and lens-7 values are unchanged.

**Sensitivity** (same command, coupled first-order shift, front / back):

| Perturbation | Fig. 9b | Fig. 7a (margin 0.7) |
|---|---:|---:|
| Water sound speed +0.1 % | 28 / 47 nm | 299 / 330 nm |
| Resin sound speed +1 % | 93 / 553 nm | 2.2 / 3.6 µm |
| Frequency +100 ppm | 2.0 / 0.3 nm | 50 / 67 nm |
| Interfacial tension +1 % | 38 / 25 nm | 161 / 113 nm |

Either design needs closed-loop trim against a measured wavefront; the resin
sound speed is still a placeholder.

**Correction of S08.** S08 concluded that no reasonable field reaches Fig. 9b in
water (674 nm at 10 W). That was a local minimum of the inverse started from
frozen fits of random seeds; the convex start reaches it with 0.17 W.

## Atlas: 81 singlets on the common machine

`tools/analysis/universal/atlas.py`, figure `figures/s09-atlas.png`
(`render_atlas.py`). Silva-Lora and Torres singlets n = 1 / 1.7 / 1 with object
at ∞, −300, −150 mm; image at 60, 90, 120 mm; thickness 10, 15, 20 mm; shape
factor −0.5, 0, +0.5 (paper units), scaled 1/14. Coverage means the refined
command reaches 10 nm on both pupils within the caps (first-order capability;
nonlinear checks of individual atlas lenses below).

| Source (one tone per band, 26 modes, ≤ 1 W, ≤ 2 MPa per band) | Lenses formed (of 68 that fit the cell) |
|---|---:|
| one tone at 1.5 MHz | 58 |
| one tone at 0.75 MHz + one at 1.5 MHz | 10 (lenses 6, 7, 30, 31, 33, 34, 42, 58, 60, 61) |
| not formed | 0 |
| do not fit the common cell (faces crossing at the rim or too close to a window) | 13 |

The convex multi-tone bound reaches the goal for all 68 lenses within 1 W
(worst 0.46 nm at a single band), so every lens that fits the cell is
reachable in principle; the second band is needed only to realize it with one
tone per band.

**Nonlinear spot checks of the atlas.** Two lenses that the first (defective)
pass called hard were taken through the nonlinear chain from the
acoustics-off meniscus at the common machine's settings:

| | Lens 45 (one tone) | Lens 7 (two bands) |
|---|---:|---:|
| Lens | object −300, image 120, t 10, σ −0.5 | object ∞, image 60, t 20, σ 0 (f/0.79) |
| Acoustics-off meniscus error (front / back) | 0.48 / 13.0 µm | 1.10 / 43.8 µm |
| Source power; peak pressure | 0.58 W; 1.44 MPa | 0.55 W; 0.98 + 1.25 MPa (bands) |
| Fold margin along the start-up | 0.94 | 0.96 |
| Start-up end, pupil error | 0.14 / 0.21 nm | 0.12 / 0.32 nm |
| Formation (dt 25 and 12.5 ms) | — | 0.12 / 0.61 nm, both steps |
| Refinement, max change | — | 0.39 nm (P3: 1.4 nm) |
| Amplification; slowest decay (128×24) | — | 1.06; −84.8 1/s |
| Spot RMS / max (Airy radius) | — | 0.010 / 0.125 µm (0.57 µm) |
| Water sound speed +0.1 % | — | 147 / 78 nm |

Lens 7 is the first nonlinear confirmation of a two-band drive
(`acoustics/multiband.py`, `afl capability-synthesis`).

**Nonlinear check of the whole atlas.** `atlas_nonlinear.py` repeats, for every
lens that fits the cell, the convex program and one-tone-per-band refinement
under both caps, then follows the coupled start-up branch from the
acoustics-off meniscus (up to 90 µm off target) at the lens's fill volumes
(`atlas-nonlinear-2026-10-27/`, 5.4 CPU-hours):

| Start-up branch end, max pupil error (front or back) | Lenses (of 68) |
|---|---:|
| below 1 nm | 61 (all 10 two-band lenses among them; median of all 68: 0.16 nm) |
| 1 to 10 nm | 3 (lenses 3, 12, 35: 2.5, 1.4, 1.1 nm) |
| 10 to 20 nm | 3 (lenses 15, 71, 78: 10.2, 12.9, 17.1 nm) |
| fold before full amplitude | 1 (lens 57: fold at 100 % amplitude, 31 nm) |

So 64 of 68 reach the 10 nm goal with the atlas's command class. In the seven
short lenses the convex bound is 0.003–0.03 nm (lens 57: 0.47 nm) and the
shortfall is already in the one-tone refinement (frozen error within 30 % of
the nonlinear one), which sits at the 2 MPa cap in six of them and at the 1 W
cap in lens 12: a limit of the one-tone restriction under the caps, not of the
source. Their follow-up:

| Lens | One tone at 1.5 MHz | Two tones, 0.75 + 1.5 MHz | One tone, fold margin ≥ 0.5 |
|---|---:|---:|---:|
| 3 | 2.5 nm | 0.06 nm (0.94 W, margin 0.21) | 129 nm |
| 12 | 1.4 nm | 0.03 nm (0.69 W) | 0.23 nm |
| 15 | 10.2 nm | 0.03 nm (0.31 W) | 0.42 nm |
| 35 | 1.1 nm | 0.06 nm (0.57 W) | not run |
| 71 | 12.9 nm | 0.04 nm (0.42 W) | not run |
| 78 | 17.1 nm | 0.03 nm (0.53 W) | not run |
| 57 | fold, 31 nm | **start-up ends on another equilibrium: 25 / 159 µm** | not run |

The fold-margin column was stopped by a container restart after lenses 3, 12
and 15; given lens 3, it was not restarted.

The second band rescues six of the seven; the fold-margin inequality trades
accuracy for margin and is not the cheap remedy (lens 3 gets worse). Lens 57
(object −150, image 60, t 15, σ −0.5; acoustics-off error 2.5 / 76 µm) is a
different failure: the two-tone command holds the target in the frozen model
(0.33 nm, convex bound 0.02 nm), but the start-up branch from the meniscus
reaches full amplitude on another equilibrium, farther from the target than
the meniscus itself. Capability is not the issue; the path is. A finer or
different start-up path (ramping the bands separately, or the volumes) has
not been tried. A 40-step uniform ramp (instead of 8 steps) is running, to tell
a branch jump of the continuation from a genuine path failure
(`lens57-fine-ramp.jsonl`; the first attempt was stopped by a container
restart).

**Full checks of two rescued lenses** (`rescued-2026-10-27/`, configs
`*-atlas15-2band*`, `*-atlas71-2band*`): the same chain as lens 7.

| | Lens 15 | Lens 71 |
|---|---:|---:|
| Lens | object ∞, image 90, t 20, σ −0.5 | object −150, image 90, t 20, σ 0.5 |
| Refinement (mesh, order, quadrature, surface), max change | 1.0 nm | 0.37 nm |
| Static fold margin at the design | 0.86 | 0.996 |
| Frozen-to-coupled amplification | 1.19 | 1.01 |
| Slowest creeping-flow decay, 128×24 mesh | −77 1/s | −87 1/s |
| Formation, dt 25 and 12.5 ms | running | running |

On the coarse 64×12 fluid mesh both show two nearly neutral modes
(−0.001 1/s) that are absent on the 128×24 mesh; the S09 stability verdicts use
the fine mesh. The first formation runs were stopped by a container restart at
t = 0.83 s and 0.90 s and are kept as partial records (`INTERRUPTED.md`).

**Minimum hardware: channel count** (`channel_study.py`, frozen capability and
one-tone refinement, 1 W, 2 MPa; `channels-2026-10-27/`). The 112 channels are
a discretization, not a requirement:

| Channels (rings per window, wall bands per layer) | 112 (32, 16) | 56 (16, 8) | 42 (12, 6) | 28 (8, 4) | 21 (6, 3) | 14 (4, 2) |
|---|---:|---:|---:|---:|---:|---:|
| Convex bound, worst of 5 lenses | 0.010 nm | 0.010 nm | 0.010 nm | 0.011 nm | 0.011 nm | 0.025 nm |
| One tone per band: Fig. 9b | 0.020 | 0.005 | 0.012 | 0.025 | 0.008 | 0.020 nm |
| Fig. 7a (pressure-capped) | 0.12 | 0.57 | 11.2 | 2.8 | 11.2 | 12.3 nm |
| Lens 22 / lens 45 | 0.01 / 0.12 | 0.02 / 0.19 | 0.01 / 0.14 | 0.01 / 0.04 | 0.04 / 0.04 | 0.34 / 0.28 nm |
| Lens 7 (two bands) | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.007 nm |

With 14 channels (4 rings per window, 2 wall bands per layer) the bound is
still 0.025 nm for every lens tried, and four of the five lenses keep a
one-tone command below 0.4 nm. Only the pressure-limited Fig. 7a needs about
56 channels for a sub-nanometre single tone. These are first-order numbers; no
reduced layout has been through the nonlinear chain.

**Mesh defect and correction.** A first pass of this atlas used a wave mesh that,
for 20 of the 68 lens geometries, contained zero-height elements: cell and port
grids shared layer fractions computed one rounding unit apart, so exact
de-duplication kept both rows and the wave matrix was singular (5 % linear
residual with both direct solvers; force discontinuous in the shape). Those
rows made 13 lenses look hard and 3 unreachable, and supported a claim that a
second band beats doubling the modes (14 vs 5 rescued lenses); that claim is
withdrawn. The mesh now merges near-duplicate nodes (`merged_nodes`); meshes
without such nodes, including every saved S06–S09 design except the two-band
atlas-38 run, are unchanged. The invalid records are kept and marked
(`atlas-2026-10-26/MESH-DEFECT.md`).

## Verdict of this stage

Within the declared axisymmetric, quiescent, creeping-flow model, one fixed
machine (the water / NOA 61 / water cell, the same 112 source channels and 26
modes, caps 1 W and 2 MPa per band) holds every one of the 68 family singlets
that fit it: the convex bound is below 0.5 nm for all. Started from the
acoustics-off meniscus, 67 of the 68 form within 1 nm: 51 with one tone at
1.5 MHz and 16 with tones at 0.75 and 1.5 MHz (10 classified two-tone by the
atlas, 6 rescued by the second tone). Lens 57 is the one exception
and it is a path, not a capability, failure. The full chain (start-up,
fixed-command refinement, coupled stability, formation at two time steps) has
been run on three designs (Fig. 9b, Fig. 7a with a fold margin, lens 7) and all
three form within 1 nm. The Fig. 7a fold-margin design was synthesized with
10 W / 5 MPa limits and runs at 1.12 W and 2.10 MPa, slightly above the
machine's 1 W / 2 MPa caps. Sensitivities of tens to hundreds of nanometres to a
0.1 % sound-speed change mean a real machine needs measurement and trim. The
channel study suggests far fewer than 112 channels suffice (first-order only).

Open for the next stage, in order: the start-up path of lens 57; the full
chain for a sample of the rescued lenses and for reduced channel counts;
acoustic streaming, heating and non-axisymmetric stability; a measured
NOA 61 sound speed; and a physical transducer realization of the source modes.
Nothing here has been built or measured.

## Not assessed

Non-axisymmetric (m ≥ 1) disturbances, inertia, acoustic streaming, heating,
contact-line hysteresis, transducer realization, curing shrinkage and
demoulding. All results are numerical in the declared model; nothing has been
built or measured. The full nonlinear chain (start-up, refinement, coupled
stability, formation) has been run on Fig. 9b, Fig. 7a and lens 7; the
start-up branch alone on all 68 atlas lenses.

## Reproduce

Configs in `configs/S09-universal-source/`: `afl faithful-synthesis` on
`fig9b-water-common-1p5mhz-1w-2mpa.json`,
`fig7a-water-common-1p5mhz-10w-5mpa(-margin0p7).json`; then the
`force-convergence-*`, `coupled-stability-*` and `formation-*` configs.
Two-band: `afl capability-synthesis atlas38-water-common-2band-1w-2mpa.json`.
Scans: `tools/analysis/universal/` (`demand_check.py`, `capability_scan.py`,
`nonlinear_check.py`, `stability_capability.py`, `atlas.py`, `atlas_nonlinear.py`,
`channel_study.py`). Videos: `tools/presentation/render_showcase.py`.
