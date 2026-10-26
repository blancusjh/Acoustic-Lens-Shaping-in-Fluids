# S08 — Reproducing a published stigmatic singlet by acoustic fluid shaping

**Question.** Can the S07 faithful inverse shape the two NOA 61 interfaces of a
three-liquid cell into a *prescribed* published lens, with the apparatus
adapted to the lens rather than the lens to the apparatus? Only the shaped
liquid equilibrium is modelled; curing is a separate, unmodelled process.

**Lens.** Silva-Lora and Torres, JOSA A 37, 1155 (2020), stigmatic singlets
with n = 1 / 1.7 / 1. Fig. 9b: object at infinity, d1 = 137.95, d2 = 90,
t = 10 mm, aperture Ø 56 mm (T/D 0.18), scaled 1/14 to a 4 mm aperture:
t = 0.714 mm, image 6.43 mm beyond the front vertex, f = 6.16 mm, f/1.42 at the
image. Fig. 7a (finite object, T/D 0.36) was tried first and is reported below.

**Method.** `afl faithful-synthesis` with a `singlet` block: both faces are the
exact Cartesian surfaces (closed-form superconic, `PinnedCartesianFace`), both
contact lines are pinned on one ring at the heights of the lens surfaces there,
and the windows sit 0.8 mm beyond the vertices. Source: 26 continuous modes
(8 Bessel modes per window, 10 cosine modes on the wall), 1.5 MHz. The optics
of every state are traced with the published indices, i.e. as the cured lens
in air.

## Optics checks (independent of the fluid model)

`tools/analysis/s08/fig9_published.py`. The published edge thicknesses give
the published 28 mm aperture radius to 1e-13; a collimated trace through the
exact surfaces is stigmatic to 1e-14 mm; the Abbe sine ratio h / (n sin u) and
the paper's M agree to 2e-15. Aplanatism (M − 1)_RMS: Fig. 9a 1.9e-2,
Fig. 9b 1.2e-4, consistent with the paper's claim that 9b reduces coma.

> **Correction (S09, 2026-10-26).** The conclusion of the next section is
> wrong. The 674 nm floor was a local minimum of the inverse started from
> frozen least-squares fits of random seeds, not a reachability limit of the
> water cell. The convex capability program of theory section 10f finds, for
> the same lens, cell type, 26-mode source and frequency, a single tone of
> 0.17 W whose nonlinear start-up branch from the acoustics-off meniscus ends
> within 4e-11 m of both faces (studies/S09-universal-source). The multi-tone
> figures below came from the same kind of local fit and are equally
> unreliable. The tuned-liquid results remain valid as results, but liquid
> tuning is not necessary for Fig. 9b.

## Water / NOA 61 / water: the field cannot close the gap

| Quantity | Fig. 7a | Fig. 9b |
|---|---:|---:|
| Pinning radius, clear radius | 2.02, 2.0 mm | 2.02, 2.0 mm |
| Acoustics-off meniscus, best fill volumes (front / back) | 11.0 / 12.5 µm | 0.52 / 1.24 µm |
| 1 W / 2 MPa synthesis, pupil max error | 7.2 µm, stopped (`STOPPED.md`) | 676 nm, stopped at iteration 8 (`STOPPED.md`) |
| 10 W / 5 MPa synthesis | 14.0 µm, fold margin 0.03, stopped | 674 nm, fold margin 0.37 |

For 9b, ten times the power gives the same 674 nm: a reachability floor, not
a power limit. Diagnostics (`scans-2026-10-26/`):

- The mismatch is a smooth W profile (an r⁴ departure) needing only
  0.05–0.3 Pa of extra Laplace pressure (`fig9b_mismatch.py`).
- An uncapped frozen fit at 0.5–3 MHz never removes it (residual 0.76–1.5 µm,
  diverging power; `fig9b_frozen_power.py`).
- A multi-tone drive (rank-r time-averaged force, tones far apart in frequency
  so cross terms average out) reaches 108–120 nm for r ≥ 2 against 616 nm for
  one tone, but at about 1.5e5 W and over 100 MPa: unphysical
  (`fig9b_multitone.py`).
- A larger lens is worse: gravity bends the front face further from the
  Cartesian shape (`fig9b_scale.py`).

## Tuned liquids: the apparatus adapted to the lens

With the acoustics off the two interfaces do not interact, and each face's
capillary-gravity shape depends only on its own density step and interfacial
tension, both set by the liquid on its outer side. Scanning them
(`fig9b_outer_density.py`, `fig9b_liquids.py`):

| Face | Outer liquid | Density | Tension vs NOA 61 | Meniscus error |
|---|---|---:|---:|---:|
| front | water | 997 kg/m³ | 25 mN/m (assumed) | 775 nm |
| front | brine (about 4.5 % NaCl) | 1030 kg/m³ | 25 mN/m (assumed) | 84 nm |
| back | water | 997 kg/m³ | 25 mN/m (assumed) | 991 nm |
| back | light oil | 850 kg/m³ | 5 mN/m (required) | 55 nm |

The liquid properties are declared assumptions
(`configs/shared/noa61-brine-oil-assumption.json`); no oil has yet been
identified that is immiscible with uncured NOA 61 at 5 mN/m.

## Results, tuned liquids at 1 W / 2 MPa (`fig9b-tuned-2026-10-26/`)

| Quantity | Value |
|---|---:|
| Acoustics-off meniscus, pupil max height error (front / back) | 84 / 55 nm |
| Design, pupil max height error | 27.0 / 1.3 nm |
| Source power, sampled peak pressure | 0.75 W, 0.48 MPa |
| Fold margin along the start-up branch (minimum) | 0.91 |
| ‖A‖/‖K‖, frozen-to-coupled amplification | 5.5e-4, 1.12 |
| Slowest creeping-flow decay, 128×24 fluid mesh | −18.3 1/s |
| Spot at the image, RMS / max (401 rays, all transmitted) | 0.21 / 2.41 µm (Airy radius 1.02 µm at 587.6 nm) |
| Shape wavefront error, focus removed (PV) | 8.8 nm = λ/67 (meniscus: λ/5) |
| Aplanatism (M − 1)_RMS of the formed lens | 1.9e-4 (exact lens 1.2e-4) |

The coarse 64×12 fluid mesh also reports two decays near −1e-3 1/s; they live
in the 20 µm annulus between the clear aperture and the pinned rim, narrower
than one coarse cell, and vanish on the 128×24 mesh (−18.30 there, −18.29 for
the coarse mesh's next mode). The design error stalled at about 27 nm on the
front face from the first iterate: below the meniscus, above the 10 nm goal.

**Fixed-command refinement** (`force-convergence/`): wave mesh 24×9 to 72×27,
P3 to P5, quadrature +2 and 36 to 72 surface elements change the predicted
equilibrium by at most 1.4 nm.

**Physical sensitivity** (same command, coupled first-order shift, front / back):

| Perturbation | Shift (nm) |
|---|---:|
| Brine and oil sound speed +0.1 % | 21 / 84 |
| Resin sound speed +1 % | 304 / 1765 |
| Frequency +100 ppm | 1.1 / 12 |
| Interfacial tensions +1 % | 37 / 12 |

These are 5–35 times larger than in S07. The 5 mN/m back face that the field
can shape is equally soft to any error in the field; the resin sound speed is
a placeholder. This design can only be held under closed-loop trim against a
measured wavefront, not open loop.

**Formation** (`formation-dt0025/`): forward start-up from the acoustics-off
meniscus under the held command, linearized acoustic load, 1 s ramp, 2 s.
It reaches the design: 27.0 / 1.3 nm, spot RMS 0.209 µm, all rays transmitted.
Halving the step (`formation-dt00125/`) changes the transient by at most
1.0 nm and the final state by less than 1 pm.

## Verdict

(See the correction above: the claim that water cannot reach Fig. 9b is
withdrawn.) In the declared axisymmetric, lossless-acoustics, creeping-flow model, a
scaled copy of the published Fig. 9b singlet is formed from the acoustics-off
meniscus by a 0.75 W field. The state is converged in the discretization,
stable, and near diffraction-limited when traced as the cured lens. Getting
there required choosing the liquids for the lens: with water on both sides no
field of reasonable power reaches it. The front face misses the 10 nm goal
(27 nm), the liquid set is an assumption, and the sensitivities demand closed
loop. Not experimental validation; non-axisymmetric modes, inertia, streaming,
heating, curing shrinkage and demoulding are not assessed. Fig. 7a remains out
of reach in the water cell.

## Aplanatic NOA 61 / water singlets (side search)

`aplanatic_search_free.py`, `aplanatic_search_fixed_m.py`, results in
`scans-2026-10-26/`. Without a magnification constraint the search converges to
the symmetric 1:1 relay (m = −1, collimated inside the lens), aplanatic by
symmetry to 1e-10. With the magnification held away from −1:

| Object | m | NA | (M − 1)_RMS | Centre thickness |
|---|---:|---:|---:|---:|
| 15 mm | −0.53 | 0.15 | 6.9e-4 | 2.6 mm |
| 10 mm | −0.59 | 0.23 | 2.8e-3 | 6.3 mm |
| 15 mm | −2.0 | 0.14 | 1.5e-2 | 1.3 mm |
| 15 mm | −3.0 | 0.16 | 2.5e-2 | 1.4 mm |

With the low index contrast of NOA 61 in water, only m ≈ −0.5 approaches
aplanatism.

## Apparatus

Declared in `configs/S08-published-singlet/apparatus-proposal-fig9b-tuned.json`,
rendered by `tools/presentation/render_apparatus.py`
(`fig9b-tuned-2026-10-26/figures/s08-fig9b-apparatus.png`), with modelled and
proposed parts separated. Figures and videos of the design and formation:
`render_cell_figures.py --prefix s08-fig9b`.

## Reproduce

Configs in `configs/S08-published-singlet/`: `fig9b-tuned-liquids-1p5mhz-caps-1w-2mpa.json`
(`afl faithful-synthesis`), then `force-convergence-fig9b-tuned.json`,
`coupled-stability-fig9b-tuned.json` and `formation-fig9b-tuned-dt0025.json`
on its `design/`. The water-cell runs use `fig9b-1p5mhz-*` and `fig7a-1p5mhz-*`.
Scans: `tools/analysis/s08/`.
