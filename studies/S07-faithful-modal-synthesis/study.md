# S07 — Faithful two-face shaping from the acoustics-off meniscus

**Question.** Can a low-order continuous source correct the acoustics-off
meniscus of both interfaces to a shared-conjugate target, such that the state
reached by the physical start-up is the target, survives fixed-command
refinement, and is robust to realistic parameter tolerances?

**Method.** Theory section `docs/theory/sections/10e_faithful_inverse.tex`.
The start is the capillary-gravity equilibrium at declared fill volumes, not
the flat equal-volume state. The source is 20 low-order modes (6 Bessel modes
per window, 8 cosine modes on the wall) sampled on ideal regions. The inverse
fits the end of the quasi-static start-up branch, with the coupled
(K − A)⁻¹ Jacobian, first-order tolerance shifts and hard caps of 1 W source
power and 2 MPa sampled peak pressure (`afl faithful-synthesis`).

**Prescription.** Same materials (NOA 61 between water layers in the 8 mm
cylinder) and conjugates as S04. Only the free vertex positions of the
shared-conjugate pair changed, to −0.855 mm (front) and +0.059 mm (back), so
each Cartesian pupil lies near a meniscus pinned at the wall. With fill
volumes optimized jointly with the vertices, the acoustics-off meniscus alone
reaches the Cartesian pupils within 289 / 40 nm (front / back) with gravity,
and 378 / 0.7 nm at zero gravity; the old vertices (−150 / +120 µm) cannot
come closer than 105 µm. Gravity helps the front face and hurts the back.

## Results at 0.75 MHz (`meniscus-2026-10-25/`)

| Quantity | Value |
|---|---:|
| Acoustics-off meniscus, pupil max height error | 382 / 189 nm |
| Design: reachable pupil max height error | 19.6 / 11.3 nm |
| Source power, sampled peak pressure | 0.55 W, 0.37 MPa |
| Fold margin along the start-up branch (minimum) | 0.99 |
| ‖A‖/‖K‖, frozen-to-coupled amplification | 3.3e-4, 1.01 |
| Slowest creeping-flow decay (64×12 and 128×24 fluid meshes) | −41.3 1/s |
| Maximum geometric spot (401 rays, all transmitted) | 46.0 µm meniscus, 7.2 µm design |

The built-in fill-volume fit of this run was looser than the joint fit above
(382 / 189 nm rather than 289 / 40 nm); the acoustics corrected the difference.
The best pupil error, 6.2 nm, occurred at the frozen start; later iterations
traded height for tolerance robustness.

**Fixed-command refinement** (`force-convergence-750khz/`): wave mesh 24×9 to
72×27, P3 to P5, quadrature +2 and 36 to 72 surface elements change the
predicted equilibrium by at most 0.09 nm. The design is converged in the
declared model, unlike the S04-type state of S06 (69 nm).

**Physical sensitivity** (same command, coupled first-order shift, front / back):

| Perturbation | Shift (nm) |
|---|---:|
| Water sound speed +0.1 % (about 0.6 K) | 13.0 / 8.4 |
| Resin sound speed +1 % | 100.6 / 50.0 |
| Frequency +100 ppm | 0.6 / 1.0 |
| Interfacial tension +1 % | 301.5 / 14.6 |

Temperature sensitivity is 100–200× lower than in S06. The tension shift is a
property of the gravity meniscus, not of the acoustics: the command cannot
change it, and at zero gravity a fixed-volume cap does not depend on tension.

**Formation** (`formation-750khz-dt005/`, `-dt0025/`): forward start-up from the
acoustics-off meniscus under the held command, linearized acoustic load, 1 s
ramp. Both step sizes reach the design state (19.6 / 11.3 nm, 7.21 µm spot,
all rays transmitted).

## Verdict

In the declared axisymmetric, lossless-acoustics, creeping-flow model, the
start-up from the acoustics-off meniscus forms the designed state, the state
is converged and stable, and the field stays inside the 1 W / 2 MPa caps. It
does not yet meet the optical gates: height errors are 19.6 / 11.3 nm against
10 nm, the spot is 7.2 µm against 1 µm because the objective constrains
heights but not slopes (front slope error up to 444 µrad), and a 1 %
uncertainty in interfacial tension moves the front face by 0.3 µm. Next:
add the fixed-detector spot sensitivity to the objective, treat interfacial
tension as a calibrated constant or remove gravity by density matching, and
test lower frequencies. Not experimental validation; non-axisymmetric modes,
inertia, streaming and heating are not assessed.

## Reproduce

Configs in `configs/S07-faithful-modal-synthesis/`:
`meniscus-750khz.json` (`afl faithful-synthesis`), then
`force-convergence-meniscus-750khz.json`, `coupled-stability-meniscus-750khz.json`
and `formation-meniscus-750khz-dt{005,0025}.json` on its `design/`. The
`probe-*` configs belong to a flat-start probe stopped before any run finished
(`probe-2026-10-25/STOPPED.md`).
