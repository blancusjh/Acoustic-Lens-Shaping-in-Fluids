# Two-face Cartesian shaping: numerical evidence

22 September 2026. **The requested 10 nm maximum error has not been achieved.**
The new implementation solves two coupled stationary mean interfaces in one
acoustic chamber. It does not yet simulate their formation or establish stability.

## Apparatus and accuracy contract

The exploratory apparatus is a sealed cylinder of radius 4 mm, with a 2 mm
clear radius on each interface. Three hypothetical liquids have densities
1200/1100/1000 kg m−3, sound speeds 1500/1200/1500 m s−1, and interface tensions
0.025 N m−1. Nominal interfaces lie at −1 and +1 mm. Both independent
interface-volume integrals are fixed, preserving all three phase volumes.
Sources share one coherent harmonic transmission field; end and side ports
have passive impedance. A command is the source component of normal velocity,
not total port velocity or electrical voltage.

Each Cartesian patch has its own `(n_o, z_o, n_i, z_i)` and prescribed absolute
vertex. Optical indices parameterize target geometry; these two individually
specified patches are not claimed to compose a stigmatic imaging singlet.
The annulus outside each pupil joins the pinned rim and satisfies volume.
Errors are absolute laboratory-coordinate heights, without piston subtraction.
The maximum search uses 2001 independent radii and local extremum refinement;
it is not a rigorous interval certificate.

## Computed evidence

Values below are maximum mean-height errors in nm on each clear aperture.
Each refined row uses exactly the design-grid physical commands, without
re-optimization. The configuration and executed-source snapshot accompany each
[campaign](../artifacts/dual-cartesian-2026-09-22/).

| Experiment, pair A | Design front / back | Refined front / back | Refined carrier front / back (nm) |
|---|---:|---:|---:|
| 56 ports, 1 MHz screening | 20660 / 10904 | 20155 / 10453 | 292 / 156 |
| 56 ports, joint annulus, 1 MHz | 1517 / 597 | 1779 / 606 | 297 / 139 |
| 56 ports, 1.8 MHz | 474 / 200 | 462 / 205 | 252 / 76 |
| 112 ports, 1.8 MHz | 292 / 97 | 284 / 97 | 255 / 62 |
| 56 closer ports, 1 MHz | 3073 / 13628 | 3064 / 13613 | 89 / 124 |

The closer-port apparatus reduces the outer-layer depths from 2 mm to 0.5 mm.
A second independently specified target pair in the same closer-port apparatus
gives 489 / 578 nm on the design grid and 492 / 577 nm after refinement, with
82 / 77 nm carrier displacement. The 112-port run subdivides the 56 physical
supports. Repeating the saved command exactly represents the original source
function; the optimizer clips its warm start to 95% of the component bounds
before selecting a starting candidate and re-optimizing the richer source
space. This is distinct from field-grid convergence, and is not itself a
controlled proof of convergence in the continuous source space.

All listed inverse optimizations exhausted their configured evaluation budgets.
Forward equilibria are independently solved with fresh wave fields: neither
target projection nor the frozen-compliance design residual is an achieved
surface. In particular, the annulus run's roughly 1 µm design residual must not
be reported as a sub-10 nm result. No global source optimum or unreachability
bound has been established. The frequency experiment also changed discretization
and normal-flux recovery; it is not an isolated causal measurement of frequency.
Two grids expose sensitivity but do not establish a nanometre error certificate.

The best tested pair-A candidate is the 112-port run: its refined absolute
mean errors are 283.67 nm and 96.54 nm, with approximately 32.085 W acoustic
source work and 9.47 MPa sampled peak pressure. Its refined field has 222145
pressure degrees of freedom. These ideal source-port quantities are not
electrical power requirements or validated operating pressures.

Even where mean errors decrease, modeled carrier displacements remain tens to
hundreds of nanometres. In the single-frequency linear interface model, the
phase-maximum height error at a point is the absolute mean error plus the
carrier height amplitude. Thus the sampled carrier alone already rejects an
unqualified instantaneous 10 nm claim for these candidates. High modeled
pressures require separate checks of material response and model validity.

### Conditional pressure-node constraint

For a pure pressure node on the interface, the theory gives the necessary
condition `f >= sqrt(osc(H)/abs(Δρ))/(π ε_a)`, where
`H = σκ + Δρ g h` and `ε_a` bounds normal carrier amplitude. Evaluating the
exact pair-A Cartesian height and its first two derivatives at 4001 pupil
radii gives sampled traction oscillations 0.60837 Pa (front) and 0.47579 Pa
(back). With `Δρ = 100 kg m−3` and the entire `ε_a = 1e-8 m` allowance assigned
to carrier motion, these give necessary frequency lower bounds of approximately
2.483 MHz and 2.196 MHz. Curvature uses
`κ = −h''/(1+h'^2)^(3/2) − h'/(r sqrt(1+h'^2))`, with the axis limit understood.
The input is pair A in `configs/dual/cartesian-pair-frequency.json`; no source
optimization enters this calculation.

The sampled oscillation is a lower bound, so it can supply a necessary bound
but not a sufficient design frequency. Annulus loads, nonzero mean error and
the stricter graph-height criterion can increase requirements. The current
campaigns are not restricted to pure pressure nodes, so these numbers do not
prove that their lower frequencies are universally infeasible. They explain
why the local pressure-node existence construction alone cannot establish the
requested instantaneous precision.

## Verification and provenance qualifications

- Independent six-amplitude traveling-wave matching tests three-layer pressure,
  interface velocity, and absolute source work. Pressure and weak-flux velocity
  converge under fixed-command refinement, rather than only satisfying a
  discrete energy identity.
- A volume-preserving Bessel capillary mode checks mechanical compliance.
  Equal acoustic materials give zero radiation contrast. Cartesian projection
  convergence is checked separately from physical attainability.
- The original screening and interrupted authority campaigns omitted the
  common azimuthal factor `2π` in **reported watts only**. Multiply their
  `source_power_w` and `boundary_absorption_w` by `2π`. Wave fields, forces,
  commands and surface errors are unchanged. Their original evidence is retained
  with an explicit correction sidecar. Later campaigns use the corrected units.
- `authority-56` stopped after its design-grid state: it is incomplete and is
  excluded from the comparison of verified refinements. Its higher drive limit
  is not evidence of improved hardware feasibility.
- Earlier manuscript-review images predate version 3 and were moved intact to
  the separate research archive. Consult the current build manifest and
  `artifacts/theory/review-v3p3/` for the current PDF review.

The imported discussion's saved **one-interface** holding trajectory has a
maximum 6.52 nm mean error when the switching frame is included, and 6.518 nm
strictly after the switch; its final error is 5.58 nm. The imported report's
6.14 nm holding maximum is not reproduced by this saved-frame audit; the
unavailable driver prevents resolving its precise holding-window convention.
A fresh final acoustic solve gives about 40.38 nm carrier displacement. This
[audit](../artifacts/imported-discussion-audit-2026-09-22-v2/report.md) neither
reproduces the missing time-integration driver nor validates two-face operation.

## Next scientific gates

1. Separate inverse optimization failure from source-space limitation using
   converged optimization, adjoint checks, and independently verified lower
   bounds or source-space enrichment. Impose actual power and carrier limits
   during design, rather than merely rejecting candidates afterward.
2. At fixed commands, refine both acoustic and surface spaces on at least three
   resolutions and budget numerical error well below 10 nm. Extend extremum
   verification beyond sampled carrier values.
3. Implement the coupled dynamic and linearized operators from the general
   theory, then test formation and full three-dimensional perturbations about
   the appropriate streaming base flow. A stationary Newton seed is not a
   formation trajectory.
4. Introduce measured materials, attenuation, wall layers, thermal feedback,
   source calibration and uncertainty before proposing physical 10 nm accuracy.
   Driven liquid optics, transient formation and curing need separate criteria.

The theory motivates these gates; it does not substitute for their execution.
