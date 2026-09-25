# Precision campaign: current evidence

23 September 2026. The same two Cartesian patches and 2 mm clear radii remain
fixed. The stationary design-grid mean pass fails fixed-command surface
refinement; no mesh-converged or physical 10 nm result is established. Materials and source ports remain
hypothetical; the model is stationary, axisymmetric and isothermal.

The user has now selected **cycle-mean driven-liquid accuracy**. The acceptance
metric is the maximum mean-height error on each aperture, with no carrier
amplitude added. Historical combined-error failures below concern the earlier
strict instantaneous specification. They do not reject mean accuracy; conversely,
their small frozen fits do not establish a coupled mean-surface pass.

## Results that must not be conflated

| Candidate | Frozen compliance front/back (nm) | Carrier front/back (nm) | Independent coupled mean front/back (nm) |
|---|---:|---:|---:|
| 112 channels, 1.8 MHz, no carrier penalty | 70 / 64 | 264 / 62 | Not run for this new iterate |
| 112 channels, 1.8 MHz, whitened strong carrier penalty | 164 / 361 | 42 / 39 | Not run |
| 224 channels, 3.6 MHz, carrier weight 1 | 13 / 18 | 24 / 38 | 383 / 135 on design grid; refinement interrupted |
| 224 channels, 3.6 MHz, low-carrier subspace | 3696 / 3464 | 0.051 / 0.018 | Not run |
| 448 channels, 7.2 MHz, carrier weight 1 | 1.84 / 2.05 | 9.93 / 11.73 | Interrupted for memory pressure; unresolved |
| 448 channels, 7.2 MHz, peak-norm objective | 1.56 / 1.57 | 9.93 / 11.73 | Not run |
| 448 channels, 7.2 MHz, low-carrier subspace | 1731 / 1917 | 0.028 / 0.012 | Not run |
| 448 channels, 7.2 MHz, dense mean-only inverse | 0.000019 / 0.000025 (polynomial audit) | 10.33 / 12.01 | **0.00138 / 0.00140 at 52 elements; 10440 / 4812 at 104 elements** |

The low-carrier restriction suppresses modeled vibration but has not reproduced
the required traction. Solver termination flags and budgets are recorded;
these candidates are not certified optima. The 3.6 MHz forward solve gives
approximately 49.48 W acoustic source work and 18.51 MPa sampled peak pressure
on its design grid. Those are not electrical calibration or validation of
linear material response.

The minimax source-step experiments reached a conservative frozen combined
metric above 45 nm and were checkpointed before their full iteration budgets.
They are not proofs of infeasibility. Their actual nonlinear frozen merit was
checked before accepting each step. The semidefinite diagnostic failed its
numerical feasibility checks: a solution with negative covariance eigenvalues
is neither a realizable waveform nor a valid lower-bound certificate.

## Numerical and computational diagnostics

The completed `verify-mean-448-7p2mhz-retry1/` run freshly assembled the shared
wave equation and evaluated nonlinear force balance at the initial seed. The
residual was 3.3504e-14 m in the declared compliance diagnostic, satisfying the
existing 1e-11 m root tolerance without a Newton step. An additional field solve
confirmed diagnostics. This is a force-balanced stationary point of that finite
model, not a simulated formation trajectory. The initial-seed acceptance is
explicit in the record. Source work is 33.9035 W, pressure peak 28.1734 MPa,
and peak source component 0.97594 m/s. These are ideal-port model quantities,
not validated material operating conditions or electrical requirements.

At unchanged commands, increasing the surface basis from 52 to 80 elements
exposes an initial global compliance residual around 1.4e-6 m. The completed
equilibria have front/back aperture errors 9941.4411 / 4814.3342 nm at 80
elements and 10439.9047 / 4812.1375 nm at 104 elements. Root compliance
residuals are respectively 2.1657e-14 and 3.1597e-13 m; small equilibrium
residuals do not imply small target errors. Acoustic resolution is unchanged.
The design-grid pass fails this refinement test. Neither the refined shapes
nor a continuum error limit have been established as converged.
Evidence: `verify-mean-448-surface-refinement/` (completed, 150 and 235 wave
solves). This is not proof of physical unreachability or unique equilibria.
Before another source-count increase, investigate load integration, surface
resolution and acoustic resolution independently at fixed physical commands.

The peak-norm run exhausted 2000 evaluations and retained a large first-order
residual. Its sum of separate sampled mean and carrier maxima is
11.489 / 13.302 nm, so it fails even the frozen combined screen. The next
sequential-convex run enforces source disks directly; neither run changes
the target or constitutes a physical formation process.

The new convex run was interrupted before its first completed step, together
with the 448-source coupled check, after concurrent jobs caused over 10 GB
swap usage on this 16 GB machine. Their partial evidence is retained and is
not an infeasibility result. Large jobs must now run sequentially; the older
224-source refinement was subsequently stopped after about 61 elapsed minutes
without a fine-grid equilibrium. Its completed design-grid state and original
reports remain unchanged, with an explicit interruption clarification. The
448-source check is next, using verified exact interior elimination; this is
a prioritization decision, not a convergence or impossibility claim.

Exact condensed trial residuals agreed with the earlier full-factorization
trial to approximately 9e-19 m (a residual comparison, not a surface error).
Repeated factorization was still expensive. The active retry,
`verify-reused-448-7p2mhz/`, reused only the first factorization as a checked
preconditioner for freshly assembled wave equations. It has not yet produced
an accepted equilibrium and was stopped after the improved mean-only candidate
became available. The mean candidate's completed retry is described above.
Prior attempts, including a metadata-write failure,
each have explicit reports and retained provenance.

An independent planar audit at 7.2 MHz reports pressure relative errors
0.0311676, 0.00181462 and 0.0000797442 at 32, 64 and 128 cells per layer.
The corresponding back-interface velocity errors are 0.0222242, 0.0014230
and 0.0000656282. This resolves the high-frequency planar test more tightly
than the original 8/16/32-cell audit. It does not bound the curved candidate's
discretization error. Evidence: `analytic-7p2mhz/` in the September 23 outputs.

## Observation audit and revised inverse

The independent audit `audit-polynomial-448-7p2mhz/` found annular maxima
101.488 / 25.561 nm for the historical weight-one candidate, versus legacy
full-domain sampled maxima 41.643 / 23.560 nm. Its polynomial pupil maxima
are 1.842 / 2.076 nm. The subsequent dense mean-only inverse uses spline
quadrature observations with equal pupil/annulus weights and exact source-disk
constraints. Its command peak is 0.97594 m/s, and its audited frozen annular
residuals are below 0.000034 nm. The first-order stationarity diagnostic did
not pass; excellent traction matching is not a certified optimum or a coupled
surface result. Fixed-command verification remains mandatory.

## Source geometry

Counts refer to independent **axisymmetric source regions**: full-azimuth annuli
on both end planes and full-azimuth bands on the cylinder sidewall. They do not
claim a tested array of that many point transducers. Every region uses
`V.n = Y P + g`; the command is the source component `g`, not total wall velocity.
Region geometry and SI command exports are implemented in
`acoustic_freeform.dual.sources`. Calibrated hardware remains a separate problem.

## Reproduce and inspect

- Inputs: `configs/dual/precision/`.
- September 22 screening: `artifacts/dual-precision-2026-09-22/`.
- Current runs: `artifacts/dual-precision-2026-09-23/`.
- Protocol: [precision program](precision-program.md).
- Earlier independently refined results: [initial findings](dual-cartesian-results.md).

Completed outputs include configuration, source snapshots and reports. Cached
operators have acquisition hashes and retain their original execution provenance.
Partial experiments remain explicitly labelled; no failed candidate is silently
overwritten or promoted to a completed result.
