# Required loading before emitter design

The first gate is now an emitter-independent mechanical problem. The canonical
derivation is manuscript version 3.8, section “Determine the load before
synthesizing emitters”. It distinguishes a fixed spatial mean traction from a
fixed acoustic command and treats the entire two-interface domain.

## Executed static check

[Report and load plots](../artifacts/ideal-load-2026-09-23/stigmatic-c2/report.md)
and [SI data](../artifacts/ideal-load-2026-09-23/stigmatic-c2/required-load.npz).
Reproduce with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m acoustic_freeform.dual.ideal_load_audit configs/dual/ideal-load-stigmatic.json --out <new-directory>
```

The existing shared-conjugate Cartesian pupils and C2 sealed-volume annuli are
unchanged. Continuous target curvature and gravity generate the prescribed
loads; the discrete mechanical gradient does not generate its own right-hand
side. A nonlinear equilibrium solve starts from flat, on five mechanical grids,
with independent 16/24-point load integration. These are root-solver iterations,
not a formation trajectory. The continuous loads stay fixed under refinement.

The upward normal load on the lower face ranges from approximately -8.85 to
34.09 Pa, and on the upper face from -10.11 to 22.88 Pa, after removing each
horizontal-disk mean. These are sampled ranges of **mean stress differences**,
not acoustic carrier amplitudes, absolute fluid pressures or cavitation bounds.
Both profiles extend to the 4 mm wall radius, beyond the 2 mm optical pupil.

Even the 20-element equilibrium has sampled pupil errors 0.175/0.107 nm and a
maximum geometric spot radius 0.0283 micrometres; all 4001 sampled rays transmit.
At 104 elements these decrease to approximately 9.93e-6/6.56e-6 nm and
1.93e-5 micrometres. Full-surface errors also decrease from 1.41/0.962 nm to
8.77e-5/5.84e-5 nm. Quadrature refinement changes neither conclusion. These tiny
numbers are numerical consistency/convergence results for ideal prescribed
loading, not achievable manufacturing or acoustic tolerances. The maximum
search is sampled, not a continuous interval certificate.

An independent spherical-cap test checks the Laplace pressure including the
axis; existing independent volume-energy tests check gravity. Four focused
tests passed. The run captures configuration, resolved parameters and source
provenance alongside its output.

## Stability and formation status

For pinned graphs, positive surface tensions and the current heavy-below
density ordering, the loaded mechanical potential is strictly convex for
arbitrary horizontal perturbations, including non-axisymmetric ones. The theory
derives a viscous energy-dissipation identity for an **ideal maintained spatial
load**, with no streaming or thermal feedback. Under the stated regularity and
coercivity assumptions it gives energy stability, not a maximum-error guarantee
or an unconditional global-settling theorem. Inviscid dynamics need not settle.

## Variational viscous extension and forward formation

Version 3.9 derives a three-fluid Rayleighian with incompressibility and graph
velocity constraints. The pressure load is their normal-velocity multiplier.
In creeping flow the required load is the capillary--gravity variation plus
the full coupled viscous resistance applied to the desired interface velocity.
For a quiescent target the latter vanishes: the previous holding map is still
correct. With inertia, the weak inverse additionally requires the fluid
acceleration; a factorization condition identifies motions that cannot be
produced by scalar normal traction alone. Streaming and thermal endpoint
stresses cannot be silently absorbed into the old static formula.

The new `dual.viscous` implementation solves axisymmetric three-fluid Stokes
flow on the changing faceted geometry, with no-slip walls, continuous velocity,
separate phase pressure spaces allowing interface jumps, and the full hoop
strain. Both surface velocities come from the same fluid solve. Load transfer
and projected kinematics are reciprocal; no fitted damping time is used.
An independent polynomial recirculation solution checks PDE convergence, and
tests check viscosity scaling, cross-interface coupling and work/dissipation.

Executed formation runs use an explicitly hypothetical **5 Pa s in every phase**.
This is a declared viscous scenario, not a qualified liquid triplet. A fixed
full-target load is switched on at t=0, starting from flat surfaces. No target
interpolation prescribes intermediate shapes. The trajectory force file uses
the reference-plane pressure gauge; the static load plot removes disk means.
The difference is constant on each sealed interface and does no shape work.

[Three-run independent audit](../artifacts/ideal-load-2026-09-23/viscous-three-run-audit/report.md)
compares 0.25/0.125 s steps and 32-by-12/48-by-18 per-layer fluid grids at fixed
20-element surface space and fixed physical loads. At 20 s, the refined fluid
run has sampled pupil errors approximately **0.175/0.108 nm**, maximum geometric
spot **0.0285 micrometres**, and all 4001 final rays transmitted. Loaded mechanical
energy decreases in each saved trajectory. At some intermediate times one rim
ray is lost; these frames are not accepted just because their surviving spot
radius is small. Endpoint agreement does not certify the whole time history:
halving the step changes early heights by up to 6.13/4.44 micrometres, whereas
the fluid-grid change reaches 28.3/16.9 nm. Final differences are much smaller.

The first three runs completed and saved all time steps and optical summaries
but failed to serialize a NumPy integer in the final fluid-diagnostics export.
Their original states and validation files are unchanged. Independent energy
checks and explicit `execution-status.json` files record this; missing per-step
fluid diagnostics are not invented. The serialization bug is fixed in current
source. A separate finer-time run uses the correction.

That fourth run (`viscous-time-finest`, 0.0625 s steps on the 32-by-12 fluid
grid) finished with all diagnostics exported. Its final pupil errors are
0.1749/0.1073 nm and its 4001-ray maximum spot is 0.02844 micrometres, with
all rays transmitted. The 0.125-to-0.0625 s common-time height difference is
3.78/2.73 micrometres at its maximum, falling to about 0.003 nm at the endpoint.
This supports the endpoint conclusion, not a time-history convergence claim
at 10 nm. See the [four-run audit](../artifacts/ideal-load-2026-09-23/viscous-four-run-audit/report.md).

[Executed notebook 08](../artifacts/notebooks/08_prescribed_load_formation.executed.ipynb)
embeds a physical-time animation of the saved refined-fluid trajectory: both
surfaces, a 3D cylinder, pressure maps, sampled error curves and evolving spots.
The setup has no emitters because this is the prescribed-load subproblem.
The offline browser check passed: frames at 0, 10 and 20 s differ, and play
and pause work without external network access. The executed notebook embeds
the animation rather than linking to an external video.

This is not the complete Navier--Stokes/acoustic/thermal apparatus. Inertial
startup and non-axisymmetric flow are not simulated. The radius-based viscous
diffusion time is 0.00384 s for these hypothetical properties; sampled Reynolds
estimates are small, but neither diagnostic bounds a 10 nm modeling error.
The next physical gates are material qualification, inertial/3D checks and
acoustic-induced streaming and heating. Matching the ideal holding map alone
does not establish that an array supplies a stable equivalent actuation.

The earlier acoustic failures and notebooks remain preserved and are not
relabeled as passing results.
