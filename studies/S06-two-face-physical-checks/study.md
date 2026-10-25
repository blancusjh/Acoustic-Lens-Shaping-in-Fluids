# S06 — Physical checks of a two-interface emitter state

**Question.** Is a converged two-interface held-command state, of the kind
recorded in S04, a physically meaningful result? Four checks: does the
equilibrium survive fixed-command refinement and realistic parameter
uncertainty; is it linearly stable; can it be reached from the unforced state
with an integrator that is itself stable; and is the optical coverage gate
measuring optics or rounding.

**State examined.** The private S04 states are not in this checkout. The
checks therefore use a surrogate rebuilt here from the historical S04 input
`configs/S04-emitter-synthesis/noa61-emitter-224-3p6mhz.json`. The surrogate has
224 ideal axisymmetric source regions, 3.6 MHz, NOA 61 between water layers in
the 8 mm sealed cylinder, and 36 surface elements. Its frozen fit is exact
(103 complex source coordinates for 78 force equations). The held-command
root reproduces the target to 10 pm with a 6.3 nm spot, at 17.3 MPa sampled
peak pressure and 29.3 W ideal source work, the same regime as S04 (18.5 MPa).
Every number below is numerical evidence for this surrogate in the declared
model only: axisymmetric, lossless longitudinal acoustics with passive ports,
creeping three-fluid flow, no streaming or heating. Nothing here is
experimental. The same commands should be rerun on the saved S04 states.

## Results

### 1. Fixed-command refinement (`afl force-convergence`)

Each trial keeps the saved geometry and command and changes one
discretization. The residual is mapped to the first-order coupled equilibrium
shift (K − A)⁻¹ r, with A the analytic acoustic shape stiffness. Pupil maxima,
front / back:

| Change from the design grid (96×48 cells, P4, 36 surface elements) | Coupled shift (nm) |
|---|---:|
| Wave mesh 64×32 | 2.1 / 17.1 |
| Wave mesh 144×72 | 0.8 / 7.7 |
| Element order P3 | 5.8 / 16.5 |
| Element order P5 | 0.7 / 5.4 |
| Quadrature +2 | 0.000 / 0.001 |
| Surface space 52 elements | 68.7 / 44.4 |
| Surface space 72 elements | 69.5 / 45.2 |
| Surface space 104 elements | 69.4 / 45.2 |

The surface space converges by 52 elements (52 → 72 changes 2.7 / 1.3 nm). The
design grid's own surface error is therefore about 69 / 45 nm, and the
refined, held-command equilibrium is predicted about 69 / 45 nm from the
target: the design-grid pass was an artifact of fitting on the grid used to
verify. The back face converges slowly in the wave mesh: the mesh-change
ratio 2.2 at refinement 1.5 is an observed order near 2 for P4 elements, which
suggests a singular corner where the pinned interface meets the wall. The
design grid's wave error on the back face is of order 10–15 nm. Either error
alone exceeds the 10 nm goal.

### 2. Physical sensitivity (same command, one parameter changed)

| Perturbation | Coupled shift (nm) |
|---|---:|
| Water sound speed +0.1 % (about 0.6 K of temperature) | 1355 / 2858 |
| Resin sound speed +0.1 % | 3327 / 2424 |
| Resin sound speed +1 % | 25796 / 16076 |
| Resin density +0.1 % | 748 / 1102 |
| Resin–water interfacial tension +1 % | 1593 / 2229 |
| Drive frequency +100 ppm | 225 / 233 |

To hold 10 nm, this state would need the temperature fixed to a few mK, the
resin sound speed known to about 3 ppm and its density to about 9 ppm, and the interfacial
tension to about 0.005 %. These are not attainable. The resin sound speed in
the material reference is itself an explicit placeholder. The sensitivity
follows from the operating regime: the acoustic radiation stress is cancelled
against itself to produce a mean load of order 1 Pa, so a part-per-thousand
change in any wave parameter leaves micrometre errors.

The first physical run (`force-convergence-224-physical/`) reused the
unperturbed surface space and is superseded; see its `SUPERSEDED.md`.

### 3. Linear stability (`afl coupled-stability`)

| Root | Slowest decay (1/s) | Unforced slowest decay (1/s) | Static softest mode, K-relative | Critical acoustic scale |
|---|---:|---:|---:|---:|
| Target root | −17.4 | −31.8 | 0.30 | 1.42 |
| Branch reached from flat (below) | −43.5 | −31.8 | — | 3.11 |

Both roots are linearly stable in the axisymmetric creeping-flow model, on
fluid meshes 64×12 and 128×24 (rates agree to 0.1 %). The analytic acoustic
tangent matches central differences of fresh wave solves to about 1e-7. At
the target root, acoustics softens the softest mechanical mode to 30 % of its
capillary-gravity stiffness and amplifies force errors 6.4× relative to the
frozen K⁻¹ used in the historical inverses. A 42 % larger acoustic stiffness
would destabilize it. This does not support the earlier review statement
that the acoustic stiffness exceeds capillarity by orders of magnitude at S04
states; that estimate was extrapolated from other commands. Non-axisymmetric
modes are not represented: the pinned m = 1 capillary-gravity stiffness is
about half of the lowest volume-preserving axisymmetric one (20.6 versus
38.9 kPa/m on the front face), so the softest physical mode is outside the
model.

### 4. Integrator stability and formation from flat (`afl form`)

The historical S05 scheme lags the whole acoustic load. At the stable target
root its one-step amplification is 1.62 at the S05 step of 0.5 s, 1.27 at
0.05 s and below one only near 0.02 s: the S05 trajectory cannot be read
physically. The new `"acoustic_load": "linearized"` mode puts the analytic
tangent in the implicit step.

Formation from the unforced flat state under the held command, with a
0.5 s ramp:

| Run | Final pupil error (µm) | Final interface speed (m/s) |
|---|---:|---:|
| Frozen load, dt 0.05 s (unstable range) | 251 / 234, still moving | 1.4e-3 |
| Frozen load, dt 0.01 s (stopped at 0.9 s) | 253.3 / 230.6 | 9e-11 |
| Linearized load, dt 0.05 s (adaptive to 0.006 s) | 253.3 / 230.6 | 4e-10 |
| Linearized load, dt 0.025 s | 253.3 / 230.6 | 2e-10 |

Two independent stable integrators reach the same steady state; the two
linearized runs end within 5 pm of each other. Because a
steady state of either scheme is an exact discrete force balance, this is a
second equilibrium of the same command. It is almost a mirror image of the
target: front vertex +103 µm instead of −150 µm, back vertex −111 µm instead
of +120 µm. The linearized growth rate stays negative along the whole path
(least margin −9 s⁻¹ near 0.25 s). The command is bistable and the flat
start falls into the wrong well. The target sag (±150 µm) is about a quarter
of the 416 µm acoustic wavelength, the spacing of standing-wave force wells.

### 5. Ray-coverage gate

The launch cone aims the rim ray at the target rim, and the trace rejects a
landing more than 1 pm outside the clear radius. Raising the front rim height
of the target root by 0.1 nm loses one ray (landing +1.8 pm outside); lowering
it by 1 nm loses none. The gate is a sign test on the rim height error at the
sub-nanometre level. The trace now reports `face_max_radius_beyond_clear_m`;
the gate itself is unchanged because the S04 protocol forbids shrinking the
pupil or discarding rays. Changing it requires declaring a physical aperture
stop in the apparatus.

## Verdict

For the surrogate, the design-grid success does not survive the surface
space, the back face is not wave-resolved to 10 nm, the equilibrium moves by
micrometres under sub-kelvin temperature changes, and the target is not the
state reached from flat. The state is linearly stable but neither robust nor
reachable. The failure is structural, not a missing digit of resolution: a
weak, nearly flat target is being held by cancelling large standing-wave
stresses from many idealized source regions. This motivates the stage-2
redesign; it does not establish that other targets or source spaces fail.

## Reproduce

Inputs are in `configs/S06-two-face-physical-checks/`; outputs are in
`artifacts/studies/S06-two-face-physical-checks/surrogate-2026-10-24/`.

```sh
afl synthesize configs/S04-emitter-synthesis/noa61-emitter-224-3p6mhz.json --out <run>/synthesis-224
afl equilibrate configs/S06-two-face-physical-checks/surrogate-held-command-224.json --out <run>/held-command-224
afl force-convergence configs/S06-two-face-physical-checks/force-convergence-surrogate-224-{wave,surface,surface104,physical}.json --out ...
afl coupled-stability configs/S06-two-face-physical-checks/coupled-stability-surrogate-224.json --out ...
afl form configs/S06-two-face-physical-checks/formation-{frozen-dt001,linearized-dt005,linearized-dt0025}.json --out ...
```

To check S04 itself, point `state_directory` (and `source_directory`) at
`artifacts/studies/S04-emitter-synthesis/noa61-emitter-2026-09-23/verify-448-c4-clear-spot/design104`.
The `held-command-224/surface52` root was stopped at the execution limit after
its first tangent and is incomplete; the surface trials above replace it. The
first single-run force study (`force-convergence-224/`) was stopped at the
execution limit under CPU oversubscription after two trials and was replaced
by the four split runs. Not added to `studies/ledger.json`: that reviewed
record and `STATUS.md` are rebuilt from the private tree.
