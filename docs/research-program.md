# Acoustic shaping of both faces of a custom lens

Revision started 22 September 2026. The geometry objective is a maximum absolute
surface error of `1e-8 m` on **each** prescribed clear aperture, in a fixed
apparatus, without fitting away piston, tilt, scale or focal position. Cartesian
ovoids are the first target family. Their refractive indices define geometry;
imaging performance is a separate question.

On 23 September the user specified the **cycle-mean driven-liquid surface**.
The 10 nm gate therefore applies to mean geometry on both apertures, not the
instantaneous oscillating interface or a cured solid. Carrier amplitudes remain
diagnostics of operating conditions and model validity.

## Scientific architecture

1. Specify both target patches, supports, all conserved phase volumes and materials.
2. Obtain required traction as the multiplier field of the geometric constraint.
3. Synthesize that traction using the chamber's continuous acoustic source operator.
4. Approximate and optimize source support, positions, frequency and commands.
5. Independently solve the coupled forward problem and refine at fixed commands.
6. Establish formation from the unforced equilibrium and stability of both faces,
   including non-axisymmetric disturbances and the actual streaming base flow.
7. Assess mean geometric error, control error, parameter uncertainty and numerical
   uncertainty; report instantaneous carrier motion separately.

The maps in this sequence are conditional. In particular, a prescribed-traction
solution does not demonstrate that boundary sources can produce it. Mean-surface
accuracy alone does not satisfy an unqualified instantaneous 10 nm specification.

## Imported developments

The [user-supplied discussion](https://chatgpt.com/share/6ab2119a-197c-83e9-9801-223a8b8ea45e)
was retrieved on 22 September 2026. Its visible messages and acquisition manifest
are preserved privately in `references/discussions/2026-09-22-cartesian/`.
The source can be re-extracted with `tools/import_shared_conversation.py`.

Accepted mathematical developments are the geometric multiplier field, the
continuous quadratic source integral, source placement as a design variable,
variational design of the non-optical annulus, and explicit separation of mean
and instantaneous surface errors. The local pressure-node construction requires
source access near both sides of the interface and analytic local Cauchy data;
it is not a theorem of controllability from distant cylinder walls.

The discussion reports single-interface numerical formation and holding below
10 nm in an ideal source-port model. Its linked `sandbox:/mnt/data/` experiment
packages are not public download URLs. A local associated package was subsequently
found at `/Users/blancus/Downloads/acoustic_blender_model/source_data/` and copied
into the private discussion directory. It contains the single-interface solver,
source commands, a saved formation trajectory and its report. Inspecting those
arrays is an audit of imported evidence, not a fresh reproduction of the complete
dynamics. The reported results concern one interface rather than two
independently customized faces.

## Two-face apparatus contract

The general formulation allows either connected outer fluid or distinct outer
phases. One connected incompressible lens has one lens-volume constraint. A
sealed three-layer cylinder additionally conserves each outer phase volume;
there are two independent interface-volume constraints, not two freely chosen
pressure gauges. Reservoirs or hydraulic bypasses change those constraints and
must be declared as apparatus components.

The first new numerical implementation is a three-layer, axisymmetric,
isothermal, inviscid harmonic transmission model with nonlinear capillarity.
Sources are boundary velocity components in passive impedance ports; they are
not calibrated electrical drives. This isolates simultaneous mean-traction
reachability. It must not inherit the old one-surface solver's streaming or
dynamic validation. Material constants in an exploratory configuration are
hypotheses unless accompanied by measurements.

## Evidence and organization

`docs/theory/` is the canonical manuscript; `artifacts/theory/` contains its build.
`src/acoustic_freeform/` contains the two-face implementation in the apparatus,
acoustics, mechanics, inverse, forward and verify packages; the
earlier `single_interface/` and `planar/` implementations and outputs retain their
declared models. Each new campaign stores its input configuration, source
snapshot/checksums, commands, states, validation and a readable report together.
Promote results only after their independent checks; failed tests remain useful
evidence and remain in their campaign.

The pre-revision source and manuscript are preserved at
`/Users/blancus/Private/research-archives/acoustic-freeform-lab/revisions/2026-09-22-before-two-surface/repository-milestone.tar.gz`,
SHA-256 `8d33e2a75d3af1ea026ba0f9b0c96487eb77349b7895c793a8578f6eddc0bf60`.
Existing uncommitted research edits were included in that milestone.

The version 3 theory and implemented two-face model were additionally archived
at `revisions/2026-09-22-two-face-theory-and-model/source-and-theory.tar.gz` under
the same research archive, SHA-256
`05aebd6bd14bcbc66114aaed0d14fd0c46b70711a0e2e7145ece5a174f71489f`.
This is a source/theory milestone, not the final numerical-results snapshot.
Superseded manuscript-review images remain recoverable under
`revisions/2026-09-22-superseded-review/review/`; current review images are in
`artifacts/theory/review-v3p3/`. Superseded review milestones are preserved
in the separate research archive.

The completed two-face campaign, its source, theory and imported-data audit are
preserved at `revisions/2026-09-22-completed-two-face-campaign/milestone.tar.gz`,
SHA-256 `a7aa76ff5bc9d213bfe3c2dd4a9f0c8023de8fcf34cf1dd54d7f96c8b8198098`.
The snapshot includes the explicitly incomplete authority trial; the 112-port
and both closer-port cases completed their configured forward refinements.
