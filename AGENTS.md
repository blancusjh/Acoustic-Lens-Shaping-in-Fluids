# Private acoustic freeform research

This is a local research repository, separate from any benchmark task package.
Keep research notes, papers, notebooks and simulation outputs here. Do not
publish or add a remote without the user's instruction. Do not forecast model
scores from past results.

The current user instruction resumes numerical investigation after the theoretical
formulation. Address spatial convergence, acoustic load feasibility, dynamical
stability and physical approach trajectories for Cartesian stigmatic interfaces.
Use the existing -50/+20 mm, 6 mm clear-aperture example unless later steering
changes it. Establish force and optical convergence with fixed physical drives
before interpreting re-optimization. The canonical LaTeX theory remains in
docs/theory/; its compiled artifact is in artifacts/theory/.
Preserve earlier experiment outputs. New experiments require separate named
artifact directories, configuration, provenance and verification. Curing remains
a distinct physical model; stable driven liquid optics is the current objective.
Snapshot executable sources before long numerical runs. Keep incomplete runs
explicitly marked. Do not equate frozen-sensitivity optimization with recomputed
stability, or a quiescent spectrum with stability of a streaming base flow.
The 2026-09-06 wall-loss diagnostic is load-bearing: viscous boundary losses were
larger than bulk absorption in the original 1.8 MHz candidate and require a
re-solved acoustic field. See docs/numerical-stability.md for the campaign.

The current extension maps signed Cartesian-diopter conjugates (n_o, z_o,
n_i, z_i), using Silva-Lora and Torres (2020), to a feasible surface, required
traction and array amplitudes/phases. Preserve the original collimated result.
Distinguish general optical construction from the liquid-air apparatus, and
prescribed in-resin illumination from a source refracted by the bottom window.
Axial stigmatism does not automatically imply aplanatism or achromatism.

Benchmark tasks, Boreal transcripts, scores and QA analyses belong exclusively
in /Users/blancus/Private/benchmark-archives, never in this project or its
scientific artifacts. Superseded exploratory outputs belong in the separate
/Users/blancus/Private/research-archives tree. Keep active results in artifacts/
with a configuration, provenance and validation report; no loose outputs.

Clearly distinguish implemented physics, approximations, numerical
verification and experimental validation. Every rendered part must correspond
to a declared apparatus component. Do not substitute interpolation of a desired
shape for a forward physical trajectory. Browser playback must actually update
geometry and the displayed physical time.

Use SI units in the solver and files. Label unit conversions and geometric
exaggeration in visualizations. Academic claims need primary references.
Validate mechanisms against independent analytic limits and convergence,
rather than tests that merely reproduce implementation formulas. Never infer
physical maturity from a convincing rendering.
