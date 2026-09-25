# Acoustic freeform research

This public repository contains source, research notes, source notebooks,
selected theory PDFs and figures. Keep private papers and simulation outputs
in the local research data tree, separate from any benchmark task package.
Do not publish additional data or add a remote without the user's instruction.
Do not forecast model scores from past results.

The current research priority is the general theory of acoustic fluid shaping
inside a cylinder, allowing arbitrary admissible source distributions and full
three-dimensional perturbations. Derive the forward coupling, inverse
conditions, source optimality equations, reachability and stability before new
numerical experiments. Finite arrays are restrictions of the continuous source
space. Cartesian targets (n_o, z_o, n_i, z_i) are a later specialization.
Continuation is an optional exploration tool, not the proposed inverse method.
The canonical LaTeX source is docs/theory/; the active compiled manuscript is
acoustic-fluid-shaping-theory.pdf at the repository root until its final home
is chosen. Preserve source and artifact milestones in the separate research
archive before replacing them.

Earlier numerical experiments remain evidence for their declared models only.
Preserve their outputs. Future experiments must test predictions of the general
theory, with independent configurations, provenance and verification. Establish
convergence with fixed physical commands before interpreting re-optimization.
Do not equate a frozen-sensitivity fit with coupled stability, or a quiescent
spectrum with stability of a streaming base flow. Thermal feedback, wall-layer
mean streaming and three-dimensional disturbances remain material questions.
Distinguish driven liquid optics, transient formation and curing. Axial
stigmatism does not imply aplanatism or achromatism. The 2026-09-06 wall-loss
finding and numerical limitations are documented in docs/numerical-stability.md.

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
