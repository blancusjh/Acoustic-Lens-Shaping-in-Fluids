# Software and evidence architecture

The package follows the sequence in the [general theory](theory/README.md).
Module boundaries describe computational responsibility, not physical
validation. The general continuous-source and full three-dimensional program
is a research objective; most current numerical implementations use finite
axisymmetric source and surface spaces.

| Package | Responsibility |
|---|---|
| `apparatus` | SI material and cylinder configurations; declared physical source regions |
| `optics` | Cartesian construction, shared conjugates and independent ray tracing |
| `mechanics` | Surface bases, capillarity, gravity, volume constraints, viscous mobility and required load |
| `acoustics` | Harmonic transmission, element operators and condensed source response |
| `inverse` | Finite-source fitting, conic/relaxation diagnostics and annulus choices |
| `forward` | Held-command coupled roots and declared formation integrators |
| `verify` | Analytic limits, response derivatives, fixed-state and fixed-command checks |
| `present` | Read-only figures and playback from saved physical states |
| `core` | Provenance and historical input-path resolution |
| `single_interface`, `planar` | Earlier separately scoped models, still tested |

`afl --help` lists current model-specific entry points. Each run must write to a
new directory under `artifacts/` and retain its configuration, provenance,
validation and report. Historical path strings are resolved by
[`configs/relocations.json`](../configs/relocations.json); new configs should
name the current `artifacts/studies/<study>/...` path directly. The old `lenslab`
command serves the single-interface model.

[The reviewed ledger](../studies/ledger.json) is the source for
[STATUS.md](../STATUS.md). The generated [report index](../studies/run-index.json)
records report hashes and whether configuration, validation and provenance are
present beside each report. It does not turn an optimizer exit flag or a file
into a scientific pass. Rebuild both outputs with
`python3 tools/build_research_index.py` and check freshness with `--check`.

## Acceptance order

1. Construct an admissible optical target and check all apparatus components,
   phase volumes and source access.
2. Derive the full-interface required load independently of source fitting.
3. Fit admissible source commands in the declared source space; report effort,
   residuals and the difference between frozen and coupled response.
4. Hold physical commands fixed while refining the wave and both surfaces.
   Record maximum height error, fixed-detector spot and complete ray coverage.
5. Assess stability about the relevant driven base flow, then integrate a
   physical formation trajectory. Curing is a separate process.

A quiescent spectrum does not certify a streaming base flow. Axial
stigmatism does not imply aplanatism or achromatism. Wall-layer mean
streaming, thermal feedback and three-dimensional disturbances remain open.
The [numerical-stability note](numerical-stability.md) records the wall-loss
finding and the limits of earlier simulations.

## Artifacts and presentation

The canonical theory source is `docs/theory/`; its active compiled PDF is
`artifacts/theory/acoustic-fluid-shaping-theory.pdf`. Active numerical outputs
are grouped in `artifacts/studies/`, while editable notebooks stay in
`notebooks/`. The earlier experiment directories retain their original report
formats. Outputs use SI units; any visualization conversion or geometric
exaggeration must be labelled. Browser playback reads saved physical states
and updates both the geometry and displayed physical time. Historical reports
and source snapshots remain in the separate research archive.
