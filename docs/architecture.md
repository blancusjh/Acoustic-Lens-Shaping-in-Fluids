# Software and evidence architecture

The package follows the sequence in the [general theory](theory/README.md).
Module boundaries describe computational responsibility, not physical
validation. The general continuous-source and full three-dimensional program
is a research objective; most current numerical implementations use finite
axisymmetric source and surface spaces.

## Repository layout

| Location | Responsibility |
|---|---|
| `src/acoustic_freeform/` | Physics and algorithm library, research steps and the `afl` command |
| `configs/<study>/` | Immutable inputs grouped by study; `configs/shared/` holds shared material records |
| `studies/<study>/` | Question, protocol, reviewed verdict and detailed records of each study |
| `artifacts/studies/<study>/` | Run outputs, private and excluded from Git |
| `notebooks/` | Editable presentations; executed copies live with their study artifacts |
| `tools/documents/` | Manuscript builds, algebra checks and the evidence index |
| `tools/audits/` | Study-specific audits of saved data |
| `tools/presentation/` | Notebook execution, README figures and browser playback checks |
| `docs/` | Program-wide theory, contract and this architecture |
| `web/` | Browser viewer assets for the single-interface model |

## Package

Library packages compute; they do not choose output directories or write run
records. Only `steps/` reads experiment configurations and writes a run
directory. A library module may be imported by any later package in the table,
never by an earlier one, except that `optics.stigmatic` uses
`mechanics.surface` to build its target patches.

| Package | Responsibility |
|---|---|
| `core` | Repository paths, historical path relocation and execution provenance |
| `apparatus` | SI material and cylinder configurations; declared physical source regions |
| `optics` | Cartesian construction, shared conjugates and independent ray tracing |
| `mechanics` | Surface bases, capillarity, gravity, volume constraints, viscous mobility and required traction |
| `acoustics` | Harmonic transmission, element operators and condensed source response |
| `inverse` | Frozen-operator objectives, convex and relaxation diagnostics, annulus choices |
| `forward` | Held-command coupled roots and formation time-stepping kernels |
| `verify` | Analytic limits, response algebra and acceptance metrics |
| `present` | Read-only figures and playback from saved physical states |
| `steps` | One module per `afl` command: load configuration, call the library, write the run |
| `models.single_interface` | Earlier single liquid–air interface model (S01) with its own command line |
| `models.planar` | Planar-interface verification limits |

Some step modules still hold the main loop of their algorithm, for example
the formation time loops and inverse drivers. These loops move into the
library as each algorithm is revised and independently tested.

## Commands

`afl --help` lists every step. Each step takes an existing configuration and a
new `--out` directory under `artifacts/`; it records the configuration,
provenance, validation and report there. Historical input paths in
configurations and saved artifacts are resolved by
[`configs/relocations.json`](../configs/relocations.json); new configs should
name current paths directly.

| Research stage | Commands |
|---|---|
| Optical target and required load | `target`, `required-load`, `ideal-load-formation` |
| Finite-source synthesis | `screen`, `synthesize`, `conic`, `relaxation`, `free-annulus`, `observations` |
| Held-command equilibrium | `equilibrate`, `independent-pair` |
| Formation | `form`, `inertial-formation` |
| Verification | `verify-wave`, `analytic-limits`, `kernel-audit`, `analytic-response`, `coupled-response`, `frozen-extremum`, `stigmatic-state` |
| Earlier models | `single-interface ...` (also `lenslab`), `planar ...` |

Execution provenance archives every source file of the package, the Git
commit, uncommitted changes and the command line. Runs recorded between the
September 25 reorganisation and 2026-10-24 archived only the `core` package
sources; their solver sources must be recovered from Git history and the
research archive.

[The reviewed ledger](../studies/ledger.json) is the source for
[STATUS.md](../STATUS.md). The generated [report index](../studies/run-index.json)
records report hashes and whether configuration, validation and provenance are
present beside each report. It does not turn an optimizer exit flag or a file
into a scientific pass. Rebuild both outputs with
`python3 tools/documents/build_research_index.py` and check freshness with `--check`.

No automated test suite is currently in the repository. Independent analytic
checks are available as `afl analytic-limits` and `afl planar validate`.

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
The [numerical-stability note](../studies/S01-single-interface/numerical-stability.md) records the wall-loss
finding and the limits of earlier simulations.

## Artifacts and presentation

The canonical theory source is `docs/theory/`; its compiled PDF is private,
in `artifacts/theory/acoustic-fluid-shaping-theory.pdf`. Active numerical outputs
are grouped in `artifacts/studies/`, while editable notebooks stay in
`notebooks/`. The earlier experiment directories retain their original report
formats. Outputs use SI units; any visualization conversion or geometric
exaggeration must be labelled. Browser playback reads saved physical states
and updates both the geometry and displayed physical time. Historical reports
and source snapshots remain in the separate research archive.
