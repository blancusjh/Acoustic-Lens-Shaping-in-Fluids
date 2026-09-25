# Project architecture

The research repository has one Python package with separately declared physical
models. The active two-face program is `acoustic_freeform.dual`, invoked with
`python -m acoustic_freeform.dual CONFIG --out artifacts/EXPERIMENT`.
The established `lenslab` command and viewer serve the earlier single-face model.

The active precision submodules separate frozen carrier-aware screening
(`dual.precision`), sequential convex steps (`dual.conic_precision`), optional
semidefinite diagnostics (`dual.relaxation`), and independent fixed-command
campaigns. `dual.sources` declares physical boundary regions and SI commands.
For many sources, the acoustic factorization is reused across source blocks;
interface responses retain all coherent cross terms without storing every
full-volume source field. Full fields are recomputed for actual fixed commands.
`dual.linear` optionally eliminates element-interior unknowns exactly before
factorization and recovers all pressures afterward. The default full solve
remains the reference. Matrix-free stationary root iterations are an optional
numerical algorithm, never a physical time evolution.
The reused-condensed option updates every physical wave operator, using an
earlier factorization only as a residual-checked preconditioner. Localized weak
flux assembly retains exactly the elements supporting the interface rows.

The general theoretical formulation is independent of that numerical apparatus.
`docs/theory/main.tex` includes ordered section sources and a primary-source
bibliography. `tools/build_theory.py` compiles only the document and records
source/PDF checksums in `artifacts/theory/`. Auxiliary TeX files and review images
remain there, while downloaded sources stay in `references/papers/`. The source
map distinguishes established results, local derivations and unresolved model
choices. It neither runs nor changes the physical experiments described below.

`cartesian` constructs the signed optical vertex branch and independently implements
the paper's parameterization and vector Snell refraction. `config` defines SI
parameters and material provenance. Its optional `[diopter]` section accepts the
four paper parameters without duplicating optical settings in `[lens]`. `geometry` creates the
finite curved chamber from a surface state. `surface` owns nonlinear capillary
energy, exact volume constraints and the optical target. `acoustics` computes the
array response on that chamber; it does not prescribe the surface. `hydrodynamics`
computes a viscous mobility from finite-element fluid solves. `design` fits physical
wall velocities to a requested optical surface. `simulation` advances the actual
fluid state under the resulting force. `optics` evaluates that state independently
of the design objective. `validation` supplies analytic and refinement checks.
`excitation` exports the solved complex wall velocities, their amplitude/phase
convention, the required surface perturbation and the continuum traction balance.
`stationary` runs joint inverse design, an independent fixed-drive equilibrium
check, spatial refinement and optional continuous local stability analysis.
`stationary_visualization` renders these states with PyVista and shows their
refinement comparison; it does not generate a physical timeline.

The maintained-state extension adds `mapping` for exact graph geometry,
`streaming` for distributed bulk absorption forcing and its stationary reciprocal
kernels, `dynamics` for inertial fluid reductions and acoustic sensitivities,
and `feedback` for explicitly sampled surface observation/control. `time_step`
advances the ALE momentum and kinematic equations; `transient` records formation
or disturbed-state recovery. A stationary streaming force is never substituted
for a distributed load in the inertial plant. `provenance` archives executable
sources at run entry so later edits cannot alter execution evidence.

The verification package separates fixed-state cavity errors, fixed-drive
coupled equilibrium refinement, absorption/energy budgets and frozen-target
frequency screening. A screened drive is not a certified equilibrium.

The dependency direction is geometry/physics → integration → output. Visualization
reads completed simulation artifacts. The browser has no optimization code and
cannot replace a state with the target. All displayed times refer to saved fluid
states; camera interpolation is unrelated to physical evolution.

The earlier flat-interface model is isolated under `verification/planar`. Its
periodic-domain assumptions do not enter the finite lens. It remains callable as
`lenslab planar ...` and has its own verification tests and configurations.

## Artifact contract

Two-face campaigns contain the requested `config.json`, an execution-source
archive in `provenance/`, independent analytic checks in `validation.json`, and
per-case source commands and surface/pressure coefficients in SI `.npz` files.
`results.json` and `report.md` distinguish mean error, carrier motion and the
status of fixed-command refinement. These stationary states have no physical
timeline. The non-optical annulus can be varied by the inverse, but the
Cartesian aperture and all phase volumes stay fixed.

The following additional formats belong to the earlier single-face solver:

Each completed experiment contains:

- `input.toml` and `configuration.json`: requested and resolved inputs;
- `trajectory.npz`: physical times, surface coefficients, complex array drives,
  acoustic pressure and slow fluid velocity samples, radiation pressure;
- `report.json`: material provenance, initial/target/final optics, per-frame
  diagnostics, volume, power, residuals and source/trajectory hashes;
- `viewer/`: self-contained HTML, JavaScript, styles and numerical data;
- `geometry/`: native PyVista/VTK apparatus meshes in metres;
- `figures/`: scientific figures and a PyVista apparatus rendering;
- `verification/` and `spatial-convergence.json`: independent checks and browser evidence.

Stationary experiments use `stationary.npz` instead of `trajectory.npz`, with
initial, target and computed coefficients plus the complex holding drive. They
contain `hold-drive.csv`, `surface-and-load.csv` and `excitation-definition.json`
at the result root. A supplied optimizer seed is retained as `initial-drive.json`.
Their interactive scenes are `stationary-viewer.html` and `optics-viewer.html`.
Neither these files nor optimization iteration numbers represent time evolution.
`spatial-convergence.json` and the refined coefficient files retain the same drive
while changing acoustic resolution and surface modes. Numerical force balance,
spatial accuracy and physical stability are separate results.

`lenslab excitation <result>` adds `excitation/hold-drive.csv`, the full
`drive-program.csv`, surface/load and traction-balance tables, and a definition
file stating phase, amplitude, spatial taper, optical conjugates and calibration
scope. This export reads the computed trajectory; it does not refit the array.

Solver arrays use SI units and float64 arithmetic. Viewer geometry uses millimetres,
with the conversion declared in the payload. Sampled field data are float32 for
display; surface integration and optical metrics use float64 coefficients.

Source notebooks are the canonical `.ipynb` files. Execution copies and HTML exports
belong in `artifacts/notebooks/`, preventing three independently edited versions of
the same document. Downloaded articles remain in `references/papers/`; the manifest
is versioned, the private PDF copies are not redistributed.

## Viewer build

PyVista constructs the apparatus meshes. A small VTK.js runtime is built from
explicit ES-module imports in `web/runtime.js`, using `web/package-lock.json`.
`lenslab render` combines that runtime with the saved numerical trajectory. The
result can be opened as a local file without a running solver or internet access.
The Flask command is only a convenient loopback file server.
