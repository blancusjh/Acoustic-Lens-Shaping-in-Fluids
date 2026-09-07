# Project architecture

The research repository has one executable package, one public command line, one
viewer application and separately configured optical experiments. Benchmark infrastructure
and production evaluation data live in a different directory tree.

The current theory-first milestone is independent of that numerical apparatus.
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

The dependency direction is geometry/physics → integration → output. Visualization
reads completed simulation artifacts. The browser has no optimization code and
cannot replace a state with the target. All displayed times refer to saved fluid
states; camera interpolation is unrelated to physical evolution.

The earlier flat-interface model is isolated under `verification/planar`. Its
periodic-domain assumptions do not enter the finite lens. It remains callable as
`lenslab planar ...` and has its own verification tests and configurations.

## Artifact contract

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
