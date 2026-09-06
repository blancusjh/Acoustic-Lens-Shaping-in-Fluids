# Project architecture

The research repository has one executable package, one public command line, one
viewer application and one active reference experiment. Benchmark infrastructure
and production evaluation data live in a different directory tree.

`config` defines SI parameters and material provenance. `geometry` creates the
finite curved chamber from a surface state. `surface` owns nonlinear capillary
energy, exact volume constraints and the optical target. `acoustics` computes the
array response on that chamber; it does not prescribe the surface. `hydrodynamics`
computes a viscous mobility from finite-element fluid solves. `design` fits physical
wall velocities to a requested optical surface. `simulation` advances the actual
fluid state under the resulting force. `optics` evaluates that state independently
of the design objective. `validation` supplies analytic and refinement checks.

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
