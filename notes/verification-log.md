# First milestone verification — 5 September 2026

Executed locally on the user's Apple Silicon Mac in the project's `uv` environment.
Exact package versions are locked in `uv.lock`; simulation reports include software
versions and source hashes. All quantities below concern this declared reference model.

- `uv run pytest -q`: **16 passed**. Includes analytic acoustic matching/force limits,
  coherent superposition, capillary dispersion, energy, Stokes relaxation and
  incompressibility. An additional evanescent test checks total normal flux including
  incident–reflected interference, which individual evanescent wave powers miss.
- `uv run lenslab validate --out runs/validation_v1`: completed. Independent Rayleigh
  quadrature, time refinement, spatial refinement, and both domain studies recorded
  in `runs/validation_v1/convergence.json`. Boundary and regime limitations remain
  documented in the README and research plan.
- Both example configurations ran through the CLI and loaded back with matching
  result checksums. Numerical results and render provenance are recorded separately.
- Both notebooks executed without error outputs: six code cells in notebook 01 and
  four in notebook 02. Readable HTML exports include the figures.
- Both PyVista animations encoded as H.264, 1600 × 1120, 121 frames, 6.05 s duration.
  This is presentation duration; simulated time is 60 ms or 300 ms respectively.
- VTK surface heights exactly match the saved height array and physical z coordinates;
  the VTK export has no display exaggeration.
- Standalone PyVista HTML scenes were exported without external script URLs. Camera
  interaction is a static-snapshot view; time exploration is provided by the desktop
  slider and the movie. No claim of browser-by-browser testing is made.
- Final water–air still and an extracted movie frame were visually inspected. The
  forced still shows 42 ms with nonzero traction; the extracted 20 ms frame shows
  drive-off surface motion. Explicit renders prevent stale screenshot buffers.
- All seven PDF hashes match `literature/manifest.json`.
- `ruff check` and `ruff format --check` pass for source, tests and scripts.

No experimental validation, resin calibration, curing calculation, moving-boundary
CFD verification, or optical-quality claim follows from these checks.
