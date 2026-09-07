# Acoustic Freeform Lab

Private scientific software for **acoustic array → liquid motion → optical surface**.
The current result is a **Cartesian liquid diopter that forms and recovers under a
fixed 1 MHz excitation in the declared axisymmetric, isothermal numerical model**.
Research, literature and results remain separate from benchmark task packages.

Open the [interactive formation viewer](artifacts/cartesian-maintenance-2026-09-06/step-formation-dt00125/viewer/index.html),
[verification figure](artifacts/cartesian-maintenance-2026-09-06/verification/maintenance.png),
or [research notebook](notebooks/03_stable_cartesian.ipynb).
The viewer initially shows the last computed state at **0.2 s**. Restart/play
shows the actual fluid trajectory with cylinder, base, liquid, rim, array,
acoustic pressure, mean flow and refracted rays. It works offline.

For `n_o=1.52`, `z_o=-50 mm`, `n_i=1`, `z_i=+20 mm`, the 6 mm clear-aperture
surface has approximately **0.05 µm geometric ray RMS** at the requested image
plane. The same physical drives give 0.05410, 0.04995 and 0.04956 µm on three
spatial resolutions. Formation and recovery runs use fixed drives, actual fluid
inertia and convection, bulk absorption flow, and viscous acoustic wall losses.
The early trajectory remains sensitive to time-step refinement, so a precise
settling time is not claimed.

These are constant-index geometric-optics results for the mean surface, not
measured focal spots or experimental stability. Thermal feedback, wall-layer
mean streaming, non-axisymmetric disturbances and acoustic index modulation
remain open. Sound speed, attenuation and loaded transducer response require
calibration. The 256 depicted sectors are tied into **16 coherent array rows**.

- [Numerical findings, equations and limitations](docs/numerical-stability.md)
- [Reproduction commands and artifact guide](docs/stable-cartesian-reproduction.md)
- [Holding amplitudes and phases](artifacts/cartesian-maintenance-2026-09-06/operating-state/hold-drive.csv)
- [Excitation conventions](artifacts/cartesian-maintenance-2026-09-06/operating-state/excitation-definition.json)
- [Machine-readable verification](artifacts/cartesian-maintenance-2026-09-06/verification/maintenance.json)
- [Apparatus, boundary conditions and model scope](docs/model.md)

## Open or reproduce the current result

```bash
uv sync --dev
npm --prefix web ci
npm --prefix web run build
uv run lenslab view artifacts/cartesian-maintenance-2026-09-06/step-formation-dt00125
```

This serves the viewer at `http://127.0.0.1:8765` and opens a browser tab. It can
also be opened directly as a local HTML file. Housing, window, liquid, array
and fields have individual controls; four camera presets show the apparatus,
optical surface, rays and acoustic cross-section. Geometry is not exaggerated.

Re-run fixed-drive formation from the saved operating state:

```bash
uv run lenslab evolve artifacts/cartesian-maintenance-2026-09-06/operating-state \
  --controller artifacts/cartesian-maintenance-2026-09-06/operating-state/hold-model \
  --initial rest_fixed --step 0.00125 --duration 0.2 --out artifacts/my_formation
uv run lenslab excitation artifacts/my_formation
uv run lenslab render artifacts/my_formation
```

The holding file has **zero feedback gain**; it supplies a radiation Jacobian
for the implicit numerical update. Every physical drive remains unchanged.
Completed results are protected from replacement. The reproduction guide also
covers inverse design, fixed-drive regridding, convergence and control models.
Electrical voltages cannot be exported without transducer calibration.

## General formulation and earlier milestones

The [LaTeX manuscript and source guide](docs/theory/README.md) and
[compiled theoretical document](artifacts/theory/acoustic-cartesian-theory.pdf)
derive optical targets, required traction, actuator feasibility, stability and
pulse/control objectives. Build with `python3 tools/build_theory.py`.
The [Cartesian-diopter construction](docs/cartesian-diopters.md) accepts signed
optical parameters `(n_o, z_o, n_i, z_i)`; apparatus feasibility is a separate
inverse problem. Axial stigmatism does not imply achromatism or aplanatism.

Earlier results remain inspectable under their original model assumptions:

| Milestone | Evidence |
|---|---|
| Original collimated asphere, earlier Stokes/radiation model | [Time viewer](artifacts/asphere/viewer/index.html), [notebook](notebooks/01_finite_asphere.ipynb) |
| First Cartesian construction and restricted-drive trajectory | [Notebook](notebooks/02_cartesian_diopters.ipynb), [results](docs/cartesian-results.md) |
| Original stationary candidate and failed refinement | [Historical analysis](docs/cartesian-results.md) |
| Corrected geometry, inertia, feedback and wall-loss investigation | [Campaign history](docs/numerical-stability.md) |

The old 0.389 µm stationary Cartesian fit did not survive refinement and geometric
corrections. It is superseded by the independently checked 1 MHz result above.
Earlier artifacts are preserved, not silently regenerated with a new model.

## Development and project map

```bash
uv run pytest -q
uv run ruff check src tests tools
uv run python tools/execute_notebooks.py notebooks/03_stable_cartesian.ipynb
uv run python tools/verify_viewer.py artifacts/cartesian-maintenance-2026-09-06/step-formation-dt00125
```

| Directory | Responsibility |
|---|---|
| `src/acoustic_freeform/lens/` | Curved chamber, acoustics, flow, capillarity, inverse design, dynamics, optics and rendering |
| `src/acoustic_freeform/verification/` | Analytic limits, geometry checks, coupled refinement and energy diagnostics |
| `configs/` | Versioned physical apparatus, numerical inputs and optimizer seeds |
| `tests/` | Independent limits, conservation and dissipation checks |
| `web/` | Time viewer source and locked frontend dependencies |
| `references/` | Primary-source reading map, manifest and private PDF library |
| `docs/` | Model contract, evidence, architecture and research decisions |
| `docs/theory/` | General LaTeX formulation and bibliography |
| `notebooks/` | Clean source Jupyter notebooks; execution/HTML copies go to `artifacts/notebooks/` |
| `tools/` | Notebook, campaign reporting and browser verification utilities |
| `artifacts/<experiment>/` | Configuration, numerical states, report, provenance, visualization and checks |

The current campaign is `artifacts/cartesian-maintenance-2026-09-06/`.
Downloaded papers, outputs, environments and dependency bundles are excluded
from Git. This local private repository has no configured remote.
Benchmark material remains exclusively in `/Users/blancus/Private/benchmark-archives/`.
Older research prototypes remain in `/Users/blancus/Private/research-archives/`.
Neither directory supplies inputs to this scientific software.
