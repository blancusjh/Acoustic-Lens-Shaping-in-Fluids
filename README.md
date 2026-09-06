# Acoustic freeform lab

A private research workspace for **acoustic array → liquid motion → freeform optical surface**.
The longer-term endpoint is a cured resin optic. The first working milestone is a
transparent, non-axisymmetric forward reference with numerical checks and PyVista views.

**Current status:** a small-deformation spectral model, not a complete moving-interface
CFD solver or a calibrated resin experiment. A 16 × 16 coherent pressure array drives
two displaced foci through pulse sequences. All 256 complex element weights can be
set independently. This permits arbitrary drive patterns; it does not establish that
arbitrary target surfaces are reachable.

## Start here

- [Equations and the pulse-memory experiment](notebooks/01_forward_physics.ipynb)
- [Numerical verification and boundary sensitivity](notebooks/02_numerical_verification.ipynb)
- [Water–air animation](runs/water_air_v1/visuals/evolution.mp4)
- [Interactive cameras on a static snapshot](runs/water_air_v1/visuals/explore.html)
- [Four-panel preview](runs/water_air_v1/visuals/overview.png)
- [Reading map and primary sources](literature/reading-map.md)
- [Research decisions and next experiments](notes/research-plan.md)

The movies and HTML are generated local artifacts. A fresh checkout reproduces them
with the commands below. Downloaded articles remain in `literature/papers/`, excluded
from Git; their provenance and checksums are in `literature/manifest.json`. Original
files in Downloads were copied, not moved. No remote repository is configured.

## Run locally

```bash
uv sync
uv run lenslab run examples/water_air.toml --out runs/my_water_case
uv run lenslab render runs/my_water_case
uv run lenslab view runs/my_water_case
```

`view` opens the desktop PyVista window with a time slider. `render` writes PNG,
MP4, a static scene with interactive cameras in HTML, and VTK exports. Use
`--no-movie` for a quicker static render. Existing simulation output is protected:
choose a new output directory for each changed experiment.

```bash
uv run lenslab run examples/stokes_pair.toml --out runs/my_stokes_case
uv run pytest -q
uv run lenslab validate --out runs/my_validation
uv run jupyter lab --ip=127.0.0.1 --no-browser
```

Open the local URL printed by Jupyter. The paired `.py` notebook sources can also
be read without Jupyter. Use `uv run python scripts/execute_notebooks.py` to rebuild
and execute both notebooks and export readable HTML versions.

To supply arbitrary element controls, write a complex NumPy `.npy` array of shape
`(number_of_beams, array_rows, array_columns)` and pass `--weights path.npy` to `run`.
Each beam's element magnitudes must be at most one; pulse gains set its time envelope.
The complex values are saved in the result. Equivalently, use the Python API:

```python
from acoustic_freeform.config import load_experiment
from acoustic_freeform.simulation import run

experiment = load_experiment("examples/water_air.toml")
result = run(experiment, element_weights=my_complex_weights)
```

## Implemented physics

| Component | Current treatment |
|---|---|
| Array | Exact Fourier coefficients of square pressure patches; individual complex weights |
| Acoustics | Lossless angular spectrum, flat two-fluid interface, reflection and transmission |
| Surface forcing | Jump in cycle-averaged normal momentum flux from pressure **and velocity**; coherent cross terms retained |
| Water–air motion | Linear deep-fluid capillary–gravity inertia with weak-viscosity free-surface damping |
| Viscous pair motion | Separate linear two-fluid Stokes limit with mobility `1 / [2(μ₁+μ₂)k]` |
| Time | Exact constant-force modal step; midpoint sampling of smooth drive envelopes |
| Volume | Fixed zero mode in a periodic transverse cell |
| Visualization | Array phases, acoustic pressure slice, traction, moving surface, reconstructed slow flow, time traces |

The source specifies **outgoing peak pressure**, not voltage, watts, piezo displacement,
or total transducer-face velocity. Returning sound is absorbed at the source plane.
That plane is not a slow-flow wall; acoustic gap and liquid depth are different concepts.
The current equations are derived in notebook 01. Radiation stress follows
[Chesneau et al., PRE 106, 065104 (2022)](https://doi.org/10.1103/PhysRevE.106.065104).

## Evidence and limits

The analytic suite covers impedance matching, oblique propagation, total internal
reflection, pressure-release force, equal-fluid cancellation, coherent cross terms,
capillary dispersion, energy, Stokes relaxation, incompressibility and volume.
The convergence study also compares incident acoustics with an independently
integrated unbounded Rayleigh solution.

Initial recorded results in [convergence.json](runs/validation_v1/convergence.json):

| Check | Observed result | Interpretation |
|---|---|---|
| Smooth forcing time refinement | Coarse/middle height error ratio 4.27 | Consistent with second-order forcing integration |
| Incident pressure at five probes, 56 mm box | 0.266% relative complex L2 difference from Rayleigh integral | Local check; not a surface-error bound |
| Water–air, 56 vs 84 mm, through 60 ms | 0.478% relative central height L2 difference | Supports this observation window and region only |
| Density-matched Stokes pair, 56 vs 84 mm, through 300 ms | **8.85%** relative central height L2 difference | **Still domain sensitive; use as a periodic verification case** |

The interface is kept flat for acoustic scattering even while the linear surface
evolves. There is no finite resin volume/frame, contact line, acoustic attenuation,
streaming, heating, curing, viscoelasticity, or optical ray tracing. The weak damping
formula is not a general viscous two-fluid closure; see
[Denner, PRE 94, 023110 (2016)](https://doi.org/10.1103/PhysRevE.94.023110).
The Stokes limit removes inertia rather than approximating every viscosity regime.
For the example's longest mode, the estimated omitted-inertia/viscous ratio is
about 0.28; see the [regime check](notes/research-plan.md). It is a mathematical
limit case, not an established all-mode approximation for those physical liquids.

Large-deformation propagation feedback, finite chamber geometry, material calibration,
and experimental comparison remain necessary before a resin-shaping prediction.
No benchmark score or difficulty forecast is part of this project.

## Outputs and units

`result.npz` contains SI-valued coordinates, times, heights, normal velocities,
complex element weights, envelopes and radiation kernels. State integration is
float64; saved surface arrays are float32. `report.json` records parameters,
diagnostics, limitations, software versions, source hashes and a result checksum.
`experiment.toml` preserves the input. Initial conditions are flat and stationary.

The PyVista dashboard shows a **central region**, not the full periodic box. Its
first panel is a fixed peak-drive acoustic reference; the other panels follow time.
Surface vertical geometry is exaggerated and labeled. Flow arrows have fixed display
length; colors carry speed. They show interface-driven slow motion, not acoustic
streaming. VTK exports retain physical geometry and SI units without exaggeration.

The acoustic stress is an acoustic-cycle average. The animation advances slow time
and does not attempt to render each 1 MHz pressure oscillation.
