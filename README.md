# Acoustic Freeform Lab

Private scientific software for **acoustic array → finite liquid motion → optical surface**.
The current milestone is the **general theoretical formulation before further
numerics**, centered on Cartesian stigmatic interfaces. Read the
[LaTeX manuscript and source guide](docs/theory/README.md), or open the
[compiled theoretical document](artifacts/theory/acoustic-cartesian-theory.pdf).
It derives optical targets, required traction, actuator feasibility, stability
and pulse/control objectives without selecting an apparatus. It reports no new
simulation or optimized drive. Build it with `python3 tools/build_theory.py`.

The established numerical result is an axisymmetric, acoustically maintained **aspherical liquid lens**.
Its shape is produced by a coupled fluid simulation and evaluated with optical rays.

The [Cartesian-diopter extension](docs/cartesian-diopters.md) accepts the incident
index/object distance and transmitted index/image distance `(n_o, z_o, n_i, z_i)`.
It constructs the exact optical target, required surface perturbation and traction,
then searches for array excitations in the stated liquid–air apparatus. The new
finite-conjugate drive is a **preliminary stationary candidate**: refinement changes
its ray RMS from 0.389 to 8.036 µm, and the nominal model has one unstable mode.
It is not yet a demonstrated, stable stigmatic diopter. Start with the
[finite-conjugate notebook](notebooks/02_cartesian_diopters.ipynb) and
[four-parameter configuration](configs/lenses/noa61_cartesian_50_20.toml).
The [research result](docs/cartesian-results.md) records the excitation, numerical
limitations and reproducible commands. The original collimated asphere below remains
the established result.

Start with the [interactive time viewer](artifacts/asphere/viewer/index.html), the
[optical validation figure](artifacts/asphere/figures/optical-validation.png), or the
[equations and analysis notebook](notebooks/01_finite_asphere.ipynb).
The viewer is self-contained and works offline. Its play button and time slider
update the actual computed surface, acoustic field, drive phases, flow and rays.

The lens has a **6 mm clear aperture**, a **20 mm focal distance** and a **0.735 mm
cap height** above an 8 mm diameter chamber. A fitted conic gives **K = −2.31027**.
The surface departs from its best-fit sphere by **4.846 µm peak to valley**.

| Optical comparison, same clear aperture | Geometric RMS ray radius at each surface's best focus |
|---|---:|
| Unforced liquid, same chamber and fill volume | 192.36 µm |
| Best-fit sphere, freely refocused | 47.59 µm |
| Computed driven asphere | 0.210 µm |
| Same drive, refined acoustics and 48 surface modes | 0.213 µm |

These are monochromatic geometric-optics results within the stated numerical model,
not measured focal spots. Diffraction, material calibration and omitted physics
limit what can be inferred for an experiment. See [model scope](docs/model.md) and
[numerical evidence](docs/validation.md). NOA 61's **liquid** optical and fluid data
come from the manufacturer; sound speed and attenuation remain explicit assumptions.

## Use the existing result

```bash
uv run lenslab view artifacts/asphere
```

This serves the viewer at `http://127.0.0.1:8765` and opens a browser tab. Alternatively,
open `artifacts/asphere/viewer/index.html` directly. The four camera presets show the
apparatus, free optical surface, light rays and acoustic cross-section. Housing,
window, liquid, array and fields have individual controls. Geometry is not exaggerated.

## Reproduce

```bash
uv sync --dev
npm --prefix web ci
npm --prefix web run build
uv run lenslab simulate configs/lenses/noa61_asphere.toml --out artifacts/my_asphere
uv run lenslab render artifacts/my_asphere
uv run lenslab view artifacts/my_asphere
```

The recorded reference integrated 1.8 seconds of fluid time in approximately
384 seconds on this machine. A completed result is protected from replacement;
use a new result directory for a new run.

For a finite object/image pair:

```bash
uv run lenslab simulate configs/lenses/noa61_cartesian_50_20.toml --out artifacts/my_cartesian
uv run lenslab excitation artifacts/my_cartesian
uv run lenslab render artifacts/my_cartesian
```

This restricted-drive transient settles with 48.4 µm geometric ray RMS at the requested
image plane; it does not produce the exact Cartesian target. Its excitation export
provides holding amplitudes/phases for every coherent array
row, a complete drive schedule, and the required surface perturbation. These are
peak physical wall velocities. Electrical voltages require transducer calibration.

For the separate stationary inverse design:

```bash
uv run lenslab stationary configs/lenses/noa61_cartesian_50_20_stationary.toml \
  --seed configs/initialization/cartesian_50_20.json --starts 1 \
  --out artifacts/my_stationary --stability
uv run lenslab verify artifacts/my_stationary
uv run lenslab render artifacts/my_stationary
```

The seed makes the recorded optimization branch reproducible. `stationary` does
not compute an approach trajectory. Its figures compare design-mesh and refined
ray fans, and its PyVista scene states the stability and refinement results.

```bash
uv run pytest -q
uv run lenslab verify artifacts/asphere
uv run lenslab verify artifacts/asphere --spatial
uv run lenslab replay artifacts/asphere --step 0.01 --out artifacts/replay_dt001
uv run python tools/execute_notebooks.py
uv run python tools/verify_viewer.py artifacts/asphere
```

`verify --spatial` holds the final physical drive fixed while changing acoustic
order, mesh resolution and the surface basis. `replay` uses the recorded
sample-and-hold drive without shape feedback. Notebook exports go under
`artifacts/notebooks/`; source notebooks remain clean.

## Project map

| Directory | Responsibility |
|---|---|
| `src/acoustic_freeform/lens/` | Finite chamber, acoustics, viscous flow, capillarity, control, optics and rendering |
| `src/acoustic_freeform/verification/planar/` | Earlier planar limits, retained as isolated verification cases |
| `configs/` | Versioned physical apparatus and numerical inputs |
| `tests/` | Analytic limits, conservation and dissipation checks |
| `web/` | Time viewer source and locked frontend dependencies |
| `references/` | Primary-source reading map, paper manifest and private PDF library |
| `docs/` | Model contract, evidence, architecture and research decisions |
| `docs/theory/` | General LaTeX formulation, equations, bibliography and derivation provenance |
| `notebooks/` | Canonical Jupyter research notebooks |
| `tools/` | Reproducible notebook and browser verification utilities |
| `artifacts/<experiment>/` | Generated configuration, trajectory, report, viewer, figures and verification |
| `artifacts/theory/` | Compiled manuscript, source checksums and document build/review files |

The active experiments are `artifacts/asphere/`, the restricted-drive
`artifacts/cartesian-50-20/` trajectory, and the preliminary
`artifacts/cartesian-50-20-stationary-p4/` candidate. Generated outputs, downloaded papers,
environments and dependency bundles are excluded from Git. This is a local private
repository with no configured remote.

Benchmark material is stored separately at
`/Users/blancus/Private/benchmark-archives/acoustic-lens-shaping/`.
Superseded research prototypes are at
`/Users/blancus/Private/research-archives/acoustic-freeform-lab/prototype-2026-09-05/`.
Neither is an input to this scientific software.

Further numerical research follows review of the theoretical formulation.
It will require calibration of the acoustic apparatus and a justified treatment
of second-order flow, inertia and thermal effects. Fully independent azimuthal control and
non-axisymmetric freeform optics require a 3D extension; the 256 depicted sectors
are tied into 16 coherent rows in this asphere experiment. Curing is a later model.
