# Acoustic Freeform Lab

Private research on general acoustic fluid shaping inside a cylinder, including
**both faces of a custom liquid lens**. Cartesian ovoids are a worked optical
specialization of the general theory. The lens goal is **10 nm maximum
absolute cycle-mean error on each fixed clear aperture** of the driven liquid
(user-confirmed 23 September 2026). Carrier motion is a separate diagnostic,
not an additive mean-error penalty. No physical 10 nm result is established.

## Start here

- [SSL generator vision and theory — PDF](artifacts/theory/ssl-generator/ssl-generator-vision.pdf) · [LaTeX source](docs/theory/ssl-generator/main.tex): proposed fixed-prescription compiler with user-supplied z1, central thickness and frame radius; explicit stationary verification and later adaptive operation.
- [Before switch-on and actual surface/spot evolution](artifacts/notebooks/10_emitter_formation.executed.ipynb) · [Browser animation](artifacts/notebooks/10_emitter_formation.html): both flat unforced interfaces, initial spot, and 22 saved emitter-driven Stokes frames through 7.1875 s. This partial, unrefined diagnostic does **not** form the target; it was stopped after the conservative Reynolds estimate undermined quantitative creeping-flow interpretation.
- [Latest emitter notebook — executed](artifacts/notebooks/09_emitter_driven_equilibrium.executed.ipynb) · [Open in a browser](artifacts/notebooks/09_emitter_driven_equilibrium.html): 0.513 µm stationary spot, both surfaces, idealized 3D source array, fresh acoustic/load checks and an embedded pressure-wavefront movie. Carrier animation, not formation; height and ray-coverage gates remain unmet.
- [Current NOA 61 / water emitter campaign](artifacts/noa61-emitter-2026-09-23/README.md): held-command design-grid equilibrium gives **0.513 µm** maximum geometric spot radius. User accepts approximately 0.5 µm for the spot; **20.2 / 24.4 nm** surface errors, incomplete ray coverage and unfinished refinement prevent a full joint pass. No physical accuracy claim.
- [Required load before emitters](docs/ideal-load-first.md) · [Executed viscous-formation notebook](artifacts/notebooks/08_prescribed_load_formation.executed.ipynb): new ideal-load three-fluid Stokes diagnostic, with both moving surfaces and spot evolution. At 20 s its refined-fluid run reaches approximately **0.175 / 0.108 nm** pupil error and **0.0285 µm** maximum geometric spot, with all final rays transmitted. Hypothetical viscosities; **not an acoustic or physical accuracy result**.
- [Joint stigmatic correction](docs/joint-stigmatic-correction.md) · [Executed notebook](artifacts/notebooks/07_stigmatic_correction.executed.ipynb) · [Browser notebook](artifacts/notebooks/07_stigmatic_correction.html): common intermediate conjugate, actual two-face errors, full spot diagram and 448-region 3D setup. The current verified direct-objective candidate fails: refined front/back errors **370 / 741 nm**, maximum geometric spot radius **51.9 µm**. [Study index](artifacts/dual-stigmatic-2026-09-23/README.md).
- [Both faces forming in time — open the animation](artifacts/notebooks/06_two_face_formation/viewer.html): cylinder and 112 source regions, two moving interfaces, both errors, and the joint spot diagram. [Executed notebook](artifacts/notebooks/06_two_face_formation.executed.ipynb) · [Browser notebook](artifacts/notebooks/06_two_face_formation.html). Initial 0–8 ms reduced-model diagnostic; no settling or 10 nm claim.
- [Theory manuscript](artifacts/theory/acoustic-fluid-shaping-theory.pdf) and [LaTeX guide](docs/theory/README.md).
- [Research contract](docs/research-program.md): source space, volumes, formation and stability.
- [Two-face results](docs/dual-cartesian-results.md): previous refined mean errors **284 / 97 nm**; carrier **255 / 62 nm**.
- [Active precision work](docs/precision-program.md): experiments and acceptance gates.
- [Earlier precision evidence](docs/precision-results.md): frozen fits versus coupled surfaces, before the shared-conjugate correction.
- [Executed two-face notebook](artifacts/notebooks/04_dual_cartesian.html): plots, refinement audit and physical-time acoustic-pressure animation ([Jupyter](artifacts/notebooks/04_dual_cartesian.executed.ipynb)). No surface-formation trajectory is implied.
- The earlier independent-target design-grid mean pass **failed fixed-command surface refinement**: front/back errors **10440 / 4812 nm**. It is preserved historical evidence, not the current prescription.
- [Physical feasibility and material selection](docs/material-feasibility.md): real-fluid requirements and experimental precedents.
- [Earlier single-interface evidence](docs/single-interface-history.md): separate model and preserved outputs.
- [Fresh formation notebook](artifacts/notebooks/05_formation_and_spots.html): 3D setup, 2D physical-time surface/error animation and evolving fixed-plane spot diagrams for the earlier single-face model.
- [Architecture](docs/architecture.md) and [documentation index](docs/README.md).
- [Evidence catalog](artifacts/catalog-2026-09-23-cycle-mean/report.md): inventory snapshot; completion is stated in each report.

## Run and verify

```bash
uv sync --dev
uv run pytest -q
uv run ruff check src tests tools
python3 tools/build_theory.py
uv run python -m acoustic_freeform.dual configs/dual/cartesian-pair-frequency.json --out artifacts/my-two-face-study
```

Completed experiment directories cannot be overwritten. Every scientific run
must retain configuration, executed-source provenance, validation and a report.
An inverse residual is not a forward surface error. Newton iterations are not
a physical formation trajectory.

## Repository map

| Location | Responsibility |
|---|---|
| `docs/theory/` | Canonical general theory and primary bibliography |
| `src/acoustic_freeform/dual/` | Two-interface stationary/inverse model, restricted inertial dynamics and ideal-load viscous formation |
| `src/acoustic_freeform/lens/` | Earlier single-interface dynamics and optics |
| `src/acoustic_freeform/verification/` | Independent verification and historical planar model |
| `configs/`, `tests/` | Reproducible inputs, analytic limits and numerical checks |
| `artifacts/` | Versioned evidence, never unlabelled loose results |
| `references/` | Reading map, acquisition manifest and private paper/discussion library |
| `notebooks/` | Source notebooks; executions belong under artifacts |
| `tools/`, `web/` | Research utilities and the earlier physical-time viewer |

SI units are used in calculations and files. Exploratory materials and source
ports remain hypothetical until calibrated. Streaming, thermal feedback,
three-dimensional stability, formation and curing require separate validation.

No publication or remote is authorized. Research milestones are preserved in
`/Users/blancus/Private/research-archives/`; benchmark material belongs exclusively
outside this repository in the benchmark archive.
