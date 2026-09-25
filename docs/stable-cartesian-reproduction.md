# Reproducing the maintained Cartesian diopter

Run from the private repository root. All numerical values use SI units in
configuration and result files. Choose unused destination directories: completed
results are protected from replacement. The commands below do not launch external
evaluations or publish anything.

## Inspect the recorded experiment

```bash
uv sync --dev
npm --prefix web ci
npm --prefix web run build
uv run lenslab view artifacts/studies/S01-single-interface/cartesian-maintenance-2026-09-06/step-formation-dt00125
uv run python tools/execute_notebooks.py notebooks/03_stable_cartesian.ipynb
```

The viewer starts at its last computed frame, 0.2 s. Restart/play shows formation
from the unforced shape with the final drive applied from time zero. Each row's
complex drive is identical in every frame. The instantaneous acoustic envelope
approximation does not resolve the initial acoustic ring-up.

`operating-state/hold-drive.csv` specifies peak wall velocities and phases.
`excitation-definition.json` defines the cosine taper, wall-normal orientation
and `exp(-i*2*pi*f*t)` convention. The full saved schedule and radial surface data
are under `step-formation-dt00125/excitation/`. A CSV comparing radiation with
capillary/gravity traction is a **partial** stress balance when circulation is
present: it does not include spatially varying mean-flow stress.

## Rebuild the stationary design and resolution study

The configuration contains the optical conjugates and apparatus. The seed
selects the recorded optimization branch; it does not prescribe the evolving
surface. Changing conjugates can require a new fill volume and a new feasible
array solution. The local nonconvex optimizer is not guaranteed to find every
reachable target or a global optimum.

```bash
uv run lenslab stationary configs/lenses/noa61_cartesian_50_20_viscous_1mhz.toml \
  --seed configs/initialization/cartesian_wall_1mhz.json --starts 1 \
  --out artifacts/reproduction/design

uv run lenslab verify-coupled artifacts/reproduction/design \
  --out artifacts/reproduction/convergence --method broyden1 \
  --level 32 48 32 --level 48 64 40 --level 64 96 48

uv run lenslab regrid-stationary artifacts/reproduction/design \
  --resolution 32 48 32 --out artifacts/reproduction/operating-state

uv run lenslab acoustic-budget artifacts/reproduction/operating-state \
  --out artifacts/reproduction/acoustic-budget
```

The first command solves a stationary inverse problem. Its iteration numbers
are not physical times. The next two commands preserve physical row velocities
and volume; neither refits the array. The original design used P4 acoustics on a
64 × 96 base mesh with 48 surface modes; the verified working model uses
32 × 48 and 32 modes. Patch endpoints add axial mesh nodes automatically.

## Form the surface and disturb it

```bash
uv run lenslab control-model artifacts/reproduction/operating-state \
  --out artifacts/reproduction/control-model
uv run lenslab feedback artifacts/reproduction/operating-state \
  --stability artifacts/reproduction/control-model \
  --out artifacts/reproduction/hold-model

uv run lenslab evolve artifacts/reproduction/operating-state \
  --controller artifacts/reproduction/hold-model --initial rest_fixed \
  --step 0.00125 --duration 0.2 --out artifacts/reproduction/formation

uv run lenslab evolve artifacts/reproduction/operating-state \
  --controller artifacts/reproduction/hold-model --initial perturbation \
  --perturbation=-0.000005 --mode 6 --step 0.00125 --duration 0.125 \
  --out artifacts/reproduction/recovery

uv run lenslab excitation artifacts/reproduction/formation
uv run lenslab render artifacts/reproduction/formation
uv run python tools/verify_viewer.py artifacts/reproduction/formation
```

For the recorded case the control surrogate has no growing mode, so `feedback`
exports **zero gain** and declares `mode = fixed_drive`. Verify that mode in its
report when reproducing a different case. This artifact also supplies the frozen
radiation Jacobian used in the consistent first-order implicit time update;
it does not replace the actual acoustic force, fluid state or physical drive.
The surrogate omits derivatives of bulk forcing and base circulation. Its
eigenvalues cannot certify the complete streaming plant.

Initial conditions are distinct:

- `rest_fixed`: unforced equilibrium geometry, zero mean velocity, final drive
  switched on and then held fixed.
- `perturbation`: displaced holding geometry, zero mean velocity; `--perturbation`
  specifies signed full-aperture RMS amplitude in metres and `--mode` selects
  a volume-null basis direction.
- `hold`: holding geometry with the stationary Stokes circulation as initial
  velocity, subsequently evolved with the nonlinear flow equations.
- `rest`: an ideal observed-surface inverse controller updates the approach
  drive every 20 ms during `--ramp`, then uses the holding drive/controller.
  This is a separate controlled-formation experiment.

Time steps used for direct formation were 0.005, 0.0025, 0.00125 and 0.000625 s.
The last ran to 0.125 s and the others to 0.2 s. A separate 0.0025 s trajectory
used `--resolution 48 64 40` and ran to 0.15 s. Surface perturbations retain the
same physical shape when refinement adds modes. A stable endpoint does not
remove early-trajectory time-discretization error; consult the complete curves.

## Campaign artifact index

All paths in this table are relative to
`artifacts/studies/S01-single-interface/cartesian-maintenance-2026-09-06/`.

| Path | Meaning |
|---|---|
| `viscous-1mhz/` | Coupled stationary inverse design including wall damping and bulk flow |
| `viscous-1mhz/convergence/` | Three independent stationary solves with the same physical drives |
| `operating-state/` | The same drive transferred to the verified working resolution |
| `operating-state/control-model/` | Radiation/quiescent-fluid surrogate; explicit omissions in report |
| `operating-state/hold-model/` | Zero-gain holding artifact and numerical radiation Jacobian source |
| `operating-state/acoustic-budget/` | Source/bulk/wall energy, circulation and fast-surface diagnostics |
| `step-formation*/` | Fixed-drive formation; time-step and spatial variants |
| `fixed-drive-recovery/` | +1 µm volume-null mode 0 disturbance, 5 ms step |
| `recovery-negative-mode6/` | −5 µm volume-null mode 6 disturbance, 1.25 ms step |
| `formation/` | Ideal inverse-controlled 0.6 s approach, then fixed hold to 1 s |
| `verification/` | Cross-run audit, convergence arrays and PNG/PDF scientific figure |
| `wall-frequency-screen/` | Frozen-target screening, before coupled verification |
| `recovery-dt005/`, `recovery-dt0025/` | Earlier 1.8 MHz feedback/time-method studies, before wall damping |

`tools/summarize_maintenance.py` reads completed files, checks unchanged drives
and volume, compares matching physical times, and creates the campaign audit.
It does not simulate, optimize or alter state arrays:

```bash
uv run python tools/summarize_maintenance.py artifacts/studies/S01-single-interface/cartesian-maintenance-2026-09-06
uv run ruff check src tools
npm --prefix web run check
```

The current executions contain `provenance/execution-sources.zip` and hashes
captured before iteration. Artifact hashes allow exact local identification;
they are not evidence of experimental truth. `configuration.json` is the resolved
input for each execution. The current numerical results and unresolved physics
are discussed in [the research report](numerical-stability.md).
