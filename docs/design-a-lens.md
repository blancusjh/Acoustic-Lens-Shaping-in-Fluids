# Design a lens on the machine

`afl design` takes a desired stigmatic singlet (SOSL) and returns the settings
of the fixed machine that form it: pin heights, fill volumes and the drive
command for every source channel. It runs in the same numerical model as
study [S09](../studies/S09-universal-source/study.md) and states plainly what
it established and what it did not.

This is a design tool for a numerical model. Nothing here has been built or
measured, and a predicted nanometre is not a manufactured one.

## Quick start

```sh
uv sync --dev
# Fig. 9b of Silva-Lora and Torres (2020), scaled 1/14 to a 4 mm aperture
uv run afl design --object collimated --image 90 --thickness 10 \
    --intermediate 137.95 --scale 1/14 --name "Fig. 9b" --out artifacts/designs/fig9b
```

From Python:

```python
from acoustic_freeform.design import Singlet, design

lens = Singlet(object_m="-inf", image_m=0.09, thickness_m=0.01,
               intermediate_m=0.13795, scale=1 / 14, name="Fig. 9b")
recipe = design(lens)                 # or design(lens, startup=False) for a quick answer
print(recipe.status)
recipe.save("artifacts/designs/fig9b")
```

## The lens

Distances follow Silva-Lora and Torres, JOSA A 37, 1155 (2020): along the axis
from the front lens vertex, multiplied by `scale`.

| Field | Meaning |
|---|---|
| `object_m` | Object point, negative below the lens; `-inf` (CLI: `collimated`) for light from infinity |
| `image_m` | Image point, positive above the lens |
| `thickness_m` | Vertex-to-vertex thickness |
| `shape_factor` or `intermediate_m` | Second design freedom: the paper's shape factor, or its intermediate stigmatic point |
| `indices` | Object-side, lens and image-side refractive indices for the optical design; default 1 / 1.7 / 1 |
| `scale` | Geometric scale applied to the three distances (e.g. 1/14 maps a 56 mm aperture to 4 mm) |

The CLI takes the distances in millimetres by default (`--units m` for metres).

## The machine

[`configs/machine/v1.json`](../configs/machine/v1.json) declares the fixed hardware once:

- cell: bore radius 2.02 mm, clear aperture 4 mm, windows at z = −0.8 and +2.2286 mm;
- liquids: water / NOA 61 / water ([material record](../configs/shared/noa61-water-pressure-reference.json));
- sources: 32 ring regions per window and 16 wall bands per layer (112 channels),
  driven through the same 26 continuous modes;
- drive options, tried in order: one tone at 1.5 MHz, then one tone each at 0.75 and 1.5 MHz;
- limits: 1 W source power and 2 MPa peak pressure per band.

Only three things follow the lens: the heights of the two pinning edges, the
three fill volumes and the drive command. A different machine file (other
liquids, sources or limits) is passed with `--machine`.

## What `design` does

1. **Fit.** Places the lens with its front vertex 0.8 mm above the bottom
   window and checks that both faces clear both windows and do not cross.
2. **Pins and fills.** The pinning edges sit at the lens rim heights; each
   layer is filled to the volume that the target faces enclose.
3. **Certificate.** The convex capability program (theory section 10f) gives
   the best pupil error any drive of that option can reach within the limits.
   If it is above the goal, no command of this machine can hold the lens.
4. **Command.** Refines the certificate to exactly one tone per band. If the
   one-tone error is above 1 nm, the next drive option is tried.
5. **Start-up.** Ramps the command from zero with the liquid at rest and
   follows the coupled equilibrium to full amplitude (skipped with
   `--no-startup`).

## Status

| Status | Meaning |
|---|---|
| `formed` | The ramp from rest ends within 10 nm of both target faces. Not yet independently verified. |
| `command-only` | A command holds the lens in the frozen model; the start-up ramp was not run. |
| `start-up-fails` | The command holds the lens, but the ramp from rest ends on another equilibrium (S09 lens 57). |
| `needs-more-source` | Reachable in principle, but no drive option reaches the goal with one tone per band. |
| `not-reachable` | The certificate shows that no command of this machine holds the lens within the goal. |
| `does-not-fit` | The lens does not fit the cell. |

## The recipe

`recipe.json`, with SI units throughout:

| Field | Meaning |
|---|---|
| `settings.pin_heights_above_bottom_window_m` | Heights of the two pinning edges |
| `settings.fill_volumes_m3` | Volume of the lower, lens and upper layers |
| `settings.drive.bands_hz` | Drive frequencies (one tone each) |
| `settings.drive.channel_commands_file` | `design/source-commands.csv`: complex velocity command, amplitude and phase per channel (and band) |
| `settings.drive.source_power_w`, `band_peak_pressure_pa` | Power and sampled peak pressure at the formed state |
| `prediction.formed_pupil_error_m` | Max height error of each face over the clear aperture at the end of the ramp |
| `prediction.frozen_pupil_error_m`, `convex_bound_m` | First-order error of the command and the certificate |
| `prediction.fold_margin_min` | Smallest static stability margin along the ramp (1 = no acoustic softening, 0 = fold) |
| `prediction.spot_*`, `airy_radius_*` | Geometric spot on the design image plane, with the published indices, and the Airy radius |
| `attempts` | Every drive option tried, with its certificate, error and outcome |

The saved `design/` folder uses the held-command layout of the research steps,
so the independent checks run on it unchanged.

## Acceptance: the independent checks

`formed` is a prediction of the start-up ramp alone. Before a design is
treated as verified in this model, run the three checks used for every S09
design (copy and adapt the S09 configurations, pointing them at the saved
design):

```sh
uv run afl force-convergence --out ... configs/S09-universal-source/force-convergence-fig9b.json
uv run afl coupled-stability --out ... configs/S09-universal-source/coupled-stability-fig9b.json
uv run afl form --out ...             configs/S09-universal-source/formation-fig9b-dt0025.json
```

They refine the mesh and order at the fixed command, test linear stability
with the full creeping flow, and simulate the formation in time at two steps.

## Run time

Most of the time goes into the acoustic kernels of the 26 source modes at each
frequency, and into the start-up ramp. Measured on a 4-core machine that was
also running two or three other solver jobs:

| Run | Wall time |
|---|---:|
| Lens 7, `--no-startup` (one tone tried, then two) | 17 min |
| Fig. 9b, `--no-startup` (slow regression test) | 13 min |
| Fig. 9b, full run with the start-up ramp (status `formed`) | 71 min |

It has not been timed on an idle machine; the S09 atlas runs, which include the
ramp, took 3 to 30 minutes per lens there. The independent checks take hours.

## Limits

The model is axisymmetric, with lossless acoustics, ideal velocity sources,
quiescent liquids and creeping flow. Acoustic streaming, heating,
non-axisymmetric disturbances, contact-line hysteresis, the transducers that
would realize the source modes, and curing are not modelled. The NOA 61 sound
speed is a placeholder, and a 0.1 % change in a sound speed moves the faces by
tens to hundreds of nanometres, so a real machine needs measurement and trim.
