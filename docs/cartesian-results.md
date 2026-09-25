# Finite-conjugate Cartesian diopter: research result

This page preserves the earlier preliminary result. The current
[convergence and stability investigation](numerical-stability.md) corrects its
geometry/discretization and records new models, excitations and dynamic checks.

The exact optical construction is verified. A reproducible acoustic stationary
candidate has been found, but its optical precision is not spatially converged
and the nominal fixed-drive model predicts one growing mode. It must not be
described as a validated, stable stigmatic device.

The primary optical source is Silva-Lora and Torres, *Superconical aplanatic ovoid
singlet lenses*, JOSA A 37, 1155–1165 (2020),
[doi:10.1364/JOSAA.392795](https://doi.org/10.1364/JOSAA.392795).
The [model note](cartesian-diopters.md) gives the equations, signs and acoustic
inverse problem. This page reports our calculations, not measurements or results
claimed by that paper.

## Optical target and physical perturbation

The example uses a real object and real image:

| Input or result | Value |
|---|---:|
| Incident liquid index, `n_o` | 1.52 |
| Object coordinate from target vertex, `z_o` | −50 mm |
| Transmitted air index, `n_i` | 1.00 |
| Image coordinate from target vertex, `z_i` | +20 mm |
| Clear aperture diameter | 6 mm |
| Chamber diameter / depth | 8 mm / 6 mm |
| Target vertex above the rim | 1.197786 mm |
| Unforced vertex, same fill | 1.045703 mm |
| Required central perturbation | +152.083 µm |
| Liquid volume | 331.389 µL |

The paper's parametric formula agrees with the independently continued unsquared
Fermat equation to approximately 10⁻¹⁸ m for this case. Independent vector-Snell
rays converge to the specified image to floating-point precision. These checks
verify the optical target; they say nothing about acoustic reachability.

The target extends to the 4 mm mounting radius and meets the fixed rim. Changing
the conjugates generally changes the required fill. The incident wavefront is
prescribed **inside the resin**; illumination corrected for the bottom window is
not yet designed. The air boundary is part of the current one-liquid model.

## What the forward and inverse computations actually produced

All ray errors below are geometric, area-weighted RMS radii at the **same specified
image plane**, with fixed incident-wavefront position. They are not diffraction
calculations, experimental measurements, or errors at separately chosen best foci.

| Calculation | Frequency | Geometric ray RMS | Interpretation |
|---|---:|---:|---|
| Restricted-drive physical transient, final time 1.8 s | 1.35 MHz | 48.402 µm | Settled in the integrated model; misses the target |
| P3 joint stationary candidate, 64 × 96 mesh, 32 surface modes | 1.8 MHz | 0.392 µm | Superseded nominal result |
| That P3 drive, re-solved with P4 acoustics | 1.8 MHz | 16.023 µm | Invalidates the P3 optical precision |
| Redesigned P4 candidate, 64 × 96 mesh, 32 modes | 1.8 MHz | 0.389 µm | Current nominal candidate |
| Same P4 drive, 80 × 128 mesh, 48 modes | 1.8 MHz | 8.036 µm | Still not converged at submicrometre ray precision |

The last refinement changes the clear-aperture surface by 0.485 µm RMS. Its optical
path RMS changes from 1.51 nm to 251.76 nm. Re-optimizing on every mesh would hide
this test: the check deliberately retains the identical physical drive.

The restricted transient is a real Stokes/interface time integration with a
recorded drive program. At its final time the computed slow fluid velocity is
about 1.24 × 10⁻⁹ m/s. The stationary optimization has **no physical approach
trajectory**. A visually accurate cap or a small stationary residual cannot
substitute for optical convergence and stability.

## Excitation that can be inspected and reproduced

The current candidate's sixteen complex row drives are in
[`hold-drive.csv`](../artifacts/studies/S01-single-interface/cartesian-50-20-stationary-p4/hold-drive.csv).
All sixteen azimuthal sectors of each row share its drive. Its frequency is
**1.8 MHz**, maximum peak wall speed **1.02962 m/s**, and maximum peak displacement
**91.038 nm**. The nominal model predicts approximately **4.03 MPa** peak cavity
pressure and **0.182 W** supplied acoustic power. These are simulated mechanical
quantities, not electrical settings or calibrated hardware specifications.

For row j, the convention is

\[
v_{n,j}(z,t)=e_j(z)|w_j|\cos(2\pi f t-\arg w_j),
\]

with the documented cosine envelope and outward liquid-wall normal. The CSV
contains real/imaginary velocity, peak amplitude, phase, relative phase and peak
displacement. A measured loaded transducer transfer function is needed to convert
these to voltages. A global phase rotation leaves the mean force unchanged.

```bash
uv run lenslab stationary configs/lenses/noa61_cartesian_50_20_stationary.toml \
  --seed configs/initialization/cartesian_50_20.json --starts 1 \
  --out artifacts/reproduced_cartesian --stability
uv run lenslab verify artifacts/reproduced_cartesian
uv run lenslab render artifacts/reproduced_cartesian
```

The initial guess selects the recorded optimization branch; it is not a substituted
final answer. The provisional speed envelope is 1.28 m/s. The optimizer uses a
conservative real/imaginary box inside the circular amplitude bound, so failed
searches do not establish mathematical infeasibility of the full envelope.

## Stability and remaining numerical questions

The nominal P4 continuous Stokes/interface linearization has one positive real
growth rate, about **37.5 s⁻¹**, with the row drives fixed. It includes the acoustic
force's shape derivative. Thus force balance alone does not demonstrate that the
candidate would remain there after a perturbation. This stability calculation has
not itself been repeated on the finer mesh, and inertia is absent from this model.

A geometry diagnostic found that the quadratic FEM boundary differs from the
modal surface by at most about 0.252 µm over the whole cap, with the largest error
near the rim; the maximum discrepancy inside the clear aperture is only 0.044 nm.
The corresponding maximum normal-vector discrepancy is about 0.0258 at the rim.
This is a concrete reason to investigate the boundary representation and rim
resolution. It **does not establish the cause** of the optical refinement error.
Acoustic field error, geometry error and surface truncation still need to be
separated. Dense and Gauss pupil sampling agree closely and do not explain the
large refinement change.

The next numerical experiment should hold the same drive and surface fixed while
independently improving boundary geometry and acoustic resolution, before fitting
another drive. After force/optics convergence, either seek a stable branch or
design and time-integrate a feedback controller with declared sensing and actuation
bandwidth. This is needed before calling the excitation a maintained stigmatic
diopter. Streaming, heating, acoustic material calibration and a physical
transducer model remain separate experimental-model questions.

## Evidence and organization

- [Canonical notebook](../notebooks/02_cartesian_diopters.ipynb) and
  [executed HTML](../artifacts/studies/S01-single-interface/notebooks/02_cartesian_diopters.html): equations,
  optical family, required perturbations, drives, actual rays and refinement.
- [Restricted transient viewer](../artifacts/studies/S01-single-interface/cartesian-50-20/viewer/index.html):
  actual time evolution of the full apparatus and computed fields.
- [Stationary PyVista scene](../artifacts/studies/S01-single-interface/cartesian-50-20-stationary-p4/stationary-viewer.html)
  and [optical scene](../artifacts/studies/S01-single-interface/cartesian-50-20-stationary-p4/optics-viewer.html):
  nominal candidate, explicitly labeled with stability and refinement limitations.
- [Report](../artifacts/studies/S01-single-interface/cartesian-50-20-stationary-p4/report.json) and
  [fixed-drive refinement](../artifacts/studies/S01-single-interface/cartesian-50-20-stationary-p4/spatial-convergence.json):
  raw numerical evidence, coefficients and source hashes.

Superseded P3 candidates and design-screening records are retained with a
checksum manifest under the separate private research archive
`/Users/blancus/Private/research-archives/acoustic-freeform-lab/cartesian-diopters-2026-09-06/`.
They are not inputs to the active viewer or notebook. The initialization JSON
preserves the specific prior drive used to reproduce the current candidate.
The original collimated-asphere trajectory and benchmark/task repository were not
modified by this extension.
