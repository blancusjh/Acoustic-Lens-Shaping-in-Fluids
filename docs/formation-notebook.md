# Formation and optical-spot notebook

## New: both interfaces forming in time

[Notebook 06 — executed](../artifacts/studies/S02-independent-two-face/notebooks/06_two_face_formation.executed.ipynb),
[browser notebook](../artifacts/studies/S02-independent-two-face/notebooks/06_two_face_formation.html),
[interactive viewer](../artifacts/studies/S02-independent-two-face/notebooks/06_two_face_formation/viewer.html).

This new, separate notebook shows an actual **two-interface forward trajectory**
over 0–8 ms, both pupil errors, rays traced through both interfaces, and a 3D
cylinder with all 112 declared annular/end and sidewall source regions.
It uses a newly derived axisymmetric inviscid, flat-domain potential-inertia
reduction with 16 dynamic modes, nonlinear capillarity and a fresh moving-graph
acoustic solve at each predicted midpoint. This is **not** the viscous single-face
solver below, the 448-source stationary experiment, or a qualified fluid triplet.

Two completed fixed-command runs use 0.25 and 0.125 ms steps. Dynamic-mode and
acoustic-mesh convergence are not established. Longer runs exceeded the 300 µm
displacement guard before finishing the 10 ms source ramp; the notebook does not
manufacture settling. Error maxima are sampled, not certified. No 10 nm pass is
claimed. No sound from the different single-face system is reused here.

Run inputs, executable-source snapshots, accepted SI trajectories and validation
are preserved under `artifacts/studies/S02-independent-two-face/dual-formation-2026-09-23/`. The viewer's validation
includes the time-step discrepancy. The canonical theory contains the derivation
in “A two-interface inertial formation diagnostic.”

## Earlier notebook 05: single-face formation and stationary two-face setup

This presentation reruns the **single-interface** model from the September 6
formation viewer. It does not add dynamics to the newer two-interface solver.

Section 0 now shows the **two-face setup** as well: both bottom/front and
top/back interfaces, a 3D camera-rotation view, and a 2D profile/error comparison.
The selector chooses saved 52-, 80- and 104-element stationary solutions with
unchanged commands. It is not a physical-time animation. The later single-face
formation, spots and audio must not be attributed to this different apparatus.
[Open the two-face setup alone](../artifacts/studies/S01-single-interface/notebooks/05_formation_and_spots/two-face-setup/setup.html).

- [Browser notebook](../artifacts/studies/S01-single-interface/notebooks/05_formation_and_spots.html)
- [Executed Jupyter notebook](../artifacts/studies/S01-single-interface/notebooks/05_formation_and_spots.executed.ipynb)
- [Source notebook](../notebooks/05_formation_and_spots.ipynb)
- [Standalone synchronized animation](../artifacts/studies/S01-single-interface/notebooks/05_formation_and_spots/formation-and-spots.html)

## What is computed

Two fresh fixed-command runs start from the unforced equilibrium with zero
mean fluid velocity and evolve to 0.2 s. Time steps are 0.00125 and 0.000625 s.
The archived operating-state commands are not re-optimized. The zero-gain
holding artifact supplies the implicit radiation Jacobian used by the existing
integrator; it does not apply feedback corrections. Actual acoustic force and
bulk forcing are recalculated on the evolving geometry.

The notebook uses the finer trajectory for surface/error and spot animation.
Every displayed frame is a saved physical state. The detector plane and incoming
wavefront stay fixed; the spot diagram is not refocused at each frame. Ray
intercepts are geometric optics on the cycle-mean surface, not a diffraction
image or a model of pressure/temperature-dependent refractive index.

The three-dimensional view includes only the configured wall, base, computed
surface and ideal inner-wall source supports. The visual cutaway does not
change the sealed chamber. It is not a manufactured transducer design.

## Sound: diagnostic sonification, not a recording

The audio player synchronizes the displayed saved frame to its playback clock.
Use its Play/seek controls for sound; the original animation controls stop the
audio and operate silently. Sound does not start automatically.

The model's 1 MHz carrier is ultrasonic. No microphone signal, air-radiation
model, structural audible-noise model or acoustic phase history is available.
The added audio is an artificial 440 Hz tone whose envelope follows saved
spatial peak cavity-pressure amplitudes. Physical time is slowed 88 times;
volume is normalized to 12% of digital full scale, with short playback fades.
This is not phase-preserving frequency shifting or a calibrated acoustic level.
The waveform, mapping configuration and validation are in the presentation's
`sonification/` directory. Start with low playback volume.

## Inspect and reproduce

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python tools/execute_notebooks.py \
  --timeout 900 notebooks/05_formation_and_spots.ipynb
uv run python tools/verify_formation_notebook.py
uv run pytest -q tests/lens
```

The notebook reads completed solver outputs. If absent, it launches the two
forward runs; if it finds a partial/active directory, it stops rather than
overwriting or launching a duplicate. Archive any existing notebook milestone
before replacing its executed copy or presentation assets. All original
September 6 solver outputs remain unchanged.

Fresh solver outputs are in `artifacts/studies/S01-single-interface/formation-notebook-2026-09-23/`.
The presentation directory contains SI analysis arrays, input hashes, source
provenance, static snapshots, browser checks and a validation report.
The older solver's dimensionless surface coefficients are explicitly converted
to metres when reconstructed. No interpolation between surface states is used.

## Interpretation

Two time steps demonstrate sensitivity, not a certified convergence order or
precise settling time. Reproduction of an older trajectory does not validate
its material assumptions. Maximum surface error and RMS geometric spot radius
are distinct quantities; neither implies the other satisfies a 10 nm bound.
The [wall-loss and stability limitations](numerical-stability.md) still apply.
No thermal feedback, wall-layer mean streaming, curing, calibrated hardware or
full three-dimensional stability is established by this visualization.
