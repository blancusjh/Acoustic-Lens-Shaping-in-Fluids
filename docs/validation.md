# Numerical evidence

The reference result is `artifacts/asphere/`. Numerical verification establishes
consistency of the implemented model; it does not validate the unmeasured resin
acoustics or omitted physical mechanisms.

## A lens with an optical purpose

The computed surface fits a conic with vertex radius **10.400011 mm**,
**K = −2.3102715**, and a **1.98 nm RMS** residual over the 6 mm clear pupil.
Its departure from a best-fit sphere is **1.43216 µm RMS**, **4.84562 µm peak to valley**.
The aspheric departure is much larger than the numerical differences found below.

The unforced liquid has a 192.36 µm geometric RMS ray radius at its own best focus.
The optical best-fit sphere gives 47.59 µm even after refocusing. The driven liquid
gives 0.210 µm at a focal distance of 19.99996 mm from its vertex. Thus the result
is distinguished by optical correction, rather than by an unusual-looking surface.

The same model gives 462.46 µm RMS for the unforced liquid at the fixed target plane.
The controller does not move the detector to hide focus error during playback.

## Independent analytic checks

The automated finite-lens tests compare a zero-gravity, fixed-volume cap against
an exact sphere; trace an equal-optical-path Cartesian oval; and check volume,
unforced viscous dissipation, incompressibility and inverse-viscosity time scaling.

An independent Fourier–Bessel series solves the flat cylindrical acoustic cavity
with the same tapered complex wall drive, rigid base and pressure-release top.
For the verification configuration, measured FEM errors are:

| Quantity | Relative L2 error |
|---|---:|
| Complex acoustic pressure | 2.11 × 10⁻⁶ |
| Surface radiation stress | 8.42 × 10⁻⁵ |
| Linear-system residual | 4.14 × 10⁻¹³ |

These results test boundary conditions and force conventions in a case with an
independent solution. Small matrix residuals alone would not establish accuracy.

## Fixed-drive spatial refinement

The final drive is frozen. Each row below re-solves nonlinear static capillarity
and acoustics on the changing trial surface. There is **no controller refit**.
The recorded trajectory remains the reference for surface differences.

| Acoustic basis; radial × vertical cells; surface modes | RMS surface change | RMS ray radius at fixed target |
|---|---:|---:|
| P3; 64 × 96; 32 | 0.009 nm | 0.20979 µm |
| P4; 48 × 64; 32 | 4.070 nm | 0.22251 µm |
| P4; 64 × 96; 32 | 1.944 nm | 0.21297 µm |
| P4; 80 × 128; 32 | 1.765 nm | 0.21276 µm |
| P4; 80 × 128; 48 | 1.765 nm | 0.21276 µm |

The coupled stationary solves converged. The higher surface-mode count preserves
the result, addressing the possibility that the array merely satisfies a short
list of projected forces while hiding an uncontrolled optical ripple.

A separate fluid-mesh check at `t = 0.7 s` holds the shape and acoustic traction
fixed. Relative to a 96 × 144 Stokes mesh, the 32 × 48 and 64 × 96 meshes differ in
pupil height rate by **0.269%** and **0.0702%**, respectively. The raw values are in
`verification/stokes-convergence.json`. This checks the moving-fluid discretization
separately from the static optical result.

Earlier quadratic acoustic elements gave an overly precise impression for a
coarser mesh; that exploratory output is archived separately. The active experiment
uses cubic elements and the refinement above. No optical accuracy is inferred from
the coarse exploratory candidate.

## Time and rendering evidence

`trajectory.npz` contains fluid states from the Stokes integrator. The actual
trajectory conserves volume to roundoff and reaches the held state without replacing
the computed surface by the design target. The smaller-time-step, open-loop replay
is stored in `verification/time-refined-replay/`. Halving the step from 20 ms to
10 ms, with the same recorded sample-and-hold drive and no controller refit, changes
the pupil surface by at most **0.420 µm RMS** during the ramp. The final surface
differs by **0.0027 nm RMS** and retains a **0.210 µm** geometric ray radius.
The maximum transient pointwise difference is 1.411 µm. These are two-step
comparisons, not an extrapolated error bound or a measured order of convergence.
The raw comparison is in `verification/time-convergence.json`.

`verification/browser.json` records browser checks of scrubbing, changing surface
coordinates and acoustic data, playback and camera/layer controls. The viewer
displays acoustic-cycle envelopes at the stored fluid times, not individual MHz
oscillations. The pressure slice and flow vectors come from solved fields. The
geometry is shown at unit scale, and field-arrow scaling is labelled.

The default camera includes a cutaway of the housing and array. The liquid remains
a filled volume, with its free optical surface highlighted. A separate view shows
that surface alone, and another shows the actual refracted rays.
