# Cartesian diopter: convergence, inertia and maintained operation

This numerical investigation follows the [general formulation](theory/README.md).
The optical example is the same real object/image pair: `n_o=1.52`, `z_o=-50 mm`,
`n_i=1`, `z_i=+20 mm`, with a 6 mm clear aperture in an 8 mm diameter chamber.
All conjugate positions refer to the target vertex. Illumination is prescribed
inside the liquid; an external source and the bottom window's refraction remain
separate optical design problems.

The objective is a maintained Cartesian diopter and a physical trajectory that
reaches it. A small stationary optical error, mesh convergence, stability and
experimental feasibility are separate requirements. The original stationary
candidate and all earlier scientific artifacts are preserved.

## Current result: a fixed-drive Cartesian surface at 1 MHz

The new candidate forms from the unforced liquid and remains near the intended
Cartesian surface with **all 16 complex row drives held constant**. The forward
calculation includes fluid inertia, convection, gravity, capillarity, distributed
bulk absorption forcing and leading viscous acoustic wall losses. It does not
insert the target as a physical state or change the drive to repair an error.
This is finite-horizon numerical evidence in an axisymmetric, isothermal model,
not experimental validation or a proof for every possible disturbance.

Open the [formation viewer](../artifacts/cartesian-maintenance-2026-09-06/step-formation-dt00125/viewer/index.html).
It starts at the **last computed state, t = 0.2 s**. Restart/play shows the actual
formation trajectory, beginning with the unforced shape and the final acoustic
drive switched on. The housing, base window, liquid cap, rim and row array are
declared apparatus components. Acoustic pressure, mean flow and optical rays
change with the recorded states. The visible 256 sectors have only **16 independent
axisymmetric row excitations**.

| Quantity | Current numerical result |
|---|---:|
| Clear optical diameter / chamber diameter | 6 / 8 mm |
| Liquid volume | 331.389 µL |
| Target vertex above rim / unforced vertex | 1.197786 / 1.045703 mm |
| Target vertex perturbation | +152.083 µm |
| Geometric ray RMS, fixed requested image plane | about 0.05 µm |
| OPD RMS after removal of piston | about 0.25 nm |
| Surface departure from best-fit sphere, peak to valley | 12.040 µm |
| Best-fit sphere's geometric ray RMS at its own best focus | 121.729 µm |
| Largest holding wall-speed amplitude | 0.98927 m/s |
| Corresponding wall-displacement amplitude at 1 MHz | 157.45 nm |
| Peak holding acoustic pressure | 1.7704 MPa |
| Net acoustic source power at hold | 0.26126 W |
| Maximum held mean-liquid speed | 0.2134 mm/s |

Ray RMS is a geometric error radius for the **mean surface** in the declared
constant-index material model. It is not a measured or diffraction-limited spot
size. Height RMS to the ideal target is approximately 0.226 µm, mostly an almost
constant axial offset; its derivative and optical error are much smaller.
Report fields returned by `best_fit_sphere` describe that fitted sphere; actual
surface vertex height is separately recorded in the stationary report.

### Independent spatial and temporal checks

The following equilibrium solves keep the **same physical complex drives and
liquid volume**. No row is re-optimized during refinement. Acoustic order is P4;
the listed axial divisions additionally include source-patch endpoints.

| Radial × axial divisions / surface modes | Geometric ray RMS | OPD RMS |
|---|---:|---:|
| 32 × 48 / 32 | 0.05410 µm | 0.2532 nm |
| 48 × 64 / 40 | 0.04995 µm | 0.2465 nm |
| 64 × 96 / 48 | 0.04956 µm | 0.2486 nm |

These results support a useful working resolution of 32 × 48 with 32 surface
modes. The equilibrium correction tolerance is 0.1 nm; smaller differences
should not be interpreted as resolved physical accuracy.

The fixed-drive formation trajectory was repeated with 5, 2.5 and 1.25 ms time
steps over 0.2 s, and with a 0.625 ms step over 0.125 s. Their terminal geometric
ray RMS is approximately 0.05411 µm. Early trajectories remain visibly sensitive
to the first-order time discretization. **No precise settling time is claimed.**
At a common 2.5 ms step, refining the trajectory to 48 × 64 / 40 changes pupil
height by at most 37.0 nm during formation and 0.164 nm at 0.15 s. Its terminal
ray RMS is approximately 0.0500 µm.

The recovery cases use volume-preserving modal disturbances and zero initial
mean velocity. They complement formation from a very different initial surface;
they do not establish a basin of attraction for arbitrary shapes. A +1 µm
full-aperture RMS disturbance in volume-null mode 0 and a −5 µm disturbance in
mode 6 both return to approximately 0.05411 µm geometric ray RMS without any
drive update. Their remaining pupil height offset from the saved stationary
state is about 0.071 nm, below its 0.1 nm equilibrium correction tolerance. The
[machine-readable audit and comparison figure](../artifacts/cartesian-maintenance-2026-09-06/verification/maintenance.json)
record each disturbance, time step, conserved volume, unchanged drive, final
optics and the complete time-refinement differences.

### Why the frequency and wall model changed

Applying the thin viscous-layer dissipation diagnostic to the old 1.8 MHz field
gave approximately 0.807 W at the walls, compared with 0.193 W of bulk absorption.
That perturbative estimate invalidates treating those walls as acoustically
lossless at the desired precision. It must not simply be added to the old
power total: the acoustic field changes when the wall condition changes.

With angular frequency `omega`, liquid density `rho` and Newtonian viscosity
`mu`, the viscous penetration depth is `delta = sqrt(2*mu/(rho*omega))`.
For a wall with prescribed normal velocity `V_wall`, no tangential drive, peak
pressure phasor `P` and the `exp(-i*omega*t)` convention, the implemented leading
thin-layer condition is

\[
 \partial_n P=i\rho\omega V_{\mathrm{wall}}
       -\frac{1+i}{2}\delta\,\Delta_\tau P.
\]

`n` points out of the liquid and `Delta_tau` is the surface Laplacian. This is the
leading locally planar viscous reduction of
[Bach and Bruus (2018), Eq. 26c](https://doi.org/10.1121/1.5049579).
Thermal layers, higher curvature terms and **mean streaming from the wall layer**
are not included. The pressure boundary condition and its nonnegative work
loss are checked against an independent quarter-wave cavity frequency/damping
shift and a source-power identity.

A frozen-target frequency screen from 0.4 to 1.8 MHz selected a promising 1 MHz
branch, followed by a fully coupled stationary inverse solve including bulk
flow. The same constant 5 Np/m attenuation was used across the screen as a
declared sensitivity scenario, **not measured NOA61 frequency-dependent data**.
The screen is neither a global optimum nor a stability result. Its old
`local_linearized_ray_error_um` field records the smallest ray error among
starts, which need not belong to the minimum-total-objective selected drive;
the current code reports the selected drive's value. All optical results above
are independently re-solved coupled states and do not use that screening field.

At the current held geometry, source work is 0.261262 W, comprising 0.065372 W
of bulk absorption and 0.195891 W of viscous acoustic wall loss. The relative
closure error is `3.3e-14`. This verifies consistency of the implemented wave
energy balance; it does not predict temperature. The model predicts a viscous
penetration depth of 8.81 µm and a peak fast normal surface displacement of
36.6 nm across the clear pupil. The latter motion and acoustic index modulation
are absent from the present optical ray calculation.

### Inversion and holding

The method is a repeatable nonlinear inverse design, not a separate manually
chosen heuristic for each radial shape. On each current surface, the solver
computes the complex acoustic response to every row, forms Hermitian quadratic
force kernels including coherent cross terms, and fits wall amplitudes/phases.
The objective combines pupil ray error, height error, integrated cavity pressure
and drive regularization, under an amplitude envelope. It alternates this fit
with a nonlinear capillary/volume solve, then independently re-solves the final
physical drive. There is no claim of uniqueness or global optimality. General
non-axisymmetric freeform targets require a 3D forward model and independent
azimuthal controls.

The current radiation/quiescent-fluid control surrogate has no growing modes
(largest real rate approximately `-238 s^-1`). Its holding file has **zero
feedback gain**. The radiation shape Jacobian is used only in the numerical
implicit time update. The surrogate omits derivatives of the bulk body force
and base circulation, so it is not a full streaming-state stability certificate.
The nonlinear trajectories apply the distributed force, solve actual mean flow,
and provide the recovery evidence reported above.

The slower alternative `formation/` experiment uses ideal full-surface
observations and an inverse-controlled 0.6 s ramp, then holds the same fixed
drive to 1 s. It reaches the same final state with a much smaller peak mean
flow Reynolds number (0.022, versus about 1.3 in the coarsest direct switch-on).
It is an alternative controlled approach, not the evidence for feedback-free
formation.

Direct switch-on can pass through larger pressures and velocities than the
holding state: the 1.25 ms run records 4.23 MPa peak pressure and Reynolds number
1.80, while the 0.625 ms run records 3.93 MPa and 2.23. These transient peaks are
not time-converged. Successive early-trajectory height differences reach 22,
35 and 13 µm across the four time steps. The steady surface and recovery
destination are reproducible; startup loads and precise arrival times need
further temporal verification before hardware design.

The [reproduction guide](stable-cartesian-reproduction.md) records commands and
artifact meanings. Sources are archived **at execution entry** for the current
stationary design, control model and nonlinear trajectories. Earlier exploratory
diagnostics sometimes contain completion-time source hashes; those are not
interchangeable with an execution-entry archive. Numerical state arrays and
earlier experiment outputs have been preserved.
Text-only corrections to several early report scopes and time-integrator labels
are recorded in `metadata-clarifications.json` at the campaign root. Original
reports remain under each result's `provenance/`; histories, numerical metrics,
state arrays and execution-source archives are unchanged.

Software verification completed with 42 passing tests, Python and frontend style
checks, an executed research notebook and real-browser time/geometry/pressure,
camera and layer checks without JavaScript errors. These checks support software
consistency; the independent physical limits and refinement studies carry the
numerical evidence.

## Earlier campaign: what was corrected

The interface is represented by volume-constrained Legendre modes. Previously,
acoustics integrated on a quadratic interpolation of that interface, while
capillarity differentiated the full modal graph. In the old holding state the
maximum geometry error was 237 nm and the maximum normal-vector error was 0.0294,
both concentrated near the rim. The new graph map evaluates the same modal
surface, Jacobian and normals in every finite-element integral:

\[
 (r,z)\mapsto\left(r,z+(1+z/d)h(r)\right).
\]

Here lengths are dimensionless, `d` is chamber depth divided by radius, and
`h(r)` is height divided by radius. Source-patch endpoints are included in the
acoustic mesh, and radial nodes cluster toward the rim. Radiation work uses
`ds*n_z=dr`, rather than combining two different geometric approximations.
The pressure penalty is now an actual volume integral; a nodal average is not
volume-weighted on a graded mesh.

The checks include independent Fourier–Bessel cavity acoustics, exact graph
volume/normals, the zero-gravity spherical cap, conserved liquid volume and
nonnegative viscous dissipation. These checks do not calibrate the apparatus.

## Fixed physical drives under refinement

The following surfaces are independently re-solved at each resolution. Neither
physical wall velocities nor the optical target are changed between rows.
`P4` denotes the acoustic approximation order; the last column counts surface
modes. The axial mesh additionally includes each source-patch endpoint.

| Original drive, corrected geometry | Surface modes | Geometric ray RMS at target |
|---|---:|---:|
| 64 × 96, P4 | 32 | 18.886 µm |
| 64 × 96, P4 | 48 | 18.908 µm |
| 80 × 128, P4 | 48 | 18.771 µm |
| 80 × 128, P4 | 64 | 18.748 µm |

The old 0.389 µm nominal result does not survive these corrections.

| Re-optimized drive, corrected geometry | Surface modes | Ray RMS | OPD RMS |
|---|---:|---:|---:|
| 64 × 96, P4 | 48 | 0.422 µm | 1.55 nm |
| 80 × 128, P4 | 48 | 0.509 µm | 8.83 nm |
| 96 × 160, P4 | 64 | 0.610 µm | 14.48 nm |

This is a substantial improvement in the disclosed numerical model, but the
remaining trend is not an optical convergence plateau. The fixed-drive state
also remains unstable. The raw studies are in
`artifacts/cartesian-coupled-resolution-2026-09-06/` and
`artifacts/cartesian-consistent-2026-09-06/convergence/`.

## Fluid inertia and acoustic shape response

Let `R`, `rho`, `mu` and `sigma` be chamber radius, liquid density, dynamic
viscosity and surface tension. Time and velocity scales are `mu*R/sigma` and
`sigma/mu`. The dimensionless inertial coefficient is

\[
 \beta=\rho\sigma R/\mu^2=2.18844.
\]

Let `xi` denote volume-preserving surface coordinates, `u` finite-element
velocity coefficients and `p` pressure coefficients. `M_u`, `A_u` and `B` are
the fluid mass, viscous and divergence matrices. The kinematic matrix `C` maps
velocity to `dot(xi)`; its transpose applies the reciprocal generalized normal
load. With `f_Gamma` the radiation load, `g` the capillary/gravity energy gradient,
and `b_E` a distributed acoustic body load, the local equations are

\[
 \beta M_u\dot u+A_u u-B^Tp=C^T(f_\Gamma-g)+b_E,\qquad
 Bu=0,\qquad \dot\xi=Cu.
\]

The nonlinear trajectory additionally includes convection relative to the ALE
mesh. Its pressure projection enforces incompressibility after the domain
moves. The shape update preserves volume algebraically. No surface is replaced
by the target, and no fitted damping time is introduced.

For a quiescent state, central differences of the acoustic response give its
shape Jacobian. A divergence-free velocity reduction uses full-fluid resolvents
at several Laplace frequencies. It preserves kinetic energy and viscous
work. For the corrected candidate, the reduction reproduces static mobility
to approximately `2e-8` relative error and an independent complex-frequency
mobility to `1.2e-5` at the highest reported check. Inertia changes the largest
positive growth rate from 37.03 to 36.48 per second. It does not stabilize the
candidate. Capillary-wave dynamics cannot generally be inferred from an
arbitrary first-order relaxation law; see [Denner (2016)](https://doi.org/10.1103/PhysRevE.94.023110).

An initial optimization with a frozen stability sensitivity predicted a decay
rate of 9.97 per second. Recomputing the response at its actual final shape
instead gives growth at 13.47 per second, and its ray RMS is 6.07 µm. That design
is rejected as a stable optical solution. A stability penalty evaluated on an
old geometry is not a certificate.

## Bulk absorption flow

Let `P` and `V` be peak complex acoustic pressure and velocity, `alpha` the
amplitude attenuation coefficient, and `c_s` sound speed. In the declared
homogeneous attenuation model, intensity, bulk Eckart force and absorption
heating are

\[
 I=\tfrac12\operatorname{Re}(PV^*),\qquad
 f_E=\frac{2\alpha}{c_s}I,\qquad
 Q_E=\frac{\alpha|P|^2}{\rho c_s}.
\]

The streaming force follows the homogeneous bulk term in
[Bach and Bruus (2018), Eq. 55](https://doi.org/10.1121/1.5049579); the heating
identity follows by taking the divergence of intensity in the same complex-
wavenumber Helmholtz problem. Boundary-layer streaming and thermal property
gradients are additional mechanisms, not included in this implementation.
[Joergensen and Bruus (2021)](https://doi.org/10.1121/10.0005005) show why thermal
fields can change streaming substantially.

For the corrected quiescent candidate, integrated absorption and source work
both give 0.190191 W, agreeing to approximately `1.2e-14` relative error. A
constrained bulk-flow calculation predicts 1.22 mm/s maximum circulation and
a Reynolds number of 0.020. Its equivalent surface force has a capillary
compliance of 0.841 µm pupil RMS. This is a force diagnostic, not the fully
coupled optical error, but it is too large to dismiss at optical precision.
The acoustic free-surface normal displacement is at most 20.5 nm over the
clear pupil in that calculation.

Bulk flow is therefore included in the stationary inverse design. Let
`W=A_divfree^{-1} C^T`, `M_h=C W`, and `u_E=A_divfree^{-1} b_E`, where the inverse
means a constrained incompressible Stokes solve. Zero interface velocity requires

\[
 f_\Gamma+M_h^{-1}C u_E-g=0.
\]

The residual circulation is `u_E-W M_h^{-1}C u_E`. Reciprocity turns the additional
stationary load into Hermitian quadratic array kernels. Independent body-load
solves verify these kernels and their energy balance. In inertial trajectories,
the distributed body force is applied directly; the stationary equivalent
surface force is **not** used as an inertial substitute.

The resulting absorption-aware stationary fit has 0.424 µm geometric ray RMS,
1.55 nm OPD RMS, 0.193 W source power and a 0.990 m/s largest wall-velocity
amplitude. These are nominal-model results before its own dynamic and spatial
verification. It is not a calibrated NOA61 experiment.

## Surface feedback and physical recovery

The first controller uses the single unstable surface mode. Its gain is derived
from that mode's left eigenvector and the array's force Jacobian. It uses height
observations and does not require fluid-velocity measurement. The full inertial
model is checked independently after this Stokes-based synthesis.

For 5 ms updates, the closed inertial spectrum has largest real part
`-19.71 s^-1`. A zero-order hold with one full sample of sensing/computation
delay has spectral radius 0.906. The same gain at 20 ms with one-sample delay
is unstable. Delay is therefore a quantitative design constraint, not an
omitted implementation detail.

Nonlinear verification uses 192 declared radial height observations, actual
complex wall-speed limits, one sample of delay and reproducible noise. It includes
inertia, convection and distributed absorption forcing. Current recovery studies
start with a 1 µm full-aperture RMS volume-preserving disturbance and zero mean
velocity. Formation from the unforced liquid is a separate experiment.

The first time refinement exposed excessive damping from advancing capillarity
implicitly while freezing the balancing acoustic load. The revised first-order
W method includes a reference radiation shape Jacobian in the implicit update,
while recomputing the actual force at every physical state. Its source matrices
and source-code snapshot are saved with each execution. The reference Jacobian
is a numerical preconditioner; the target is never inserted as an evolving state.
The corresponding 1.8 MHz recovery and refinement results remain in
`artifacts/cartesian-maintenance-2026-09-06/recovery-dt005/` and
`recovery-dt0025/` as method-development evidence. The cancelled earlier
split-step refinement is explicitly marked incomplete. These experiments
precede the load-bearing wall-loss correction and are superseded by the 1 MHz
result at the start of this document.

## Experimental inputs still missing

The liquid's sound speed and attenuation remain assumptions, and loaded
transducer response has not been measured. Sustained operation additionally
requires liquid heat capacity and conductivity, thermal boundary conditions,
temperature-dependent material properties, boundary-layer streaming and
acousto-optic index modulation. The current calculation is isothermal and uses
a constant optical index. Neither an electrical drive voltage nor long-duration
physical stability follows from these simulations alone.

Primary references and derivations for the optical target, normal load and
inverse/control problem are in the [theoretical manuscript](theory/README.md).
Research files and numerical data remain in this private repository; benchmark
material is not an input to this project.
