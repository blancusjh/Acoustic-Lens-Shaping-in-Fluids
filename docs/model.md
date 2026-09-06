# Finite acoustic asphere: model contract

This is a numerical experiment in acoustically correcting a finite liquid lens.
It is not an experimentally calibrated NOA 61 fabrication process. Every numerical
state starts from the same declared chamber and fill volume.

## Apparatus and material

The liquid occupies a cylinder of radius 4 mm and depth 6 mm, plus a convex cap.
The free surface is pinned at the sharp rim `r = 4 mm, z = 0`; the optical pupil is
`r ≤ 3 mm`. A 0.7 mm flat window closes the bottom. The surrounding housing is
0.8 mm thick. Slow fluid motion obeys no slip on the wall and window. Perfect rim
pinning is an imposed boundary condition, not a measured wetting law.

Sixteen axial rows of sidewall actuators each have sixteen electrical sectors.
All sectors in one row share a complex drive for this axisymmetric experiment.
The acoustic calculation treats a row as azimuthally continuous. The visual seams
between electrical sectors do not represent a solved 3D scattering geometry.
Each row has an axial cosine taper and 84% active fill. Sources specify complex
normal wall velocity, not voltage. The optical axis passes through the flat base
without being obstructed by the sidewall array.

The [Norland NOA 61 technical data sheet](https://norlandproducts.com/wp-content/uploads/2025/02/Norland-Products-NOA-61-TDS.pdf)
provides liquid refractive index 1.52, density 1231 kg/m³, viscosity 0.3 Pa·s at
25°C and surface tension 0.04 N/m. The **cured** index is 1.56 and is not used here.
The assumed acoustic speed is 1600 m/s and amplitude attenuation is 5 Np/m.
Neither acoustic value is supplied by that manufacturer sheet. The nominal
monochromatic optical wavelength is 589.3 nm; liquid dispersion is not calibrated.

The initial condition is the **unforced nonlinear equilibrium** for the specified
volume. It is not a transient labelled as a rest state. The volume is chosen to
equal that of the target cap plus the cylindrical chamber. Gravity is 9.81 m/s².
All geometry, material and timing values are in the experiment TOML.

## Optical objective

For the original `noa61_asphere` experiment, parallel rays enter through the flat base, propagate vertically through the liquid
and refract at the curved liquid–air boundary. The target is equal optical path to
a point 20 mm above the target vertex. If `n = 1.52`, `f = 20 mm` and
`R = f(n − 1)`, its vertex-relative sag is

\[
z(r)=-\frac{r^2}{R+\sqrt{R^2+(n^2-1)r^2}}.
\]

The height is shifted by `−z(4 mm)` to meet the rim. This is a hyperbolic conic with
`K = −n² = −2.3104`, not an arbitrary sum of bumps. The simulation evolves surface
coefficients; it does not enforce this conic as a kinematic constraint.

The finite-conjugate extension uses the signed Cartesian-oval construction of
Silva-Lora and Torres (2020). Its four optical parameters, prescribed incident
wavefront inside the resin, mounting/fill constraint and exported excitations are
specified in [Cartesian diopters](cartesian-diopters.md). The original formula
above is its collimated-object limit. A finite-conjugate oval is generally not an
exact conic. Both the incident conjugate and image plane remain fixed during each
physical trajectory.

After evolution, Snell rays and optical path are recomputed from the actual height
and slope. Pupil sampling is uniform in area. The target detector remains at the
same physical plane throughout the movie. Best-focus metrics allow each comparison
surface to refocus. The best-fit sphere is only an optical comparison, not a second
fluid equilibrium. Diffraction, Fresnel loss, vector polarization, roughness,
chromatic aberration and curing shrinkage are outside this optical calculation.

Connecting fluid shape and ray objectives follows the research direction of
[Elgarisi et al., Optica (2021)](https://doi.org/10.1364/OPTICA.438763) and
[Na et al., SIGGRAPH Asia (2024)](https://doi.org/10.1145/3680528.3687584).
Their experiments do not validate this acoustic apparatus.

## Coupled acoustics and liquid motion

Complex amplitudes use `exp(−iωt)` and **peak**, not RMS, pressure. In the current
curved chamber,

\[
\nabla^2 P+(\omega/c+i\alpha)^2P=0,
\qquad \mathbf V=\frac{\nabla P}{i\rho\omega}.
\]

The free surface is pressure release (`P = 0`), a liquid–air impedance-contrast
approximation. Inactive wall and base are acoustically rigid. Active wall patches
prescribe normal velocity. Glass elasticity, piezoelectric mechanics and acoustic
loading by the housing are not included.

The cycle-averaged normal acoustic momentum flux is

\[
M_{nn}=\frac{|P|^2}{4\rho c^2}
+\frac{\rho}{4}(|V_n|^2-|V_t|^2).
\]

At the pressure-release surface, `P = V_t = 0`, giving the upward radiation traction
`Π = ρ|V_n|²/4`. Coherent array cross terms are retained. The acoustic field is
recomputed on the actual evolving curved surface. The coupled radiation/flow
approach is motivated by [Chesneau et al., PRE 106, 065104 (2022)](https://doi.org/10.1103/PhysRevE.106.065104);
our implementation uses finite elements, a finite chamber and a one-fluid
pressure-release approximation, rather than reproducing their two-fluid BEM solver.

The free-surface energy, after removing constants associated with the fixed chamber,
is

\[
E=2\pi\sigma\int_0^a r\sqrt{1+h_r^2}\,dr
+\pi\rho g\int_0^a rh^2\,dr,
\qquad V=\pi a^2d+2\pi\int_0^a rh\,dr.
\]

Full nonlinear curvature is used. The instantaneous fluid velocity solves the
axisymmetric creeping-flow equations, with the hoop strain included. A P2/P1
Taylor–Hood discretization gives the mobility connecting generalized surface forces
to height rates. The surface basis is `P_j(2(r/a)² − 1) − 1`; every basis function
has zero rim height and zero axial slope. Integration is confined to the exact
constant-volume subspace.

The mobility comes from fluid solves, not an arbitrary relaxation constant. A
first-order semi-implicit step treats capillary stiffness implicitly and acoustic
forcing at the beginning of the step. This is a Stokes approximation: fluid inertia
is omitted. The observed maximum convective Reynolds number is about 0.005. The
momentum-diffusion time `ρa²/μ ≈ 0.066 s` is approximately 5.5% of the 1.2 s drive
ramp; fast transients still require an unsteady-flow model for quantitative timing.

The finite lens uses cubic acoustic elements on a quadratic curved mesh, with
quartic-element refinement checks. These discretization choices are explicit.
The separate planar verification package contains earlier small-deformation limits;
none of its periodic boundaries or flat scattering assumptions enter this lens.

## Control and time

The controller has ideal knowledge of the current complete surface and updates at
50 Hz. It requests a smooth change from the unforced equilibrium to the optical
target over 1.2 s, then holds the target through 1.8 s. At each update it fits sixteen
complex physical wall velocities on the current acoustic geometry. Its objective
uses ray error, a smaller figure-error term and a pressure penalty. Each complex
wall speed remains at or below 0.16 m/s.

Only the controller setpoint is interpolated. The fluid state is always integrated
from force balance and viscous motion. A saved trajectory contains its actual
coefficients and drive program. Replay holds each recorded drive until the next
scheduled update, with no shape fitting or observation.

Ideal full-surface feedback is not a demonstrated sensor/control implementation.
Sensitivity of a focused standing-wave cavity makes that distinction relevant.
Experimental transient deformation is established in a different apparatus by
[Sisombat et al., Scientific Reports 13, 14703 (2023)](https://doi.org/10.1038/s41598-023-39464-0).

## Remaining physical work

The held solution has approximately 1.64 MPa peak cavity pressure and 47 mW acoustic
source power in the model. No cavitation threshold is inferred from these numbers.
The attenuation model removes acoustic energy but does not solve heating or the
associated streaming. Thermoviscous boundary layers, streaming, temperature-dependent
properties, actual transducer/window impedance and wetting stability need to be
quantified before predicting a physical device. Curing requires a separate index,
shrinkage and evolving-rheology model. General freeform surfaces require a 3D
interface and independently controlled azimuthal modes.
