# Theory sketch: from a prescribed SSL to acoustic commands

Companion to the [version-1 vision](ssl-generator-vision.md), 23 September 2026.
This is a design-level specialization of the existing general theory, not a
replacement for the canonical LaTeX source and not an additional validation
result. Derivations below state the reduced model in which their conclusions
hold. The source space is continuous first; an array is its finite restriction.

## 1. Prescribed optical geometry

Let the vertices be v0 and v1 = v0 + t, with t > 0. All axial coordinates are
laboratory coordinates. Write the full surfaces as z = h0(r), z = h1(r),
where h0(0) = v0, h1(0) = v1 and 0 ≤ r ≤ R. Light enters medium 0, traverses
medium 1 and exits into medium 2. The indices are evaluated at the declared
design wavelength.

The two local Cartesian prescriptions are exactly

\[
 (n_0,z_0-v_0,n_1,z_1-v_0),\qquad
 (n_1,z_1-v_1,n_2,z_2-v_1).
\]

No sign flip alone can replace the translation by the second vertex. The
intermediate point is common in laboratory space. For example, if z1 lies
beyond both faces, it is a real image for the first face and a virtual object
for the second, provided the transmitted rays meet the second face before
crossing that point.

For interface j joining conjugates Zj and Zj+1, define distances Dj and Dj+1
from a surface point to these axial points. The unsquared path equation is

\[
 \Psi_j(r,z)=s_{o,j}n_jD_j+s_{i,j}n_{j+1}D_{j+1}=C_j,
 \quad C_j=\Psi_j(0,v_j).
\]

Here the object/image signs are +1 for real and −1 for virtual conjugates.
For finite nonzero local axial distances and forward propagation at the
vertex, s_o,j = −sign(z_j−v_j) and s_i,j = sign(z_j+1−v_j).
Choose a regular graph branch through the vertex, with ∂zΨj ≠ 0, and verify
the physical transmission branch over the entire admitted aperture.

The tangential derivative of Ψj is zero, giving the tangential Snell condition
n_j d_in,t = n_j+1 d_out,t. This alone does not choose forward transmission:
normal orientation, nonnegative Snell radicand, non-grazing incidence and
successive ray intersections must also be checked. With the shared z1 and
the correct branches, adding the two signed path relations gives the
constant optical path through the actual singlet. This construction follows
[Silva-Lora and Torres (2020)](https://doi.org/10.1364/JOSAA.392795); it does
not establish arbitrary field-angle or wavelength correction.

## 2. Frame, volume and optical access compatibility

Require h1(r)−h0(r) > 0, clearance from both end caps, and the specified
rim conditions h_j(R) = H_j. Central thickness t is not generally H1−H0.
The enclosed lens volume is

\[
 V_1=2\pi\int_0^R r\,[h_1(r)-h_0(r)]\,dr.
\]

For flat end caps at Zb and Zt, the two surrounding phase volumes are

\[
 V_0=2\pi\int_0^Rr[h_0(r)-Z_b]dr,\qquad
 V_2=2\pi\int_0^Rr[Z_t-h_1(r)]dr.
\]

Their sum is fixed by the cylinder. Two independent phase-volume constraints
must match an existing sealed apparatus. Optical rays must also clear the
declared source/window geometry; an interface pupil alone is not a complete
apparatus-level obstruction check.

These equations explain a central restriction on adaptability: full-frame
Cartesian surfaces with freely varying conjugates generally do not preserve
fixed H0, H1, V0 and V1. Compatible families must be found subject to those
constraints, or the apparatus/completion policy must explicitly supply extra
degrees of freedom. Version 1 checks compatibility rather than silently
changing a user-supplied prescription.

## 3. Variational mechanical load, including gravity

Let Δρ_j = ρ_j−ρ_j+1, with the normal directed upward from the lower phase to
the upper. For constant surface tension and gravity g > 0 directed downward,
the capillary–gravity energy, up to fixed constants, is

\[
 E[h]=\sum_{j=0}^1\int_{D_R}
 \left[\sigma_j\sqrt{1+|\nabla h_j|^2}
       +\tfrac12\Delta\rho_j g h_j^2\right]dA.
\]

The surfaces here are absolute laboratory graphs, not displacements from an
unspecified plane. Under pinned, volume-preserving variations η_j,

\[
 DE[h](\eta)=\sum_j\int_{D_R}
 \left[-\sigma_j\nabla\cdot
 \frac{\nabla h_j}{\sqrt{1+|\nabla h_j|^2}}
 +\Delta\rho_j g h_j\right]\eta_j\,dA.
\]

Thus a quiescent target needs the upward acoustic mean traction

\[
 f_{\mathrm{req},j}=\sigma_j\kappa_j+\Delta\rho_j g h_j-\lambda_j,
 \qquad
 \kappa_j=-\nabla\cdot
 \frac{\nabla h_j}{\sqrt{1+|\nabla h_j|^2}}.
\]

The constants λ_j are the independent hydrostatic/volume pressure multipliers,
not arbitrary spatial correction functions. For axisymmetric graphs,

\[
 \kappa_j=-\left[
 \frac{h_j''}{(1+h_j'^2)^{3/2}}
 +\frac{h_j'}{r\sqrt{1+h_j'^2}}\right],
 \qquad \kappa_j(0)=-2h_j''(0).
\]

Regularity gives h'_j(0) = 0. All terms in f_req have units Pa. The density
jump can be negative; gravity must not be assigned the same restoring sign
on both faces by convenience. The formula is a quiescent, constant-property
load requirement. Persistent mean flow, thermal gradients or tangential
stresses require the corresponding full balances instead. See canonical
[mechanics](theory/sections/02_mechanics.tex) and
[viscous inverse](theory/sections/08d_viscous_inverse.tex).

## 4. Continuous acoustic source space

For a fixed geometry and carrier angular frequency ω, let u(x) be an admissible
complex boundary command on the permitted support Γa, in m/s. Source support,
amplitude limits and regularity define U_ad. Volume sources can be added in
the general theory; the present specialization uses boundary actuation only.

In the declared inviscid, homogeneous, linear bulk acoustic reduction,

\[
 \nabla\cdot(\rho_\ell^{-1}\nabla P_\ell)
 +\frac{\omega^2}{\rho_\ell c_\ell^2}P_\ell=0,
 \qquad \mathbf V_\ell=\frac{\nabla P_\ell}{i\omega\rho_\ell}.
\]

Peak phasors use exp(−iωt). Pressure and normal velocity are continuous across
the interfaces in this high-frequency reduction. On the outer boundary,

\[
 \mathbf V\cdot\mathbf n_a=YP+u,
 \qquad \operatorname{Re}Y\geq0,
\]

with u = 0 outside Γa. The optimized command is u, not total normal velocity
when Y ≠ 0. This is a fully specified idealized source model, not a promise
about electrical voltages. Require a well-posed wave problem at the chosen
frequency. In operator notation P = A_h^{-1} B_h u; the operator changes
with the surfaces.

The normal acoustic momentum flux in one phase is

\[
 \mathcal M_{nn}=\frac{|P|^2}{4\rho c^2}
 +\frac{\rho}{4}(|V_n|^2-|\mathbf V_t|^2).
\]

The upward radiation traction is its lower-minus-upper jump. It is quadratic
in the source phasors and can have either sign at a two-fluid interface.
It is not the oscillating pressure P and cannot generally be replaced by a
single pressure-intensity term. The two surfaces interact through one shared
wave problem. Radiation-pressure deformation and the importance of coupling
field to geometry are studied by
[Chesneau et al. (2022)](https://doi.org/10.1103/PhysRevE.106.065104).
Their work is a mechanism reference, not validation of this apparatus.

Bulk loss, wall-layer streaming and temperature feedback change these
operators and balances; they are not supplied simply by choosing a passive
Y. Relevant extended acoustic theory is given by
[Bach and Bruus (2018)](https://doi.org/10.1121/1.5049579) and
[Jørgensen and Bruus (2021)](https://doi.org/10.1121/10.0005005).

## 5. Inversion and the finite-array restriction

At the fixed target h*, take a dense collection of admissible, dimensionless,
volume-preserving graph test functions φ_i spanning both whole interfaces.
Pair the acoustic traction and required load with these tests. The resulting
generalized forces have units N. The continuous inverse is

\[
 \langle u,\mathcal K_i(h^*)u\rangle=b_i(h^*),
 \qquad u\in\mathcal U_{\mathrm{ad}},
\]

where the Hermitian quadratic operators arise from the linear acoustic trace
map and momentum-flux pairing. Equality for finitely many tests is only a
discrete projection of this statement.

Restrict the source to dimensionless patterns e_k(x):

\[
 u(x)=\sum_{k=1}^{N}e_k(x)w_k,
 \qquad w_k=A_k e^{i\phi_k}.
\]

Then F_i(w;h*) = w†K_i(h*)w. Both faces are in the same system; coherent
cross terms must be retained. A practical dimensionless regularized problem is

\[
 \min_{|w_k|\leq w_{k,\max}}
 \frac12\sum_i\left(
 \frac{w^\dagger K_iw-b_i}{F_{i,\mathrm{ref}}}\right)^2
 +\frac{\beta}{2}\sum_k\left|\frac{w_k}{w_{k,\mathrm{ref}}}\right|^2.
\]

Positive force and source reference scales make the trade-off explicit.
Source effort is not automatically acoustic power. At fixed geometry,

\[
 DF_i(w)[\delta w]=2\operatorname{Re}(w^\dagger K_i\delta w).
\]

This supplies the force Jacobian for source optimization. A common phase
rotation leaves every mean force unchanged, so phase gauge/nonuniqueness
must be handled. Phase is immaterial for a zero-amplitude channel.

The feasible set is nonconvex. Lifting to W = ww† makes force constraints
linear, but a deterministic coherent command requires W ≥ 0 and rank W ≤ 1.
Dropping rank gives a relaxation, not an automatically realizable array.
Increasing N enlarges the available subspace only when the old patterns and
constraints are retained; arbitrary array redesigns are not nested.

For any real coefficients a_i and bound ||w||₂ ≤ C,

\[
 |\textstyle\sum_i a_i b_i|
 \leq C^2\|\textstyle\sum_i a_i K_i\|_2
\]

is necessary. If the combined matrix is positive semidefinite, its required
force combination must also be nonnegative. Violations can certify restricted
model infeasibility. Passing these tests, or optimizer stagnation, proves
neither reachability nor its absence. The general limitations are developed
in canonical [reachability](theory/sections/05_reachability.tex) and
[finite arrays](theory/sections/06_arrays.tex).

## 6. The actual coupled equilibrium and optical verification

Let q describe admissible interface coordinates, w the fixed source command,
and R(q,w) = ∇E(q)−F_ac(q,w) the generalized force residual. The stationary
forward problem is

\[
 A_qP=B_qw,\qquad R(q,w)=0,
\]

with fixed rim/volume constraints and positive fluid gaps. A numerical root
iteration changes q to satisfy these equations; it is not a time integrator.
Starting near the desired q* is legitimate. Multiple roots or attraction
basins remain possible and must not be hidden by that initialization.

Write K_mech = ∂q∇E, A_ac = ∂qF_ac and B_ac = ∂xF_ac, where
x = (Re w, Im w) uses real command coordinates. If the constrained derivative
is invertible, the local static sensitivity is

\[
 K_{\mathrm{eff}}=K_{\mathrm{mech}}-A_{\mathrm{ac}},
 \qquad \delta q=K_{\mathrm{eff}}^{-1}B_{\mathrm{ac}}\delta x.
\]

This distinguishes a purely mechanical compliance approximation from the
coupled response. Near singular K_eff, small loading errors may produce
large shape changes. A pressure norm alone is therefore not an optical
accuracy certificate.

For a differentiable optical objective J(q,x), solve
K_effᵀ λ = ∂qJ. The total derivative along this equilibrium branch is
dJ/dx = ∂xJ + B_acᵀλ. This is the local adjoint route for improving commands
while leaving the prescribed target fixed. It assumes a regular branch and
valid optical derivatives; clipping, critical refraction and a maximum over
changing worst-case rays need constrained or nonsmooth treatment. The general
adjoint design framework is described by
[Giles and Pierce (2000)](https://doi.org/10.1023/A:1011430410075).

Always evaluate the final finite change with fresh coupled solves. Compute
both height maxima on the fixed qualified pupils and the maximum radius at
the prescribed detector for the fixed incoming cone. Check full ray coverage
and actual window/source clearance separately. Establish convergence under
fixed physical commands before treating re-optimization as an improvement.

## 7. Stability and adaptation are separate extensions

Under a justified three-fluid Stokes reduction the graph evolution is

\[
 \dot q=M(q)[F_{\mathrm{ac}}(q,w(t))-\nabla E(q)],
\]

where M comes from incompressible viscous flow, not an arbitrary relaxation
constant. Around a quiescent equilibrium its linearized generator is
−M* K_eff. A positive mechanical stiffness alone does not establish stability
because the acoustic shape derivative enters. In an appropriate reduced
inertial–viscous description one instead examines

\[
 M_{\mathrm{in}}\delta\ddot q+D\delta\dot q
 +K_{\mathrm{eff}}\delta q=0.
\]

That equation is schematic until the mass, dissipation, moving-domain and
acoustic approximations are derived consistently. A streaming base state
requires linearization of its complete flow and field equations, including
advection; it cannot inherit a quiescent stability label. Axisymmetric design
does not exclude non-axisymmetric disturbances.

A reconfigurable family needs both apparatus-compatible stationary states
and admissible transitions. For each requested path, test tracking error,
settling, command bandwidth and applicable model limits. Linear interpolation
of source commands does not generally interpolate surfaces or preserve
stigmatism, since the force is quadratic and the field depends on geometry.
This is why version 1 solves a valuable stationary problem first, with
adaptation as a subsequent, explicitly verified layer.

## 8. Independent checks before family-wide claims

- Optical limits: independently trace the unsquared Cartesian branches,
  check a supported collimated limit, and verify coordinate translations of z1.
- Geometry: independently integrate all phase volumes, check minimum gaps
  and rim heights, and reject incompatible fixed-machine requests.
- Mechanics: recover planar hydrostatics and the spherical Laplace pressure
  with the declared normal convention; compare analytic energy variations.
- Acoustics: check independent slab/cavity limits, interface flux continuity
  and power balance; distinguish pressure continuity assumptions from full
  thermoviscous interface physics.
- Inverse sensitivities: compare analytic derivatives against independent
  finite differences at multiple steps and operating states.
- Stationary optics: refine wave mesh/order, surface representation and ray
  sampling with fixed physical commands and fixed optical prescription.
- Adaptation: use an independently configured dynamic model and perturbation
  tests before claiming formation, stability or continuous stigmatic tracking.

The central conditional statement is precise: **if an admissible command
produces a verified coupled equilibrium within the specified surface and
optical tolerances, it realizes that SSL within the declared stationary
model.** It does not require a formation simulation to be a meaningful
stationary result, and it does not imply that every SSL is reachable or stable.
