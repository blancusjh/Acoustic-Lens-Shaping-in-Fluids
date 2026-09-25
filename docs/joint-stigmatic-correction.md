# Joint stigmatic prescription correction — 2026-09-23

The earlier `asymmetric-pair-A` specified two independent Cartesian patches.
Its front image conjugate and back object conjugate were different laboratory
points. It was not a point-to-point stigmatic singlet; increasing acoustic
accuracy toward that prescription would not repair this optical mismatch.

## Construction, before acoustic synthesis

Use the unsquared signed-path construction and physical Snell branch already
derived in the canonical manuscript, section “Cartesian geometry as a target
family”, and [Silva-Lora and Torres (2020), section 3, p. 1158](https://doi.org/10.1364/JOSAA.392795).
The paper explicitly uses local distances dₖ − ζₖ and dₖ₊₁ − ζₖ.
Let laboratory vertices be v₀ and v₁. For a common intermediate
conjugate z₁, prescribe local distances:

| Interface | Object distance from vertex | Image distance from vertex |
|---|---|---|
| Front | z₀ − v₀ | z₁ − v₀ |
| Back | z₁ − v₁ | z₂ − v₁ |

Between the interfaces the front outgoing spherical wave must be the back
incoming spherical wave. For z₁ beyond both interfaces it is converging at both:
z₁ is a real image of the front and a **virtual object** of the back. Using a
negative back object distance in this case constructs the wrong incident wave.
The signed intermediate optical-path terms cancel along the actual rays,
giving constant total positive path from z₀ through both faces to z₂. Physical
transmission, aperture interception and branch selection must still be checked.

The initial corrected example retains z₀ = −101 mm, z₂ = +151 mm, the two 2 mm
clear radii, chamber dimensions and vertex displacements −150/+120 µm. The
selected middle conjugate is +201 mm; the user subsequently confirmed no
preference for the numerical conjugate values.
The old targets change: this is a declared optical correction, not improved
accuracy against the old independent pair. Material indices remain hypothetical.

Pinned, C2 non-optical annuli retain sealed-fluid volumes. Their load objective,
and every subsequent stationary mechanical residual, includes nonlinear
capillarity and (ρ_below − ρ_above) g h with g = 9.81 m/s². No zero-gravity
substitution is made. The general inverse conditions remain unchanged: find a
shared source command whose recomputed acoustic traction balances this complete
mechanical load, then test fixed-command convergence and stability. A target
projection is neither a solved force balance nor a physical trajectory.

## Acceptance and ray convention

The added optical gate uses maximum radius about the prescribed optical axis
at the fixed detector, not RMS or a fitted centroid. All sampled rays must
transmit; discarding rays cannot produce a pass. Axis and rim rays are included
in this audit, and ray counts are refined. This is not an interval certificate
over the continuous pupil or a diffraction PSF calculation.

The corrected target audit fixes its illumination cone using points across
the exact front target pupil. Reuse those same directions throughout a future
formation run; do not aim rays at its changing shape. This differs explicitly
from the old viewer's cone through a flat reference plane. Real entrance/exit
windows and their refraction are not included in this two-interface model.

## Reproduction and present result

Run `.venv/bin/python -m acoustic_freeform.dual.stigmatic_audit
configs/dual/stigmatic-optical-audit.json --out <new-artifact-directory>`.

Results: `artifacts/dual-stigmatic-2026-09-23/optical-audit/`, with configuration,
source provenance and validation. All 4001 exact-target rays transmit and the
maximum geometric radius is at floating-point roundoff. At 52 surface elements,
the *projected target* errors are approximately 0.00137/0.000913 nm and its
maximum geometric spot radius is 0.00108 µm. Refinement to 80 elements reduces
both representation errors. These numbers establish an optical/numerical target
prerequisite, **not acoustic formation, physical 10 nm accuracy or stability**.

Tests independently check vector Snell propagation and total positive optical
path through the pair. Separate volume-energy tests check the gravitational
term for stable, inverted and equal-density stacks.

## Slope-sensitive inverse objective

The first corrected 224-source coupled equilibrium has design-grid height
errors 27.8/20.8 nm but a sampled maximum geometric spot radius around 20 µm.
Thus height-only fitting is insufficient. Differentiating vector Snell's law
shows that changing the graph normal changes the transmitted direction; after
free propagation a direction error produces a detector displacement proportional
to propagation distance. High-spatial-frequency height errors can have much
larger slopes than smooth errors of the same amplitude.

The inverse now optionally augments its existing frozen compliance observation
H K⁻¹(F_ac − F_required) with ℓ D_r K⁻¹(F_ac − F_required), restricted to the
optical pupils. Here ℓ has units of metres; ℓ = 1 mm assigns a 10 µrad slope
error the same residual magnitude as a 10 nm height error. The complex-source
gradient follows by applying the same additional linear observation to the
already-derived quadratic force derivative. Gravity remains in K and the
required force. This changes the source-selection objective, not the target,
fluid physics or aperture. Height and slope diagnostics are reported separately.
It is a regularity proxy, not an exact optical bound: actual coupled states
must still pass the full two-interface maximum-radius ray trace.

## Direct optical objective and coupled-response check

The direct optical observation now uses the derivative of both ray intersections,
both vector Snell maps and the fixed-detector intersection. Its normalized
residual is the signed meridional spot divided by 1 µm; the height residual is
divided by 10 nm. This is a local optical prediction, not a ray trace of a
coupled equilibrium. Independent finite perturbations of both surfaces check
the optical Jacobian.

The immutable `direct-candidate-01/source.npz` gives frozen pupil displacement
estimates 7.927/7.981 nm, but its full-surface estimates reach 83.52/83.32 nm.
`kernel-audit-direct-448` recomputes a single driven wave at the same target.
Its discrepancy from the cached all-source quadratic force is below
3.1 × 10⁻¹⁶ m in mechanical-compliance units. Thus cache disagreement does not
explain the much larger errors seen in subsequent coupled root **trials**.
Those trials are not accepted equilibria or physical trajectories.
This command also produces approximately 53.3 MPa sampled pressure and 47.66 W
source work in the ideal model. These are material validity concerns, not
evidence of an experimentally workable device.

To test the omitted static feedback, write the stationary residual as
R(q,a) = F_mech(q) − F_ac(q,a). At the reference target and fixed command, let
K = ∂F_mech/∂q, A = ∂F_ac/∂q and d = F_ac − F_mech. The first-order correction
is (K − A) δq = d, **not** K δq = d. This is the static specialization of the
canonical coupled linearization, not a new physical assumption.

The low-mode audit selects columns V from the generalized mechanical
stiffness/mass eigenproblem, sets W = (Vᵀ K V)⁻¹ Vᵀ K, and measures A V by
central differences of independently recomputed wave fields. Its explicitly
truncated approximation is A ≈ (A V) W. With L = K⁻¹ A V, its compliance is

    C = K⁻¹ + L (I − W L)⁻¹ W K⁻¹.

This formula is independently tested against a fully coupled linear system.
The configured experiment compares ranks 4, 8 and 16 and halves the finite
difference step on the first four directions. Rank dependence must be checked
before using this operator in an inverse. Even a converged static response is
not a stability spectrum: inertia, dissipation, streaming-base-flow perturbations,
thermal feedback and non-axisymmetric disturbances are not tested here.

The completed 4/8/16-mode test is not converged: predicted front/back maxima
are 15.8/38.5, 41.9/22.9 and 35.4/40.0 nm. The independently solved 104-element
equilibrium instead has 368.9/740.7 nm errors and a 51.66 µm maximum spot radius;
its force-balance residual is 5.74 × 10⁻¹⁴ m. It fails both requested criteria.
The finite-difference half-step discrepancy in the first four directional
compliances is below 2.6 × 10⁻⁷ (dimensionless), but this does not repair the
inadequate direction space. See `coupled-response-direct-448` and
`optics-verify-direct-448-design`; the latter explicitly marks remaining
campaign refinements as unfinished.

The next restricted response audit generates V from the Krylov space of
K⁻¹ A, starting with K⁻¹ d and using twice-reorthogonalized K-inner products.
This directs the static audit toward the actual force defect rather than
assuming that the lowest mechanical modes capture it. The same inverse
identity applies. This space remains load-specific: it does not establish
the full source-to-shape derivative. Predictions at ranks 8/16/32 are checked
by fresh **nonlinear** residual evaluations, not merely by the projected
matrix equations.

For the fixed outward-diverging entrance cone in this prescription, the rim
ray stays inside the front pupil only if the front height at the rim does not
rise above its target value (on the same non-grazing interception branch).
An optional one-sided rim observation is now available in the inverse.
Its actual quadratic residual violation is penalized and reported separately;
the final full ray trace remains mandatory. A direct geometric test verifies
the interception sign using ±5 nm front shifts. No pupil is shrunk and no
lost ray is removed from the acceptance gate.

## Annular regularity experiment

The optical pupils do not determine the non-optical extensions uniquely.
The C4 experiment leaves the exact pupils, vertices, shared conjugates, rim
pinning and phase volumes unchanged. It replaces the C2 degree-seven annular
polynomial with a degree-eleven polynomial matching four derivatives at the
pupil boundary, and minimizes the same gravity-containing mechanical-load
objective over its remaining coefficients. Taylor coefficients follow by
implicit expansion of the unsquared Fermat equation, not differencing sag
samples. Independent checks verify fifth-order optical-path contact, pinning
and phase-volume conservation.

The hypothesis is that a smoother required traction near the pupil/annulus
join is easier for the finite acoustic source space to approximate. This is
not established by the construction: the required load may also increase.
The C4 run is a declared redesign of the non-optical target extension, not
a convergence test for the earlier C2 surface and not a change to either
optical pupil or the requested accuracy. No dynamic trajectory is supplied
by polynomial matching.
