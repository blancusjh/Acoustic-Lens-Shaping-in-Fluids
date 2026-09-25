# Active precision inverse

The target remains the same two absolute Cartesian patches in one chamber;
neither aperture nor target amplitude is reduced to make a test pass.
The user confirmed on 23 September 2026 that the 10 nm objective applies to
the **cycle-mean driven-liquid surface**. Carrier motion is reported separately;
it is not added to the mean-error acceptance metric. Historical carrier-aware
runs below tested a stricter specification and remain unchanged as evidence.
The user also authorized selecting realistically attainable materials and
operating conditions. The [material feasibility screen](material-feasibility.md)
keeps missing physical data explicit; hypothetical constants are not renamed
as real fluids, and old numerical evidence remains unchanged.

## Hypothesis and inverse formulation

Earlier source synthesis minimized a capillary-compliance-weighted traction
residual. Its best tested candidate still has substantial carrier motion.
The earlier inverse explicitly included the normal-velocity trace, using the
general manuscript's quadratic source map and graph-displacement criterion.
At a specified geometry, for complex source vector `a`,

`F_i(a) = a* K_i a`, `d = O K_mech^(-1) (F(a) − F_required)`,
`b_j = V_j a / (ω |n_z,j|)`.

Here `d` is a **frozen mechanical compliance residual**, not the coupled shape
error; `b` is the modeled carrier height. Both are available from the same
shared acoustic field. A scaled least-squares inverse penalizes `d`, the real
and imaginary parts of `b`, and source effort, with explicit source-component
bounds. Weight scans are numerical optimization experiments, not physical-time
trajectories. Increasing emitters enriches the allowed source space; it does
not remove chamber compatibility, power limits or missing physics.

## Acceptance gates

Observable-source coordinates are obtained from the eigenvectors of
`G = sum_i K_i* K_i`, with a declared eigenvalue cutoff and fourth-root scaling.
This is numerical preconditioning and, when truncated, a source-space
restriction. The transformed inverse penalizes physical amplitude-disk
violations, then uniformly rescales its candidate to satisfy the exact source
limit before reporting metrics. Its solver termination flag is not a
first-order optimality certificate; the gradient diagnostic is also retained.

The semidefinite diagnostic extends the manuscript's existing rank relaxation
with carrier matrices `C_s = b_s* b_s`. It minimizes `t` subject to
`trace(K_i W) = F_i`, `diag(W) <= a_max^2`, `trace(C_s W) <= t`, and `W >= 0`,
omitting the coherent rank-one condition. A higher-rank solution does not supply
a realizable coherent command. Numerical solver status alone is not a verified
dual certificate; truncation bounds apply only within the chosen source space.
The implementation uses [CVXPY's complex semidefinite constraints](https://www.cvxpy.org/tutorial/constraints/index.html),
with the actual package version recorded in each run. This optional diagnostic
can be run using `uv run --with cvxpy==1.9.3 python -m acoustic_freeform.dual.relaxation`.

## Verification sequence

An additional low-carrier restriction selects an orthonormal source basis `N`
such that `||B N||_2 <= ε_a/(sqrt(n_sources) a_max)`. Here `B` contains the
sampled carrier-height response rows. Source limits imply
`||a||_2 <= sqrt(n_sources) a_max`, so commands in this space satisfy
`||B a||_∞ <= ||B a||_2 <= ε_a`. The operator norm is checked numerically.
This is a conservative sampled frozen-model bound, not an unsampled physical
certificate; failure in this restricted space does not imply impossibility.

The sequential convex variant linearizes the quadratic traction map, retains
the exact linear carrier map, and enforces source-amplitude disks in each
subproblem. A trust region and evaluation of the true quadratic merit determine
acceptance. Its mean surrogate remains frozen capillary compliance, not coupled
accuracy. The canonical derivation is in the version 3.3 LaTeX manuscript,
including sampled peak majorants and the low-carrier subspace restriction.

Large-source forward operators can be assembled in blocks: one factorization
is reused across source blocks and only interface pressure, normal velocity
and tangential gradient responses are retained. This preserves coherent cross
terms when constructing the full quadratic force kernels. Full-volume fields
are recomputed separately for each actual fixed command. Tests compare blocked
and full-field responses on curved interfaces; this is a numerical equivalence
check in addition to, not a replacement for, the analytic limits.

1. Analytic and derivative checks pass for the implemented operators.
2. Retain feasible fixed commands, all selection criteria, and the inverse's
   termination and stationarity diagnostics. Optimality is not required to
   demonstrate feasibility, but unconverged candidates must not be called optima.
3. Independently solve the nonlinear coupled stationary problem at fixed commands.
4. On both clear apertures, verify `max|mean error| <= 1e-8 m`. Report carrier
   amplitudes separately; they are not a rejection gate for this specification.
5. Refine acoustic and surface spaces with unchanged physical commands on at
   least three resolutions, leaving a numerical margin below the criterion.
6. Establish the formation trajectory, stability under three-dimensional
   disturbances, and the validity of neglected streaming/thermal effects before
   claiming maintained physical precision. Experimental validation is separate.

Frozen-target screening cannot pass gates 3–6. Even a numerically passing
stationary candidate must not be labelled a stable physical 10 nm lens.

The new mean-only inverse sets the carrier penalty to zero; it does not remove
acoustics from the force balance. Radiation stress, streaming, thermal feedback
and nonlinear averaging can still influence the mean surface. Existing combined
peak objectives and low-carrier restrictions remain optional stricter studies,
not requirements of the selected mean-surface specification.

### Observation-grid audit

The original uniform grid missed a sharp annular compliance extremum near the
wall. The `spline_quadrature` option rebuilds only the observation map from the
unchanged target mechanics, including every spline quadrature radius and the
axis, pupil edge and rim. Cached acoustic force/carrier operators are unchanged;
the replacement observation operator is saved separately with its configuration.
Whole-aperture and annular polynomial extrema are also checked using derivative
roots, spline knots and interval ends. These are floating-point maximum searches,
not rigorous interval bounds, and still refer only to frozen compliance.
An independent volume-preserving quartic tests the known interior annular maximum.

## Organization

### Sampled peak objective (23 September extension)

For each face, let `h = O(F(a)-f)` be the frozen capillary-compliance
error samples and `b = B a` the carrier-height samples, divided by 10 nm.
The optional `peak_power=p` objective replaces their RMS penalties by the
two residuals `s_j = ||h_j||_p + ||b_j||_p`, with **unnormalized** finite-vector
norms and `p >= 2`. Consequently `max|h_j| + max|b_j| <= s_j` on those
sample sets. Mean and carrier samples need not coincide for this conservative
inequality. Annular mean residuals and source effort remain separate penalties;
the physical amplitude disks and final feasibility rescaling are unchanged.
The legacy carrier weight is unused in this mode (set it to one in configs).

For nonzero vectors, differentiation uses
`d||v||_p = sum(|v_k|^(p-2) Re(conj(v_k) dv_k))/||v||_p^(p-1)`.
At the all-zero vector a zero subgradient is selected. Scale-normalized
evaluation avoids overflow. Directional finite differences test physical and
reduced complex-source coordinates at powers 8 and 16; a separate assertion
checks the sampled maximum majorant. This changes the numerical objective,
not the mechanics or the meaning of the frozen approximation. Minimizing
the sum of squared face residuals is not exact minimax optimization and
does not establish continuous-aperture or coupled accuracy.

### Exact interior elimination

The optional `wave_linear_solver: static_condensed` changes only factorization.
Partition pressure unknowns into element interiors `i` and the mesh skeleton
`b`. Because interior basis functions do not cross element boundaries, `A_ii`
is block diagonal. Solve
`(A_bb - A_bi A_ii^-1 A_ib) p_b = f_b - A_bi A_ii^-1 f_i`, then recover
`p_i = A_ii^-1(f_i - A_ib p_b)`. All pressures are recovered before computing
the original full-system residual, traces and traction. No wave modes or
coherent cross terms are discarded. Local singular blocks are rejected,
not regularized. The full factorization remains the default and reference.
Curved-interface tests compare P2, P3 and P4 pressures, weak fluxes and force
kernels, including blocked source solves. This is linear-algebra verification,
not a substitute for analytic limits or mesh convergence. Memory/time benefits
on the large candidate must be measured before being claimed.

The experimental `reused_condensed` option retains that first factorization
only as a GMRES preconditioner. For every new geometry it reassembles `A(q)`
and the source right-hand side, solves the **new** system, and checks its
unpreconditioned relative residual. GMRES uses relative tolerance 1e-12 and
at most three 20-vector restart cycles; failure or residual above 2e-12
triggers a fresh exact condensed factorization. Changed-geometry tests compare
with a fresh full solve, and a forced-failure test checks the fallback.
This is not a frozen-wave or frozen-sensitivity model. Iteration and refresh
counts are numerical diagnostics, not physical time or stability evidence.

Weak normal-flux recovery can also assemble only elements supporting the
interface test functions. Compact support makes all other element contributions
to those rows identically zero; the phase-restricted boundary and source terms
are retained. The legacy whole-domain row assembly remains available for
comparison via `DualAcoustics(..., local_flux_assembly=False)`. P2/P3/P4 curved
tests compare its recovered fluxes and loads with the localized assembly.
This removes redundant assembly, not interface physics. A running experiment
keeps its captured source version; later edits do not change it in flight.

### Higher-order acoustic verification

Orders five and six use complete equispaced triangular Lagrange polynomials.
At the barycentric node `(i,j,k)/p`, `i+j+k=p`, the local function is the product
of `prod_{m=0}^{i-1}(p*l_0-m)/i!` and the analogous factors for `j,l_1` and
`k,l_2`. This gives the nodal delta property and spans the complete polynomial
space. Products and derivatives are evaluated without dividing by a potentially
zero barycentric coordinate. Vertex/edge/interior ordering matches the reference
triangle. Tests reproduce every monomial and its gradient through degree `p`,
compare order four with the installed independent implementation, and check
random shared-edge traces. Full/condensed curved-wave comparisons include P5/P6.
The independent planar scattering limit must also be checked before using
these elements for candidate refinement. This supplies additional pressure
degrees of freedom without changing the physical source regions or commands.

### Locations

Fixed-command verification optionally uses a matrix-free Krylov root solve
(`stationary_solver: krylov`) instead of the default dense finite-difference
hybrid method. Both solve the same freshly coupled force residual with the
same post-solve acceptance test. Each trial records a residual and solve count,
not a physical time or an accepted iterate. The Krylov iteration limit is an
outer-iteration budget, not a bound on acoustic solves. A small shared-wave
regression compares the two algorithms; convergence of a large candidate
still requires independent fixed-command refinements.

New precision configurations are in `configs/dual/precision/`; their outputs
are grouped under `artifacts/dual-precision-2026-09-22/`. Older configurations
and evidence keep their original paths. All runs include provenance and a
declared model. The private research archive preserves pre-edit milestones.
September 23 continuations are in `artifacts/dual-precision-2026-09-23/`.

The [evidence catalog](../artifacts/catalog-2026-09-23-cycle-mean/report.md) inventories
configured experiment directories without inferring scientific success from
file presence. Rebuild to a new destination with `tools/catalog_research.py`.

Run large two-dimensional wave jobs sequentially on the current 16 GB host.
Concurrent refined equilibria and convex factorizations caused severe swapping
on September 23; interrupted runs retain explicit reports. More source regions
require memory planning as well as convergence checks. Do not confuse resource
limits with a physical reachability bound.
