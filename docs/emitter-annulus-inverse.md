# Optical pupils fixed, annuli determined by full equilibrium

This discrete specialization uses the general source/shape inverse in
`docs/theory/sections/07_source_adjoint.tex`; it does not prescribe a physical
formation trajectory or replace coupled stability by a fit.

Let q be the two volume-conserving, pinned spline coefficient vectors, u the
shared complex source command, M(q) the capillary/gravity force, and A(q,u)
the harmonic acoustic force. The stationary inverse constraints are

    M(q) − A(q,u) = 0                 (every generalized force)
    B_p (q − q_target) = 0            (both fixed Cartesian pupils)
    |u_j| ≤ u_max, u_j = 0 in optical end-cap openings.

Choose orthonormal Z spanning the nullspace of the pupil restriction B_p.
Each column vanishes throughout the optical pupils and retains the original
pinning, axis regularity and volume constraints. Then q = q_target + Z b.
The complement W satisfies WᵀZ = 0. For K = M'(q), a mechanically
preconditioned frozen-wave source step minimizes

    || Wᵀ K⁻¹ [A(q,u) − M(q)] ||².

The remaining displacement Z Zᵀ K⁻¹[A−M] updates the non-optical annuli.
The wave equation and full force residual are then recomputed on the new q.
At an exact fixed point, both W and Z components vanish: **all force equations
hold**, not only pupil observations. Finite residuals must be reported before
any optical acceptance. This iteration has no general convergence guarantee;
failure is not an infeasibility certificate. Damping is numerical relaxation,
not physical time or continuation of the target.

The exact coupled linearization replaces K by K−∂q A. The initial algorithm
uses mechanical preconditioning and fresh nonlinear acoustic solves. It may
require a coupled Newton or secant correction. Source optimization and fixed
point iterations are not evidence of hydrodynamic stability.

A second implementation retains the annulus increment explicitly. With
J = K−∂q A at the reference command, it minimizes all components of
`J^-1[A(q,u)−M(q)] − Z δb`, plus a small penalty on the *total* annulus
change `b+δb`. This regularization discourages unnecessarily large annulus
displacements. It does not remove any force equation. The geometry and field
are recomputed after every annulus update; only a small unprojected nonlinear
force residual permits a stationary result. A reference J is an iteration aid,
not a frozen-field substitution in the final equations.

The pupil nullspace is checked on independent radii and derivatives; grid
convergence must still be tested holding the completed source command fixed.
No optical aperture, conjugate, refractive index or target pupil is relaxed.
The resulting annuli replace the arbitrary C4 extension, not the Cartesian
surface within either clear aperture.
