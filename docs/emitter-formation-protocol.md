# Emitter-driven viscous formation diagnostic

This test applies the existing variational Stokes mobility from canonical
theory section `08d_viscous_inverse.tex` to the synthesized NOA61/water commands.
It is not a new inverse design or a prescribed-target trajectory.

The declared approximation is

    dq/dt = M(q) [F_ac(q, a(t) g) - grad E(q)].

Here E contains nonlinear capillarity and signed density-jump gravity on both
whole interfaces. M is the reciprocal three-fluid, no-slip, incompressible
Stokes mobility. The harmonic field is solved on the current curved interfaces
using the same saved source commands g. The envelope is sin²(pi t/(2 T_ramp))
until T_ramp, then one. The target enters only optical observations, never the
evolution right-hand side. The fluid viscosities are the explicit material
reference values, not the earlier hypothetical 5 Pa s triplet.

The unforced initial state is q=0, which exactly balances the discrete
mechanical equations for the declared pinned levels and fixed volumes. Its
capillary-gravity Hessian is checked positive on the complete discrete space.
This is a local discrete check, not a global nonlinear or 3D stability proof.

Time stepping freezes M and the acoustic load at the old geometry, evaluates
the commanded envelope at the midpoint, and solves capillarity/gravity
implicitly. It is first order. A step exceeding the declared surface-change
or gap guard is rejected and retried with a smaller step. Every accepted state
is saved; no target interpolation or endpoint replacement is allowed.
The acoustic-work minus energy-change diagnostic is computed for the frozen
load of each step; it is not evidence of stability of the actual nonlinear
acoustic feedback.

Inertia, convection, streaming and heat transport are absent. Report the
radius-based Reynolds estimate and viscous diffusion time. In particular the
outer water phases can invalidate quantitative Stokes timing. An arbitrary
long ramp does not remove the need to test these approximations. Failed
formation is an admissible outcome, not a reason to interpolate a successful
movie. Fixed-command time and fluid-mesh refinement are required before any
convergence or optical-accuracy interpretation of this trajectory.

The pressure-wave movie in notebook 09 remains a separate harmonic carrier
visualization. Notebook 10 presents the unforced state and actual saved
viscous trajectory with the instantaneous *cycle-mean* optical spot.
