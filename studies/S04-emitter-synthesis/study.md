# S04 — Emitter synthesis for the stigmatic target

**Question.** Can a bounded finite source array generate the required
full-interface traction and pass coupled optical checks?

**Protocol.** Fit admissible complex source velocities, solve the complete
wave and both nonlinear surface force balances at the held command, and trace
all launch rays. Refine the wave and surface discretizations without changing
the command before reporting accuracy. Finite arrays restrict the continuous
source space of the [general theory](../../docs/theory/README.md).

**Verdict.** [S04-emitter-design104](../../STATUS.md) converges as a coupled
design-grid root but fails the joint optical gates; fixed-command refinement
is unfinished. The source boundaries and material response remain idealized.
See the [case audit](../../artifacts/studies/S04-emitter-synthesis/noa61-emitter-2026-09-23/README.md)
and [saved validation](../../artifacts/studies/S04-emitter-synthesis/noa61-emitter-2026-09-23/verify-448-c4-clear-spot/design104/validation.json).
