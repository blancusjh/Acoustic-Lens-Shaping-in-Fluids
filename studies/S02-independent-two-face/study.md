# S02 — Independent two-face Cartesian patches

**Question.** Does a source fit to two separately prescribed Cartesian patches
remain accurate when the physical command is held and the coupled surface and
wave discretizations change?

**Protocol.** First record the source command; then solve the forward coupled
equilibrium on independent grids, with both phase-volume constraints and both
interfaces retained. Report surface and optical errors separately from frozen
inverse residuals. The two patches in this study were not constructed as a
shared-conjugate stigmatic singlet.

**Verdict.** [S02-independent-pair-refinement](../../STATUS.md) fails the fixed
command height check. This is evidence of inverse discretization sensitivity,
not a certificate that arbitrary acoustic sources cannot reach other targets.
See the [fixed-command report](../../artifacts/studies/S02-independent-two-face/dual-precision-2026-09-23/verify-mean-448-surface-refinement/report.md).
The [historical LaTeX evidence section](historical-numerical-evidence.tex) was
removed from the general theory manuscript and retained with this study.
