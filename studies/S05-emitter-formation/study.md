# S05 — Emitter-driven formation diagnostic

**Question.** Does the saved emitter command actually form the target from
the unforced flat state in the declared fluid model?

**Protocol.** Advance a three-fluid moving-interface Stokes model under the
saved source command. Record physical time, both surface states, the fixed
detector spot and a Reynolds estimate at every saved frame. Stop when the
model assumptions no longer support quantitative interpretation.

**Verdict.** [S05-emitter-formation](../../STATUS.md) is a partial negative
diagnostic for this command. It neither proves physical impossibility nor a
reliable settling time. See the [trajectory report](../../artifacts/studies/S05-emitter-formation/noa61-emitter-2026-09-23/viscous-formation/report.md)
and [protocol](../../docs/emitter-formation-protocol.md).
