# S01 — Single-interface fixed-drive formation and recovery

**Question.** Can a declared fixed source command form a Cartesian diopter and
recover after disturbances in the earlier single-interface model?

**Protocol.** Hold the physical command under temporal and spatial refinement;
compare the entire early trajectory, final surface, optical ray metric, volume
and disturbance recovery. This model includes axisymmetric inertia, bulk
absorption and leading viscous acoustic wall loss.

**Verdict.** [S01-fixed-drive](../../STATUS.md) records a positive finite-horizon
numerical result in that model. It does not assess three-dimensional streaming
base-flow stability, thermal feedback or an experimental apparatus. The
[maintenance audit](../../artifacts/studies/S01-single-interface/cartesian-maintenance-2026-09-06/verification/maintenance.json)
and [model limitations](../../docs/numerical-stability.md) specify the scope.
