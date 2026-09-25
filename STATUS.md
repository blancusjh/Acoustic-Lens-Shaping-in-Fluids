# Research status

Generated from [the reviewed evidence ledger](studies/ledger.json) by `python3 tools/build_research_index.py`.
Each status applies only to its declared numerical model and evidence file. No experimental validation is recorded.

| Record | Source | Stationary root | Optical gates | Fixed-command convergence | Formation |
|---|---|---|---|---|---|
| [S01-fixed-drive](studies/S01-single-interface/study.md) | passed | passed | not_applicable | model_limited | model_limited |
| [S02-independent-pair-refinement](studies/S02-independent-two-face/study.md) | passed | not_assessed | failed | failed | not_assessed |
| [S03-ideal-load-formation](studies/S03-stigmatic-target-ideal-load/study.md) | not_applicable | passed | model_limited | not_applicable | model_limited |
| [S04-emitter-design104](studies/S04-emitter-synthesis/study.md) | passed | passed | failed | not_assessed | not_assessed |
| [S05-emitter-formation](studies/S05-emitter-formation/study.md) | passed | not_assessed | not_assessed | not_assessed | failed |

`model_limited` means the reported check exists only within the stated approximation. `not_assessed` is not a failure or a pass. The ideal-load study uses a prescribed traction, so acoustic source and fixed-command fields are `not_applicable`.

## Reviewed verdicts

- **S01-fixed-drive — Single-interface fixed-drive formation and recovery:** Finite-horizon formation and recovery demonstrated in the declared single-interface model. Axisymmetric, isothermal single-interface model with bulk absorption and leading viscous wall loss; numerical evidence only.
- **S02-independent-pair-refinement — Independent two-face target under fixed-command refinement:** Fixed-command refinement fails the height criterion; this prescription is historical evidence. Hypothetical three-fluid cylinder and independently chosen Cartesian patches; the pair is not a shared-conjugate stigmatic singlet.
- **S03-ideal-load-formation — Shared-conjugate pair under prescribed mean load:** The declared mechanical trajectory reaches the numerical optical target; acoustic reachability remains open. Numerical three-fluid axisymmetric Stokes trajectory with prescribed spatial traction and hypothetical viscosities; no acoustic source synthesis.
- **S04-emitter-design104 — NOA 61 / water emitter-driven design-grid equilibrium:** Coupled design-grid root converged but height and ray-coverage gates fail; fixed-command refinement is unfinished. Idealized lossless longitudinal acoustics with passive Robin port loss; hypothetical source boundaries, no thermal or wall-layer mean streaming model.
- **S05-emitter-formation — Partial emitter-driven Stokes startup diagnostic:** Saved trajectory does not form the target; no settling-time or impossibility inference follows. Partial unrefined creeping-flow numerical trajectory; the conservative Reynolds estimate motivates a review stop.

The next theory gates are acoustic reachability under admissible continuous sources and stability of the driven three-dimensional state. Any numerical refinement must hold physical commands fixed before interpreting a new inverse solve. Wall-layer mean streaming and thermal feedback remain open.
