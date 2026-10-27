# Research status

Generated from [the reviewed evidence ledger](studies/ledger.json) by `python3 tools/documents/build_research_index.py`.
Each status applies only to its declared numerical model and evidence file. No experimental validation is recorded.

| Record | Source | Stationary root | Optical gates | Fixed-command convergence | Formation |
|---|---|---|---|---|---|
| [S01-fixed-drive](studies/S01-single-interface/study.md) | passed | passed | not_applicable | model_limited | model_limited |
| [S02-independent-pair-refinement](studies/S02-independent-two-face/study.md) | passed | not_assessed | failed | failed | not_assessed |
| [S03-ideal-load-formation](studies/S03-stigmatic-target-ideal-load/study.md) | not_applicable | passed | model_limited | not_applicable | model_limited |
| [S04-emitter-design104](studies/S04-emitter-synthesis/study.md) | passed | passed | failed | not_assessed | not_assessed |
| [S05-emitter-formation](studies/S05-emitter-formation/study.md) | passed | not_assessed | not_assessed | not_assessed | failed |
| [S06-surrogate](studies/S06-two-face-physical-checks/study.md) | passed | passed | failed | failed | failed |
| [S07-meniscus-750khz](studies/S07-faithful-modal-synthesis/study.md) | passed | passed | failed | passed | model_limited |
| [S08-fig9b-tuned](studies/S08-published-singlet/study.md) | passed | passed | failed | passed | model_limited |
| [S09-common-machine-fig9b](studies/S09-universal-source/study.md) | passed | passed | passed | passed | model_limited |

`model_limited` means the reported check exists only within the stated approximation. `not_assessed` is not a failure or a pass. The ideal-load study uses a prescribed traction, so acoustic source and fixed-command fields are `not_applicable`.

## Reviewed verdicts

- **S01-fixed-drive — Single-interface fixed-drive formation and recovery:** Finite-horizon formation and recovery demonstrated in the declared single-interface model. Axisymmetric, isothermal single-interface model with bulk absorption and leading viscous wall loss; numerical evidence only.
- **S02-independent-pair-refinement — Independent two-face target under fixed-command refinement:** Fixed-command refinement fails the height criterion; this prescription is historical evidence. Hypothetical three-fluid cylinder and independently chosen Cartesian patches; the pair is not a shared-conjugate stigmatic singlet.
- **S03-ideal-load-formation — Shared-conjugate pair under prescribed mean load:** The declared mechanical trajectory reaches the numerical optical target; acoustic reachability remains open. Numerical three-fluid axisymmetric Stokes trajectory with prescribed spatial traction and hypothetical viscosities; no acoustic source synthesis.
- **S04-emitter-design104 — NOA 61 / water emitter-driven design-grid equilibrium:** Coupled design-grid root converged but height and ray-coverage gates fail; fixed-command refinement is unfinished. Idealized lossless longitudinal acoustics with passive Robin port loss; hypothetical source boundaries, no thermal or wall-layer mean streaming model.
- **S05-emitter-formation — Partial emitter-driven Stokes startup diagnostic:** Saved trajectory does not form the target; no settling-time or impossibility inference follows. Partial unrefined creeping-flow numerical trajectory; the conservative Reynolds estimate motivates a review stop.
- **S06-surrogate — Physical checks of a converged two-interface emitter surrogate:** Linearly stable but neither robust nor reachable: the design-grid success does not survive the surface space, the back face is not wave-resolved to 10 nm, and the target is not the state reached from flat. Axisymmetric, lossless acoustics with ideal velocity sources, creeping flow; numerical evidence only.
- **S07-meniscus-750khz — Faithful modal synthesis from the acoustics-off meniscus:** Formed from the acoustics-off meniscus, converged and stable within 1 W / 2 MPa, but the optical gates fail (19.6 / 11.3 nm against 10 nm; spot 7.2 um). Axisymmetric, lossless acoustics with ideal velocity sources, creeping flow; numerical evidence only.
- **S08-fig9b-tuned — Published Fig. 9b singlet in a tuned-liquid cell:** Formed from the meniscus, converged and stable, near diffraction-limited; the front face misses 10 nm (27 nm). Its claim that water cannot reach Fig. 9b is withdrawn by S09. Axisymmetric, lossless acoustics with ideal velocity sources, creeping flow; numerical evidence only. The liquid set is an assumption.
- **S09-common-machine-fig9b — Fig. 9b on the common machine (one of three fully checked designs):** One fixed water / NOA 61 / water machine holds all 68 fitting family singlets (convex bound below 0.5 nm); 67 form from rest within 1 nm with one or two tones. Fig. 9b, Fig. 7a (fold margin) and lens 7 pass refinement, coupled stability and formation within 1 nm; lens 57 is a start-up path failure. Axisymmetric, lossless acoustics with ideal velocity sources, creeping flow; numerical evidence only.

Within the declared model, S09 answers source reachability for a family of stigmatic singlets on one fixed machine. Open: start-up paths (S09 lens 57), the full chain on more designs and reduced channel counts, stability of the driven three-dimensional state, wall-layer mean streaming, thermal feedback, measured material constants, transducer realization and curing. Any numerical refinement must hold physical commands fixed before interpreting a new inverse solve.
