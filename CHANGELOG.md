# Changelog

All notable changes to this project. Results are numerical and scoped to their
declared models; see [STATUS.md](STATUS.md).

## 0.1.0 — first version (2026-10-27)

### Added
- `afl design` and `acoustic_freeform.design`: a desired stigmatic singlet in,
  the settings of the fixed machine out (pin heights, fill volumes, per-channel
  drive command), with a status that states what was established
  ([docs/design-a-lens.md](docs/design-a-lens.md)).
- The fixed machine declared once in `configs/machine/v1.json`.
- Theory section 10f: load demand, the conic matching rule, convex capability
  of a fixed source, pressure and fold-margin constraints, several bands.
- Study S09: one machine for the Silva-Lora and Torres family; 67 of the 68
  lenses that fit form from rest within 1 nm; three designs fully checked;
  channel study; stage verdict.
- `acoustics.radiation`: Kirchhoff-Helmholtz sound radiated to an outside listener.
- Test suite with closed-form checks and CI (lint, tests, status page).
- Presentation: formation showcase, story video and sonified soundtracks.

### Changed
- `capability-synthesis` split into build, command, settle and write steps
  (same results), shared with `afl design`.
- `cvxpy` is a core dependency.
- The status builder reports evidence files kept in the research archive
  instead of failing; the ledger covers S01–S09.

### Corrected
- S08: the claim that water cannot reach Fig. 9b was a local minimum (S09).
- S09: a first pass of the atlas used degenerate wave meshes; the affected
  claims were withdrawn and the mesh now merges near-duplicate nodes.
