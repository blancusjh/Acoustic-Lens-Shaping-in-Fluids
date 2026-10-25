# Configuration archive

Configurations are grouped by the study whose question they serve, matching
`studies/` and `artifacts/studies/`. All values are SI.

| Folder | Content |
|---|---|
| `shared/` | Material reference used by several studies (`noa61-water-pressure-reference.json`) |
| `S01-single-interface/` | `lenses/*.toml` and `initialization/*.json` for the single-interface model (`afl single-interface`, `lenslab`) |
| `S02-independent-two-face/` | Independent Cartesian-pair campaign, inertial formation steps and `precision/` inverse and fixed-command inputs |
| `S03-stigmatic-target-ideal-load/` | Shared-conjugate optical audit, required load and ideal-load viscous formation |
| `S04-emitter-synthesis/` | NOA 61 / water emitter synthesis, held-command checks and `precision/` shared-conjugate (`stigmatic-*`) and audit inputs |
| `S05-emitter-formation/` | Emitter-driven viscous formation diagnostic |
| `S06-two-face-physical-checks/` | Surrogate held-command state, fixed-state force studies, stability and formation time-step checks |
| `verification/` | Planar-interface verification limits (`afl planar validate`) |

The files are immutable inputs to preserved experiments; they were moved into
these folders on 2026-10-24 without changing their bytes. Similar files
intentionally retain their actual resolved values so older results can be
reproduced without reconstructing a chain of overrides. New campaigns should
keep shared machine/material/prescription choices explicit in their input
record and write the resolved SI configuration beside each output.

`relocations.json` maps old paths to current ones: `artifacts/<campaign>/...`
to `artifacts/studies/<study>/...` or the separate research archive, and the
former `configs/dual/`, `configs/lenses/` and `configs/initialization/` files to
their study folders. `acoustic_freeform.core.paths.data_path` reads that map, so
historical configurations, saved artifact configs that name old paths, and
`afl <step> <old path>` keep working. New configurations should use current
paths directly.

One historical follow-up,
`S04-emitter-synthesis/precision/stigmatic-conic-coupled-448.json`, names a missing
`coupled-observations-448/observation-operator.npz`. Its upstream coupled
observation output must be produced and verified before that configuration can
run. An unrelated observation operator is not an interchangeable input.
