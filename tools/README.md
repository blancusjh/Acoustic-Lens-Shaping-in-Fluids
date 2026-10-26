# Research utilities

Numerical experiment entry points are research steps in
`src/acoustic_freeform/steps/`, exposed by `afl`. These utilities build
documents, index evidence and check presentations; each folder below holds one
role. Run them from the repository root, for example
`uv run python tools/documents/build_research_index.py --check`. No utility
publishes data or adds a remote.

## Documents and evidence index: `documents/`

| Tool | Role |
|---|---|
| `build_theory.py` | Compile the canonical manuscript and record source/PDF hashes |
| `build_ssl_report.py` | Compile the standalone SSL report; refuses to overwrite a released PDF |
| `verify_theory_algebra.py` | Selected symbolic identities of the manuscript, not physical validation |
| `verify_ssl_report_math.py` | Algebra and finite-difference checks of the SSL report; no simulation |
| `build_research_index.py` | Index preserved reports and render `STATUS.md` from `studies/ledger.json` |

## Study-specific audits: `audits/`

| Tool | Study | Role |
|---|---|---|
| `summarize_maintenance.py` | S01 | Summary of the single-interface maintenance campaign |
| `import_shared_conversation.py` | S01 | Private acquisition of the supplied discussion |
| `audit_imported_discussion.py` | S01 | Audit saved one-interface data; not a dynamics rerun |
| `audit_viscous_runs.py` | S03 | Saved-state energy and refinement audit of ideal-load trajectories |

## Notebooks and browser presentations: `presentation/`

| Tool | Study | Role |
|---|---|---|
| `execute_notebooks.py` | all | Execute source notebooks into their study artifacts |
| `render_readme_figures.py` | S01, S04 | Render the README setup figures and formation animation from saved states |
| `render_s07_figures.py` | S07 | Modelled cell, continuous source, face errors, spots, formation video and pressure cycle from saved states |
| `render_s07_apparatus.py` | S07 | Proposed apparatus from its declared component list, modelled and proposed parts distinguished |
| `present_viscous_formation.py` | S03 | Build the source notebook for saved ideal-load trajectories |
| `verify_viewer.py`, `verify_stationary_viewer.py` | S01 | Browser behavior of the single-interface viewers |
| `verify_formation_notebook.py` | S01 | Surface, error and spot animation synchronization |
| `verify_dual_notebook.py` | S02 | Offline playback and pause of the two-face pressure animation |
| `verify_dual_setup.py` | S02 | Both-interface selection and camera-only rotation |
| `verify_two_face_formation.py` | S02 | Physical-time playback, not camera motion alone |
| `verify_viscous_notebook.py` | S03 | Embedded animation playback and saved-state correspondence |
| `verify_emitter_notebook.py` | S04 | Reproduced equilibrium metrics and pressure-wave playback |
| `verify_emitter_formation_notebook.py` | S05 | Unforced state and saved-state surface/spot playback |
