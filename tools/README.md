# Research utilities

Numerical experiment entry points live with their declared Python model and
are exposed by `afl`; these utilities build documents, index evidence and check
presentations. No utility publishes data or adds a remote.

## Documents and evidence index

| Tool | Role |
|---|---|
| `build_theory.py` | Compile the canonical manuscript and record source/PDF hashes |
| `build_ssl_report.py` | Compile the standalone SSL report; refuses to overwrite a released PDF |
| `verify_theory_algebra.py` | Selected symbolic identities of the manuscript, not physical validation |
| `verify_ssl_report_math.py` | Algebra and finite-difference checks of the SSL report; no simulation |
| `build_research_index.py` | Index preserved reports and render `STATUS.md` from `studies/ledger.json` |

## Study-specific audits

| Tool | Study | Role |
|---|---|---|
| `summarize_maintenance.py` | S01 | Summary of the single-interface maintenance campaign |
| `import_shared_conversation.py` | S01 | Private acquisition of the supplied discussion |
| `audit_imported_discussion.py` | S01 | Audit saved one-interface data; not a dynamics rerun |
| `audit_viscous_runs.py` | S03 | Saved-state energy and refinement audit of ideal-load trajectories |

## Notebooks and browser presentations

| Tool | Study | Role |
|---|---|---|
| `execute_notebooks.py` | all | Execute source notebooks into their study artifacts |
| `render_readme_figures.py` | S01, S04 | Render the README setup figures and formation animation from saved states |
| `present_viscous_formation.py` | S03 | Build the source notebook for saved ideal-load trajectories |
| `verify_viewer.py`, `verify_stationary_viewer.py` | S01 | Browser behavior of the single-interface viewers |
| `verify_formation_notebook.py` | S01 | Surface, error and spot animation synchronization |
| `verify_dual_notebook.py` | S02 | Offline playback and pause of the two-face pressure animation |
| `verify_dual_setup.py` | S02 | Both-interface selection and camera-only rotation |
| `verify_two_face_formation.py` | S02 | Physical-time playback, not camera motion alone |
| `verify_viscous_notebook.py` | S03 | Embedded animation playback and saved-state correspondence |
| `verify_emitter_notebook.py` | S04 | Reproduced equilibrium metrics and pressure-wave playback |
| `verify_emitter_formation_notebook.py` | S05 | Unforced state and saved-state surface/spot playback |
