# README figures

Rendered by `tools/presentation/render_readme_figures.py` from saved local states under
`artifacts/studies/`. `provenance.json` records the SHA-256 of every input and
output file, plus the consistency checks run while rendering. Nothing is
interpolated toward a target shape; every drawn part is a declared apparatus
component or a saved numerical state. None is experimental evidence.

| Figure | Source state | Checks while rendering |
|---|---|---|
| `single-interface-cell.png` | S01 `formation-notebook-2026-09-23/dt000625`, final state at 0.2 s | Display ray trace RMS spot agrees with the saved analysis (0.059 vs 0.054 µm) |
| `interface-formation-and-spots.gif` | The same run: all 321 saved states via `notebooks/05_formation_and_spots/formation-analysis-si.npz` | The held command is constant over the run; playback speed is printed on every frame |
| `two-face-cell.png` | S04 `noa61-emitter-2026-09-23/verify-448-c4-clear-spot/design104` | Saved spots are reproduced by a fresh ray trace; a fresh harmonic solve reproduces the saved source power |
| `s09-formation-two-tones.gif` | S09 `atlas7-2band-verification-2026-10-26/formation-dt00125`, all 161 saved states (0–2 s), rendered by `render_showcase.py`; provenance in `s09-formation-provenance.json` | The trajectory starts at the acoustics-off meniscus; final errors and spot equal the study's (0.12 / 0.61 nm, 0.125 µm); faces are drawn with their deviation magnified ×10 and labelled so |

The single-interface rays start at an object 50 mm below the vertex, with the
incident wave prescribed inside the resin (window refraction is not modeled).
The pressure colour scale in `two-face-cell.png` saturates at the 99.5th
percentile; the peak value is printed on its colour bar.
