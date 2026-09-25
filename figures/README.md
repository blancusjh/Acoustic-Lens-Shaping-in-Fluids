# README figure provenance

These are presentation copies of saved numerical visualizations. The original
study outputs remain under the local `artifacts/studies/` tree with their
configurations, provenance and validation reports. None is experimental
validation or a photograph of built hardware.

| Figure | Original local artifact | SHA-256 |
|---|---|---|
| `modeled-single-interface-apparatus.png` | `artifacts/studies/S01-single-interface/notebooks/05_formation_and_spots/apparatus-3d.png` | `fbce80be8f11a3a0363357caf97e6612fc99db8ef098043cf41657cbf5a37cf5` |
| `interface-formation-and-spots.gif` | 28 sampled plot frames from `artifacts/studies/S01-single-interface/notebooks/05_formation_and_spots/formation-and-spots.html` | `8901b7fe2d36e06bec446f039b0bcd350025f97b6cf8f4411fe4b1b54ab21fd7` |
| `modeled-two-face-cell.png` | `artifacts/studies/S01-single-interface/notebooks/05_formation_and_spots/two-face-setup/both-interfaces-3d.png` | `f3d860ee14a03e2615cd8471c3261929b7aeab2fe4cc8b44bd7c4b02d3e3fc53` |

The GIF samples saved fine-run frames with indices `0, 2, ..., 40, 48, 56, 64,
80, 96, 120, 160` from the 161-frame standalone animation, resizes them to
1100 pixels wide and converts them to a 128-color GIF. Each frame retains its
original plotted geometry, spot diagram and physical-time label. The local
`artifacts/studies/S01-single-interface/notebooks/05_formation_and_spots/validation-report.md`
states the run's scope and limits; the public [formation notes](../docs/formation-notebook.md)
summarize it. The two-face image comes from a separate stationary
reconstruction; it is not a frame of that GIF.
