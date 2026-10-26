# Proposed apparatus for the S07 design

Nothing here has been built or measured. The component list is declared in
`configs/S07-faithful-modal-synthesis/apparatus-proposal.json`, and the figure
is rendered from it by `tools/presentation/render_s07_apparatus.py`
(`artifacts/.../figures/s07-apparatus.png`). **Modelled** parts are in the S07
numerical model with the stated values; **proposed** parts are needed to build
it but are not yet represented in any simulation.

## Cell and liquids (modelled)

| Item | Value |
|---|---|
| Bore | 8 mm diameter, 6 mm high, axis vertical |
| Layers, bottom to top | water 79.48 µL, NOA 61 (uncured) 123.10 µL, water 99.02 µL; 301.59 µL in total |
| Contact lines | pinned at the wall at z = −1 and +1 mm |
| Acoustics-off faces | displaced volumes −21.05 µL (front) and +1.515 µL (back) relative to flat |
| Drive | 0.75 MHz; 112 complex channel amplitudes from 20 continuous modes |
| Source surfaces | 32 rings per window and 48 wall bands, all 0.125 mm pitch |
| Peak source motion | 0.345 / 0.146 / 0.107 m/s (lower window / upper window / wall), i.e. 73 / 31 / 23 nm |
| Field | 0.55 W ideal source power, 0.37 MPa sampled peak pressure |
| Optics | on-axis object at z = −101 mm, image at z = +151 mm, clear apertures 4 mm |

## Hardware to realize it (proposed)

| Component | Purpose | Not yet modelled |
|---|---|---|
| Transparent transducer windows with 32 concentric transparent ring electrodes | Drive both window source distributions while passing light | Window thickness and refraction, electrode-gap diffraction, piezoelectric coupling; transparent-transducer precedent still to be referenced |
| Segmented piezoelectric tube, 48 electrode bands | Drive the wall source distribution | Inter-band coupling, tube modes |
| Backing and matching layers | Approximate the modelled impedance-matched source boundaries | Per-channel calibration from voltage to velocity |
| Pinning edges at z = ±1 mm | Hold both contact lines | Contact-line hysteresis and wetting |
| Fill and vent ports with syringe pumps | Set the three volumes | Dispensing error: the vertex moves about 40 µm per µL |
| Temperature-controlled enclosure | Hold the liquids at constant temperature | Thermal expansion: about 0.8 µm/K of front-face motion from the lower water layer alone |
| 112-channel continuous-wave transmitter | Amplitude and phase per channel at 0.75 MHz | Electrical drive, heating of transducers |
| Pinhole source, image-plane camera, wavefront sensor | Measure the spot and surface shape | Imaging of the real optical train |
| Control computer | Close the loop on command and volume trim | Feedback law |
| UV source (optional) | Cure the lens if it is to be solidified | Shrinkage and heating during curing |

## Requirements that follow

- **Volume:** with about 40 µm of vertex motion per µL, the fill cannot be set
  to 10 nm by dispensing (that would need about 0.25 nL in 100 µL). The design
  needs a final trim measured optically, either by a small volume actuator or
  by the acoustic command.
- **Temperature:** sound-speed sensitivity alone needs about ±0.4 K for 10 nm
  (S07 study), but thermal expansion of the lower water layer moves the front
  face by about 0.8 µm/K. That estimate ignores the resin and the cell, which
  partly compensate; it is not modelled. Taken at face value it needs about
  10 mK stability or closed-loop correction.
- **Interfacial tension:** a 1 % change moves the front face by 0.3 µm in
  this gravity-loaded meniscus; it must be measured and trimmed, or gravity
  removed by density matching.
- **Channels:** 112 independent amplitude-and-phase outputs. A smaller count is
  possible only after a port-count convergence study of the modal source.
