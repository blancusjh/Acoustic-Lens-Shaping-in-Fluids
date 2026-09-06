# Primary-source reading map

PDFs were copied from the user's Downloads and existing private research folder, or
downloaded from author preprints. `manifest.json` records original locations, DOIs,
bytes and SHA-256 hashes. Local text conversions support equation searches. These
copies are for private study; the repository does not assert redistribution rights.

| Paper | Role in this investigation | Boundary on what we infer |
|---|---|---|
| [Bertin et al., PRL 109, 244304 (2012)](https://doi.org/10.1103/PhysRevLett.109.244304), *Universal morphologies of fluid interfaces deformed by the radiation pressure of acoustic or electromagnetic waves* | Large-deformation morphology and coupling between propagation and interface shape | Does not demonstrate arbitrary optical-quality surfaces from our array |
| [Chesneau et al., PRE 106, 065104 (2022)](https://doi.org/10.1103/PhysRevE.106.065104), *Numerical simulation of universal morphogenesis of fluid interface deformations driven by radiation pressure* | Radiation tensor and propagation/deformation coupling | Our finite chamber uses curved Helmholtz/Stokes FEM with a one-fluid pressure-release approximation; it does not reproduce their two-fluid BEM solver |
| [Sisombat et al., Scientific Reports 13, 14703 (2023)](https://doi.org/10.1038/s41598-023-39464-0), *Contactless deformation of fluid interfaces by acoustic radiation pressure* | Experimental transient deformation and capillary response to ultrasonic forcing | Different apparatus; current examples are not fits or reproductions |
| [Gu et al., ACS Nano 14, 14635–14645 (2020)](https://doi.org/10.1021/acsnano.0c03754), *Acoustofluidic Holography for Micro- to Nanoscale Particle Manipulation* | Spatial acoustic patterning, interface response, and streaming considerations | Particle manipulation is not evidence of a cured optical-resin process |
| [Denner, PRE 94, 023110 (2016)](https://doi.org/10.1103/PhysRevE.94.023110), *Frequency dispersion of small-amplitude capillary waves in viscous fluids* | Viscous-dispersion context for the isolated planar verification cases | The finite lens uses creeping flow, not the earlier weak-damping closure |
| [Elgarisi et al., Optica 8, 1501–1506 (2021)](https://doi.org/10.1364/OPTICA.438763), *Fabrication of freeform optical components by fluidic shaping* | Connection from fluid equilibria and supporting frames to cured optical components | Frame/volume shaping is not arbitrary acoustic control; roughness and figure error are different quantities |
| [Na et al., SIGGRAPH Asia (2024)](https://doi.org/10.1145/3680528.3687584), *End-to-end Optimization of Fluidic Lenses* | Links surface formation, optical objectives and manufactured prototypes | Optical optimization and curing calibration do not transfer automatically to an acoustic apparatus |
| [Norland, NOA 61 technical data](https://norlandproducts.com/wp-content/uploads/2025/02/Norland-Products-NOA-61-TDS.pdf) | Liquid index, density, viscosity and surface tension | Manufacturer data, not an academic paper; acoustic speed and attenuation are not supplied |

## Suggested order

Read Sisombat for an experimentally tangible transient first. Use Chesneau's
Sections II–III for the governing acoustic stress and coupling logic, then compare
Bertin's morphology evidence. Read Denner before choosing a viscous wave closure.
Use Gu to identify spatial-control and streaming issues. Return to Elgarisi and Na
when finite resin geometry and final optical quality become the modeling target.

## Equation conventions to keep straight

- Chesneau writes acoustic-cycle averages; our code stores complex **peak** phasors.
  Consequently `mean(p'^2) = |P|²/2`. The implemented normal momentum flux has
  coefficients `1/4`, not `1/2`, in its pressure and velocity terms.
- A transmitted pressure coefficient is not a transmitted energy fraction; include
  normal admittance when comparing acoustic power.
- Fast acoustic particle velocity and slow interface-driven fluid velocity are
  different fields. Acoustic streaming is a third phenomenon and currently omitted.
- A transient surface, a relaxed liquid equilibrium, and a cured solid optical
  figure are different outcomes. The active result is an acoustically held liquid asphere.

## Additional leads, not used as calibration

- [Prosperetti, Physics of Fluids 19, 195–203 (1976)](https://doi.org/10.1063/1.861446),
  *Viscous effects on small-amplitude surface waves*. Bibliographic details checked
  against the [author's publication list](https://pages.jh.edu/aprospe1/publ.html);
  full paper not downloaded in this milestone.
- [Jørgensen and Bruus, JASA 149, 3599–3610 (2021)](https://doi.org/10.1121/10.0005005),
  thermal and streaming modeling for a later scale analysis.

The apparatus dimensions are design inputs. The original idealized viscous pair
remains an isolated verification case. NOA 61 has sourced liquid optical/fluid data;
its acoustic properties and the complete apparatus still need calibration.
