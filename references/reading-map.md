# Primary-source reading map

PDFs were copied from the user's Downloads and existing private research folder, or
downloaded from author preprints. `manifest.json` records original locations, DOIs,
bytes and SHA-256 hashes. Local text conversions support equation searches. These
copies are for private study; the repository does not assert redistribution rights.

| Paper | Role in this investigation | Boundary on what we infer |
|---|---|---|
| [Silva-Lora and Torres, JOSA A 37, 1155–1165 (2020)](https://doi.org/10.1364/JOSAA.392795), *Superconical aplanatic ovoid singlet lenses* | Exact signed conjugates, Cartesian-oval implicit and parametric equations, degenerate conic limits | One axial stigmatic diopter is not automatically an aplanatic two-surface singlet; the paper's polar rho is not cylindrical radius |
| [Bertin et al., PRL 109, 244304 (2012)](https://doi.org/10.1103/PhysRevLett.109.244304), *Universal morphologies of fluid interfaces deformed by the radiation pressure of acoustic or electromagnetic waves* | Large-deformation morphology and coupling between propagation and interface shape | Does not demonstrate arbitrary optical-quality surfaces from our array |
| [Chesneau et al., PRE 106, 065104 (2022)](https://doi.org/10.1103/PhysRevE.106.065104), *Numerical simulation of universal morphogenesis of fluid interface deformations driven by radiation pressure* | Radiation tensor and propagation/deformation coupling | The general theory keeps both phases; earlier apparatus-specific solvers have their own declared reductions and do not reproduce this paper's BEM solver |
| [Sisombat et al., Scientific Reports 13, 14703 (2023)](https://doi.org/10.1038/s41598-023-39464-0), *Contactless deformation of fluid interfaces by acoustic radiation pressure* | Experimental transient deformation and capillary response to ultrasonic forcing | Different apparatus; current examples are not fits or reproductions |
| [Gu et al., ACS Nano 14, 14635–14645 (2020)](https://doi.org/10.1021/acsnano.0c03754), *Acoustofluidic Holography for Micro- to Nanoscale Particle Manipulation* | Spatial acoustic patterning, interface response, and streaming considerations | Particle manipulation is not evidence of a cured optical-resin process |
| [Denner, PRE 94, 023110 (2016)](https://doi.org/10.1103/PhysRevE.94.023110), *Frequency dispersion of small-amplitude capillary waves in viscous fluids* | Viscous-dispersion context for the isolated planar verification cases | Neither an inviscid frequency nor a fitted damping constant proves stability of a finite viscous, driven interface |
| [Elgarisi et al., Optica 8, 1501–1506 (2021)](https://doi.org/10.1364/OPTICA.438763), *Fabrication of freeform optical components by fluidic shaping* | Connection from fluid equilibria and supporting frames to cured optical components | Frame/volume shaping is not arbitrary acoustic control; roughness and figure error are different quantities |
| [Na et al., SIGGRAPH Asia (2024)](https://doi.org/10.1145/3680528.3687584), *End-to-end Optimization of Fluidic Lenses* | Links surface formation, optical objectives and manufactured prototypes | Optical optimization and curing calibration do not transfer automatically to an acoustic apparatus |
| [Norland, NOA 61 technical data](https://norlandproducts.com/wp-content/uploads/2025/02/Norland-Products-NOA-61-TDS.pdf) | Liquid index, density, viscosity and surface tension | Manufacturer data, not an academic paper; acoustic speed and attenuation are not supplied |

## Suggested order

For the current general formulation, start with the
[theory manuscript and source map](../docs/theory/README.md). The map records
exactly which ingredients are drawn from each source and which deductions are
made in the manuscript. Additional primary sources acquired for this stage are:

| Source | Role in the theory |
|---|---|
| [Bach and Bruus (2018)](https://doi.org/10.1121/1.5049579) | Viscous acoustic boundary layers and consistent streaming reductions |
| [Joergensen and Bruus (2021)](https://doi.org/10.1121/10.0005005) | Thermoviscous state equations, temperature and mean-flow coupling |
| [Qian, Wang and Sheng (2006)](https://doi.org/10.1017/S0022112006001935) | Moving-contact-line regularization and dissipative variational structure |
| [Giles and Pierce (2000)](https://doi.org/10.1023/A:1011430410075) | General constrained-state adjoints and boundary terms |
| [Luo et al. (2010)](https://doi.org/10.1109/MSP.2010.936019) | Quadratic lifting, rank constraints and semidefinite relaxation |
| [Hubenthal and Onofrei (2016)](https://doi.org/10.1016/j.apnum.2016.03.003) | Distributed Helmholtz boundary control, regularization and source-support sensitivity |
| [Pieper et al. (2020)](https://doi.org/10.1007/s10589-020-00205-y) | Measure-valued source inversion, adjoint observation pairing and singular-source caveats |
| [Boyd et al. (1994)](https://doi.org/10.1137/1.9781611970777) | Lyapunov inequalities, decay rates and state-feedback synthesis |

Finn's *Equilibrium Capillary Surfaces* (1986) and Huh and Scriven's moving-line
paper (1971) supply further background references with the access limits stated
in the source map. Their full texts have not been acquired in this stage.

For the earlier experimental context:

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
  different fields. Mean flow driven by acoustic momentum transfer is streaming. Earlier runs include bulk-absorption forcing, while wall-layer mean streaming remains an open model extension; see the numerical campaign record.
- A transient surface, a relaxed liquid equilibrium, and a cured solid optical
  figure are different outcomes. The active result is an acoustically held liquid asphere.

## Additional leads, not used as calibration

- [Prosperetti, Physics of Fluids 19, 195–203 (1976)](https://doi.org/10.1063/1.861446),
  *Viscous effects on small-amplitude surface waves*. Bibliographic details checked
  against the [author's publication list](https://pages.jh.edu/aprospe1/publ.html);
  full paper not downloaded in this milestone.

The apparatus dimensions are design inputs. The original idealized viscous pair
remains an isolated verification case. NOA 61 has sourced liquid optical/fluid data;
its acoustic properties and the complete apparatus still need calibration.
## 22 September 2026 additions

- **Egarguin, Onofrei and Platt (2018)**, [Sensitivity analysis for the active
  manipulation of Helmholtz fields in 3D](https://arxiv.org/abs/1810.02407).
  Private PDF: `papers/egarguin-2018-source-sensitivity.pdf`. Read the problem
  formulation, source separation discussion and conclusions. Useful for
  continuous-source approximation and placement; not a nonlinear traction or
  two-face reachability theorem.
- The shared research conversation and available associated numerical data are
  private research inputs under `discussions/2026-09-22-cartesian/`. See
  `docs/research-program.md` for acquisition and scope.
