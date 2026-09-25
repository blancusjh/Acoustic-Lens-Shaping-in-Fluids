# Primary sources and derivation provenance

## September 2026 revision

The user-supplied [research discussion](https://chatgpt.com/share/6ab2119a-197c-83e9-9801-223a8b8ea45e)
motivates the geometric multiplier field, continuous source integral, source
placement, annulus design and maximum geometric-error specification. It is a
research input, not a primary publication or independent validation. The
two-face constraint bookkeeping and instantaneous-error budget are explicit
extensions in this manuscript.

The added [Egarguin–Onofrei–Platt manuscript](https://arxiv.org/abs/1810.02407)
was downloaded and its formulation, source-distance discussion and conclusions
checked. It treats active linear Helmholtz fields in prescribed regions, not
the nonlinear acoustic lens problem. Kowalevsky's original 1875 publication is
cited for the analytic Cauchy theorem; its bibliographic record was checked,
but no claim is made to have audited the historical proof in full.

The local pressure-node/cutoff construction is conditional on analytic patches
and nearby two-sided volume-force access. The source-free wall-only inverse
and simultaneous two-face boundary control do not follow from that result.
The saved single-interface package from the discussion is recorded separately
in `docs/research-program.md`; its assertions are not promoted to two-face evidence.

The manuscript's bibliography is `references.bib`. The private acquisition
manifest records the origin and checksum of available PDFs. This record
distinguishes published ingredients from deductions made in this formulation;
none of the cited authors is claimed to have solved the complete inverse
acoustic Cartesian-lens problem presented here.

| Source | Material checked and used | Scope of attribution |
|---|---|---|
| [Silva-Lora and Torres (2020)](https://doi.org/10.1364/JOSAA.392795) | User-provided published article; Cartesian implicit and parametric equations, form parameters, exact refraction and conjugates | Optical construction. The general two-fluid mechanics, analytic required-traction expression, array feasibility and control formulation are not attributed to this paper. Its polar distance is distinguished from cylindrical radius. |
| [Chesneau et al. (2022)](https://doi.org/10.1103/PhysRevE.106.065104) | Full author manuscript; model, wave/interface coupling and radiation tensor | The displayed momentum flux is the negative of their radiation-stress convention. Their assumptions do not become universal apparatus conditions. |
| [Bertin et al. (2012)](https://doi.org/10.1103/PhysRevLett.109.244304) | Local published PDF; deformation morphology and coupling evidence | Establishes observed wave-driven deformation, not arbitrary controllability or our optical accuracy. |
| [Sisombat et al. (2023)](https://doi.org/10.1038/s41598-023-39464-0) | Local published PDF; ultrasonic deformation and transient observations | Motivation for a dynamic interface model; not a calibration of an unspecified setup. |
| [Bach and Bruus (2018)](https://doi.org/10.1121/1.5049579) | Full author manuscript; first-order acoustics, boundary-layer assumptions and second-order streaming structure | Conditional viscous boundary-layer reductions, with geometric and scale restrictions. |
| [Joergensen and Bruus (2021)](https://doi.org/10.1121/10.0005005) | Full author manuscript; mass, momentum, thermodynamics and perturbation hierarchy | Need for consistent acoustic, thermal and streaming treatment. No universal thermal correction is inferred. |
| [Denner (2016)](https://doi.org/10.1103/PhysRevE.94.023110) | Full author manuscript; viscous capillary-wave dispersion and inviscid limiting frequency | Supports retaining inertia and distinguishing viscous regimes; not a cavity-specific damping law. |
| [Finn (1986)](https://doi.org/10.1007/978-1-4613-8584-4) | Publisher bibliographic record and subject scope; book not acquired or read in full | Background reference for equilibrium capillarity. The manuscript supplies its own first-variation derivation rather than claiming a page-specific theorem was checked in this book. |
| [Huh and Scriven (1971)](https://doi.org/10.1016/0021-9797(71)90188-3) | Publisher abstract and bibliographic record, cross-checked against Qian et al. | Classical moving-contact-line/no-slip singularity. No detailed equation is copied from inaccessible full text. |
| [Qian, Wang and Sheng (2006)](https://doi.org/10.1017/S0022112006001935) | Full author manuscript; introduction and variational/dissipation framework | Moving-line regularization and thermodynamic consistency. The simple line-friction law in the manuscript is explicitly an illustrative closure, not asserted to be their full generalized Navier boundary condition. |
| [Giles and Pierce (2000)](https://doi.org/10.1023/A:1011430410075) | Full author-hosted article; sections 2.1–2.6 and continuous-adjoint boundary-term discussion | State-constrained sensitivity and adjoint methodology. The present sign convention follows its explicitly defined plus-sign Lagrangian. |
| [Luo et al. (2010)](https://doi.org/10.1109/MSP.2010.936019) | Full author-hosted manuscript; quadratic lifting, removal of the rank constraint, and recovery interpretation | Semidefinite relaxation method. Its application to acoustic traction and the coherence caveats are derived in this manuscript. |
| [Boyd et al. (1994)](https://doi.org/10.1137/1.9781611970777) | Author-hosted book; Lyapunov inequalities, decay-rate and state-feedback LMI sections | Conditional finite-dimensional stability certificates. The complete coupled free-boundary problem is not thereby convex. |
| [Elgarisi et al. (2021)](https://doi.org/10.1364/OPTICA.438763) | Local author manuscript, published metadata; fluidic shaping and solidification | A distinct fabrication and retention route, not evidence of unforced acoustic shape persistence. |
| [Hubenthal and Onofrei (2016)](https://doi.org/10.1016/j.apnum.2016.03.003) | Author manuscript; abstract, section 2 operator formulation, compactness/range discussion and equations (2.9)–(2.11); journal metadata checked on arXiv | Regularized active Helmholtz field control, support separation and sensitivity. Their exterior-domain field approximation is not a controllability theorem for nonlinear fluid shapes in a cylinder. |
| [Pieper, Tang, Trautmann and Walter (2020)](https://doi.org/10.1007/s10589-020-00205-y) | Author manuscript; source model (1.1), very weak formulation (2.4), Theorem 2.1, source/observation separation, section 3 adjoint and Proposition 3.4; journal metadata checked on arXiv | Measure-valued source inversion requires singular-source regularity and an appropriate observation pairing. Its pressure-recovery results are not transferred to our quadratic traction inverse. |

The following are deductions supplied in the manuscript, using standard
variational, Hilbert-space and control identities; no novelty priority is claimed:

- The source weak form and its power identity under the declared phase, source
  and normal conventions.
- Continuous quadratic traction operators, necessary sign/effort bounds, the
  conditional compactness obstruction, regularized existence and admissible
  source-space approximation proofs.
- The exact zero-source Hessian and amplitude minimization along a source
  direction for the projected penalized objective.
- The explicit interface covectors, adjoint pressure/flux jumps and spatial
  boundary/body-force/dilation gradients for the frozen transmission problem.
- Angular-order difference selection, source-phase nonuniqueness, and the
  finite-channel local rank bound.
- The conditional unforced-retention obstruction, full-base-flow stability
  requirements and local Lyapunov remainder estimate.
- Cartesian path-Hessian curvature and required traction, and the conditional
  target-to-source selection within the general fluid problem.

The new compactness and existence results state their functional assumptions;
none proves the regularity or global solvability of the full thermoviscous
free-boundary system. The explicit acoustic adjoint is derived by Green's
identity, including its interface terms. Its extension to the full coupled
problem remains an identified theoretical step.

The manuscript does not reproduce paper figures. Its dependency diagram is an
original vector diagram, not a proposed experimental apparatus. The algebra
audit is an exact symbolic check of selected equations, not CFD verification
or experimental validation.
