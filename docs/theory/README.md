# Acoustic shaping of custom lens surfaces

Canonical LaTeX source, **version 3.9, 23 September 2026**. The organizing chain
is geometry → dual traction → continuous acoustic synthesis → finite source
realization → formation and stability. The theory now explicitly treats both
faces of one lens, with Cartesian ovoids as a geometric target family and a
10 nm maximum-error specification on each aperture.

- [Compiled manuscript](../../artifacts/theory/acoustic-fluid-shaping-theory.pdf)
- [Main LaTeX source](main.tex)
- [Bibliography](references.bib)
- [Source attribution and reading record](source-map.md)
- [Build provenance](../../artifacts/theory/build-manifest.json)

## Reading guide

| Sections | Content |
|---|---|
| 1–3 | Scope, mechanics, geometric multiplier field, front/back loads, volume constraints and annulus design |
| 4–6 | Waves, distributed sources, weak form, local constructive trace synthesis and a carrier-displacement bound |
| 7–9 | Reachability, coherent finite arrays, the quadratic source integral and placement derivatives |
| 10–13 | Source adjoints, three-dimensional stability, reduced inertial formation diagnostic, formation and feedback |
| 14–18 | Cartesian construction, two-face maximum-error specification, initial numerical evidence, carrier-aware inverse, regimes and unresolved physics |
| Appendices | Supporting derivations and optional optical-performance diagnostics |

The source-to-wave-to-load equations and optimality conditions are reusable for
new targets. Continuation can investigate branches but is not the proposed
inverse definition or a requirement for producing the Cartesian family.

## Build and algebra audit

From the repository root:

```bash
python3 tools/build_theory.py
uv run --with sympy==1.14.0 python tools/verify_theory_algebra.py
```

The PDF needs TeX Live/MacTeX with `latexmk`, pdfLaTeX, biber, biblatex, TikZ
and cleveref. The build rejects unresolved citations/references and overfull
boxes. Auxiliary files stay in `artifacts/theory/build/`; the active PDF and
source checksums are in `artifacts/theory/`. The symbolic audit checks selected
algebraic identities; it performs no physical simulation, source optimization,
or numerical stability experiment. It does not validate the continuum model.

The private PDF library is `references/papers/`, with acquisition provenance
and checksums in `references/manifest.json`. The new source-sensitivity paper
by Egarguin, Onofrei and Platt was recovered from the shared discussion.
The [revision program](../research-program.md) records the discussion's role,
available associated data, and the distinction between reported and reproduced results.

## Scientific status and preservation

Version 3.9 derives the coupled viscous load inverse from a Rayleighian,
including an inertial weak-form compatibility condition for normal-only
actuation. Its first forward implementation is explicitly creeping flow, not
full Navier--Stokes/streaming/thermal dynamics. See [ideal-load evidence](../ideal-load-first.md).
Version 3.8 was archived before replacement under
`2026-09-23-before-viscous-load-variational/theory.tar.gz` in the external archive.

Version 3.8 adds an emitter-independent full-domain load problem in
`sections/08c_ideal_load.tex`: upward traction signs for both interfaces,
pressure gauges, a conditional non-axisymmetric energy-stability result for
ideal fixed spatial loads in a stably stratified graph stack, and the distinction
between static loading and formation histories. No viscous formation simulation
or acoustic stability certificate is claimed. Version 3.7 source and PDF were
archived before replacement in the external revision archive under
`2026-09-23-before-ideal-load-stability/theory.tar.gz`.

Version 3.5 makes the common laboratory conjugate of a stigmatic pair explicit,
derives the normal/direction/detector first variations, and adds a slope-sensitive
source objective and a maximum-spot gate. These are not a formation or stability
certificate. The preceding source and compiled manuscript are preserved in the
separate research archive before replacement.

Version 3.4 derives a flat-domain, two-interface potential-inertia reduction
in `sections/08b_reduced_formation.tex`. Its modal mass is independently checked
against integrated fluid kinetic energy. Notebook 06 shows a short forward
trajectory, not viscous settling or full three-dimensional dynamics. The v3.3
source and active PDF were preserved before editing in the separate archive:
`research-archives/acoustic-freeform-lab/revisions/2026-09-23-before-dual-dynamics/theory.tar.gz`.

This is a research formulation with explicit conditional deductions. It does
not prove global controllability or a stable solution for every Cartesian oval.
The distributed-source adjoint is explicit for the stated inviscid harmonic
transmission subproblem. The general coupled and dynamic adjoints retain
operator form; thermoviscous, wall and moving-interface specializations still
need their detailed boundary and constitutive terms before implementation.

The new `dual` implementation tests simultaneous stationary shaping in a
declared inviscid three-layer model. Its numerical results are recorded in
`artifacts/dual-cartesian-2026-09-22/`; those experiments do not establish physical
10 nm accuracy, three-dimensional stability or formation. Existing single-face
artifacts retain their original model assumptions and remain preserved.

Version 2.0 and the pre-revision working tree were preserved in the separate
research archive before any replacement. The archive path and checksum are
recorded in [the research program](../research-program.md).

Version 1.0, including its LaTeX sources, PDF, original builder and build/review
records, is preserved at
`/Users/blancus/Private/research-archives/acoustic-freeform-lab/theory/v1.0-2026-09-06/`.
Its original PDF SHA-256 is
`1db823159dda34b2f7b10dd9da261b9d1dae72984926c9fe9416afc7a6ffffa1`.
