# From fluidic shaping to acoustic shaping

Theory paper (LaTeX source): *From fluidic shaping to acoustic shaping: the
load that produces a prescribed liquid surface*. Draft 1, 27 October 2026.

It starts from the fluidic-shaping method of Bercovici's group (Frumkin and
Bercovici, Flow 2021; Elgarisi et al., Optica 2021). It poses the extension as
an inverse problem: which normal traction holds a prescribed surface (the
*load demand*). It covers the series form, the conic matching rule
Δρg/σ = 4kc² and two-face lenses. It then proposes acoustic radiation stress
as the actuator: Hermitian forces, the multi-tone convex cone, the
semidefinite capability program with its dual certificate and rank bound,
pressure and fold-margin LMIs, start-up reachability and creeping-flow
stability. The paper is theory in a declared axisymmetric, bulk-lossless,
ideal-source, creeping-flow reduction. Nothing in it has been built or
measured. Numbers quoted from `studies/S09-universal-source/study.md` are
illustrations only.

- Source: `main.tex` (single file, same preamble and macros as
  `docs/theory/main.tex`)
- Bibliography: the shared `docs/theory/references.bib`, loaded as
  `\addbibresource{../../theory/references.bib}`

## Build

With TeX Live or MacTeX (pdfLaTeX, biber, biblatex, TikZ, cleveref), from
this directory:

```bash
latexmk -pdf -outdir=../../../artifacts/papers/fluidic-to-acoustic-shaping main.tex
```

`latexmk` runs biber automatically. Tectonic's biber step rejects `../`
paths, so a Tectonic build needs a copy of `main.tex` whose
`\addbibresource` points to a local copy of `references.bib`.

Compiled PDFs are private. They go to `artifacts/papers/fluidic-to-acoustic-shaping/`
and are never committed or published.
