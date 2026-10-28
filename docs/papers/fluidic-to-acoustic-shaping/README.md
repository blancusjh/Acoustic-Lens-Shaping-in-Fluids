# Fluidic shaping with a custom pressure field

Theory paper (LaTeX source): *Fluidic shaping with a custom pressure field: the
load that produces a prescribed liquid surface, and how sound can apply it*.
Draft 2, 27 October 2026.

The paper extends the fluidic-shaping method of Bercovici's group (Frumkin and
Bercovici, Flow 2021; Elgarisi et al., Optica 2021) with one more control: a
spatially distributed pressure field applied to the interface. The physical
model is the fluidic-shaping model unchanged (energy, Young–Laplace balance,
pinned frame, volume constraints) plus that one term.

It is written as a physics paper, without a theorem structure:
- problem statement for the cell;
- the loaded Young–Laplace balance, with orders of magnitude for NOA 61 in
  water;
- what fluidic shaping reaches without the field;
- what the field adds: the inverse problem (*load demand*), the series near
  the axis, the conic matching rule Δρg/σ = 4kc² and its remainder, and both
  faces at once;
- sound as the way to apply the field: radiation traction, Hermitian forces,
  the multi-tone convex set, the semidefinite drive program with its proof of
  impossibility, pressure and stability-margin constraints;
- the ramp from the meniscus, and creeping-flow stability.

The theory rests on a declared axisymmetric, bulk-lossless, ideal-source,
creeping-flow model. Nothing in it has been built or measured. Numbers quoted
from `studies/S09-universal-source/study.md` are illustrations only.

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
