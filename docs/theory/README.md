# Theoretical formulation

This is the canonical LaTeX source for **Acoustic formation and stabilization
of Cartesian optical interfaces**. It was prepared in response to the request
to establish the general equations and inverse problem before further numerical
execution. It does not use the existing simulator's apparatus, fitted drives,
mesh, or numerical performance as a theoretical premise.

- [Compiled manuscript](../../artifacts/theory/acoustic-cartesian-theory.pdf)
- [Main LaTeX source](main.tex)
- [Bibliography](references.bib)
- [Source attribution and reading record](source-map.md)

The manuscript progresses from signed Cartesian optical paths to continuum
mechanics, required traction, acoustic wave transfer, load feasibility,
stability, and dynamic optimal control. The appendix derives the sign,
curvature, energy, and actuator-sensitivity checks. Every reduced model states
its additional conditions. A finite pulse, a maintained liquid diopter and a
cured component are distinct outputs.

## Build

From the repository root:

```bash
python3 tools/build_theory.py
```

The document requires TeX Live/MacTeX with `latexmk`, pdfLaTeX, biber, biblatex,
TikZ and cleveref. The build only compiles the manuscript. It produces the PDF
and a source/PDF checksum manifest in `artifacts/theory/`; TeX auxiliary files
and logs remain in `artifacts/theory/build/`. The build fails on unresolved
citations/references and overfull boxes. Downloaded articles stay in the private
`references/papers/` library, with provenance in `references/manifest.json`.

## Scientific status

This is a theoretical research formulation and literature synthesis, not a
peer-reviewed publication, a computed excitation, a proof of global
controllability, or an experimentally validated acoustic-lens design. Its
propositions are conditional and their short proofs are included. The
adjoints are given in general operator form because apparatus-specific
boundary conditions have deliberately not yet been selected. They must be
specialized, including all interface and boundary terms, before implementation.

No new forward simulation, inverse optimization, or benchmark evaluation is
part of this milestone. The existing numerical experiments remain separate
under their original artifact directories and retain their documented limits.
