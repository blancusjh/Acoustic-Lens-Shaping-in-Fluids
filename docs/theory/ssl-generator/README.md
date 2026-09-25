# Standalone SSL generator report

Version 1.2 (24 September 2026) begins on PDF page 4 with the general
three-dimensional multi-fluid problem, followed by the constrained shape
Lagrangian and its traction multiplier, compartment pressure gauges, and
the viscous Rayleighian. Acoustic inversion follows; Cartesian SSLs are
introduced only in Sections 11–13. The previous version is archived under
`/Users/blancus/Private/research-archives/2026-09-24-ssl-report-v1p1/`.

Editable LaTeX source: `main.tex`, `multifluid-variational.tex`, `theory.tex`,
and `vision.tex`.
The shared bibliography is `../references.bib`. This report is separate from
the main theoretical manuscript; it does not replace its PDF.

Build with `python tools/build_ssl_report.py` from the repository root.
XeLaTeX, latexmk and biber are required. The builder checks unresolved
references, missing characters and overfull boxes, and records source/PDF hashes.
Before rebuilding a released report, preserve the source and PDF milestone
in the separate research archive. The builder refuses to overwrite its final PDF.

Output: `artifacts/theory/ssl-generator/ssl-generator-vision.pdf`.

The LaTeX bodies were migrated from the Markdown design notes and then
typeset/edited in LaTeX. The conversion helper is not part of normal builds;
do not regenerate over edited LaTeX without reviewing and archiving it.

Optional algebra checks (SymPy and NumPy, without changing project dependencies):

```sh
uv run --no-project --with numpy --with sympy python tools/verify_ssl_report_math.py
```

The check records are in `artifacts/theory/ssl-generator/algebra-checks.json`;
archive them before rerunning. They are not numerical or physical validation
of the lens apparatus.
