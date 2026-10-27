# S09 results report

LaTeX source of a results report for colleagues (about 15 pages compiled) on
study [S09](../../../studies/S09-universal-source/study.md): can one fixed
acoustic machine shape NOA 61 between two water layers into the stigmatic
singlets of Silva-Lora and Torres (JOSA A 37, 1155, 2020), with the lens chosen
only by the drive command, the pin heights and the fill volumes?

It summarizes the published-lens designs (Fig. 9b, Fig. 7a), their
sensitivities, the 81-lens atlas and its nonlinear check, the two-tone
follow-up, the channel study and the design tool `afl design`, the corrections
made along the way, and the open questions. Every number is taken from the S09
study text or from the run records under
`artifacts/studies/S09-universal-source/`; the report's last section lists them.
All results are numerical within the declared model. Nothing has been built or
measured.

| File | Content |
|---|---|
| `main.tex` | Report source; bibliography from `../../theory/references.bib` |
| `make_figures.py` | Plots the atlas, channel-study and formation records into `artifacts/reports/s09-results/figures/` (seconds; reads the records, recomputes no physics) |

## Build

The report needs the private run records and figures under `artifacts/`.
From the repository root:

```sh
uv run python docs/reports/s09-results/make_figures.py
cd docs/reports/s09-results
latexmk -pdf -outdir=../../../artifacts/reports/s09-results main.tex   # runs pdfLaTeX and biber
```

Tectonic also works, but its biber step rejects the `../` path of
`\addbibresource`; build from a copy whose bibliography path is local.

The compiled PDF and the generated figures are private: they stay in
`artifacts/reports/s09-results/` and are never committed or published.
Pending items at the time of writing (formation of the rescued lenses 15 and 71,
the 40-step ramp test of lens 57) are marked as pending in the text; update
the report from their records when they finish.
