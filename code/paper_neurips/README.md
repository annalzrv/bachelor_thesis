# Building the LC-PINN paper

Target venue: NeurIPS 2026 main track (9 pp main content + unlimited
references + checklist + appendix). The repository contains everything
needed to build `main.pdf` from source.

## Files in this directory

- `main.tex` — top-level LaTeX file
- `neurips_2026.sty` — official NeurIPS 2026 style file (2026-01-29)
- `refs.bib` — bibliography (`plainnat`)
- `sections/*.tex` — abstract, introduction, related_work, method,
  theorem, experiments, results, discussion, appendix, checklist
- `figures/*.pdf` — vector figures (Pareto, per-k Helmholtz,
  per-α Schrödinger), produced by `code/scripts/make_*_plot.py`
- `openreview_submission.md` — title / TL;DR / abstract for OpenReview
- `main.pdf` — current compiled PDF (committed for convenience)

All inputs are plain CTAN packages (`amsmath`, `booktabs`, `cleveref`,
`graphicx`, `hyperref`, `mathtools`, `microtype`, `natbib`, `nicefrac`,
`xcolor`, `lineno`, …). No custom macros outside `neurips_2026.sty`.

## Quickest build: `tectonic` (recommended)

`tectonic` auto-fetches missing packages from CTAN, so no separate TeX
distribution is required.

```bash
brew install tectonic        # macOS; or `cargo install tectonic`
cd paper
tectonic -X compile main.tex --reruns 3
```

`--reruns 3` ensures `natbib` / `hyperref` / `cleveref` cross-references
all resolve in one go. The output is `main.pdf` next to `main.tex`.

## Alternative: `latexmk` (TeX Live / MacTeX)

```bash
cd paper
latexmk -pdf -bibtex main.tex
# clean intermediate files when done:
latexmk -c
```

## Alternative: manual `pdflatex` + `bibtex` cycle

```bash
cd paper
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Switching to camera-ready / preprint mode

`main.tex` currently calls

```tex
\usepackage[main]{neurips_2026}
```

which is the **anonymous submission** mode (line numbers on, anonymous
author block, "Submitted to … Do not distribute" footer). For other
modes change the option:

| Option        | Effect                                                      |
|---------------|-------------------------------------------------------------|
| `main`        | Anonymous submission (line numbers, no author info)         |
| `final,main`  | Camera-ready: real authors, no line numbers, conference notice |
| `preprint`    | Real authors, no line numbers, "Preprint." footer (arXiv)   |

For example, to produce the camera-ready PDF:

```tex
\usepackage[final,main]{neurips_2026}
```

and update the `\author{...}` block in `main.tex`.

## Page-limit sanity check

NeurIPS 2026 caps **main content** (everything before References) at
**9 pages**. References, the NeurIPS Paper Checklist, and the Appendix
do not count. The current build:

- Pages 1–9 — main content (Abstract through Discussion)
- Page 10 — References begin
- Pages 13–14 — NeurIPS Paper Checklist
- Pages 15–20 — Appendix

To confirm after edits:

```bash
pdfinfo main.pdf | grep Pages
pdftotext main.pdf - | awk 'BEGIN{p=1} /\f/{p++} /^References/{print "References on page", p; exit}'
```

The "References" line should report page **10** (or earlier — never
later, since that would push main content past 9pp).

## Editing the abstract

The abstract lives in **two places** that must stay in sync:

1. `paper/sections/abstract.tex` — LaTeX source rendered into the PDF.
2. `paper/openreview_submission.md` — plaintext copy uploaded to the
   OpenReview submission form.

After editing `abstract.tex`, update `openreview_submission.md` to
match (LaTeX → plaintext: `\emph{X}` → `*X*`, `\citep{key}` →
`(Author, Year)`, `$\lambda$` → `λ`, `--` → `–`, `---` → `—`,
`$k\!\in\![1,10]$` → `k ∈ [1, 10]`).

## Regenerating figures (optional)

The committed `figures/*.pdf` are sufficient to build `main.pdf`. If you
want to rebuild them from the JSON results:

```bash
cd ..                 # repo root
python scripts/make_pareto_plot.py
python scripts/make_per_k_plot.py
python scripts/make_per_alpha_plot.py
```

Each script reads `code/results/*.json` and writes the corresponding
`paper/figures/*.pdf`.

## Reproducing the experimental results

See the top-level `code/README.md` and `code/scripts/` for training
scripts. Each method has a one-line entrypoint, e.g.

```bash
python scripts/lc_pinn_helmholtz.py --seeds 4
python scripts/sa_pinn_helmholtz.py --seeds 4
```

JSONs land in `code/results/`; figure scripts above pick them up.
