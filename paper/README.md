# Building the four-page manuscript

The [paper](paper.pdf) and [LaTeX source](paper.tex) describe synthetic experiments.
There is no fabricated author, institution or review claim. Numeric macros and
tables come from `examples/results/metrics.json`.

From the repository root, install `python -m pip install -e ".[dev]"` and a TeX
engine, then run:

```sh
python scripts/check_samples.py
python paper/build.py --engine pdflatex
# Or: python paper/build.py --engine /path/to/tectonic
pdftoppm -scale-to 1600 -png paper/paper.pdf paper/build/page
```

Ubuntu prerequisites: `texlive-latex-base texlive-latex-recommended
texlive-fonts-recommended poppler-utils`. Windows can use Tectonic plus Poppler.
No shell escape is enabled. First-use Tectonic may require network access to its
TeX bundle. This manuscript references external generated figures: use the
project build script rather than a single-file compiler.

The script generates `numbers.tex`, compiles twice with pdfLaTeX (Tectonic reruns
internally), rejects undefined references, overfull boxes and empty pages, and
enforces **at most five pages including references**. Only after passing does it
replace the reading copy and write input hashes into `validation.json`.
Intermediate files stay under ignored `paper/build/`.

CI reproduces metrics, compiles with pdfLaTeX, and uploads the PDF, log, manifest
and all rendered pages as `verified-paper`. Inspect every page after changes;
automatic checks cannot establish visual quality. The local Tectonic pages were
inspected on 1 October 2026; crowded plot ticks were corrected.
