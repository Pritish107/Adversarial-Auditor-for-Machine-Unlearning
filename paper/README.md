# Paper build

`PAPER_DRAFT.md` is the **source of truth**. The deposit PDF is generated from it — the draft
is never edited by the build.

```bash
bash paper/build_pdf.sh "Name · Affiliation · contact"     # -> paper/paper.pdf
```

## Toolchain

| tool | version | why |
|---|---|---|
| [pandoc](https://pandoc.org) | 3.10.1 | Markdown → LaTeX |
| [tectonic](https://tectonic-typesetting.github.io) | 0.16.9 | LaTeX → PDF |

Tectonic is a **self-contained XeTeX engine** — no system TeX distribution, no admin rights.
It fetches the LaTeX packages the document needs on first run and caches them, so the first
build needs network access and takes a few minutes; later builds are fast.

Both are portable archives. `winget` was unusable here (its CDN source failed to update), so
they were installed straight from the official GitHub releases into `%LOCALAPPDATA%\Programs`:

```powershell
$dest = "$env:LOCALAPPDATA\Programs"
# pandoc  -> $dest\pandoc-3.10.1\pandoc.exe
Invoke-WebRequest "https://github.com/jgm/pandoc/releases/download/3.10.1/pandoc-3.10.1-windows-x86_64.zip" -OutFile "$env:TEMP\pandoc.zip"
Expand-Archive "$env:TEMP\pandoc.zip" -DestinationPath $dest -Force
# tectonic -> $dest\tectonic\tectonic.exe
Invoke-WebRequest "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.16.9/tectonic-0.16.9-x86_64-pc-windows-msvc.zip" -OutFile "$env:TEMP\tectonic.zip"
Expand-Archive "$env:TEMP\tectonic.zip" -DestinationPath "$dest\tectonic" -Force
```

`build_pdf.sh` finds them on `PATH` or in those directories.

## Pipeline

1. **`prepare_for_pdf.py`** — applies the deposit-only transforms to the draft and writes
   `build/paper_body.md`. Every transform is an **exact-string** replacement that is asserted
   to have matched; if the draft changes underneath it, the build fails loudly rather than
   quietly depositing something other than what was reviewed. It:
   - drops the internal `DRAFT — v0` block and the `VERIFY-BEFORE-SUBMIT` checklist
     (`Reproducibility` is kept);
   - substitutes the real author line;
   - unwraps the four inline verification flags — the substance (config path, `2025-03-27`,
     `341/341`, the fp32-vs-bf16 detail) is kept as plain parentheses, the
     "verified live / RE-CHECK" meta-clauses are dropped;
   - renumbers the Table captions into reading order (the draft has them 1, 4, 3, 2);
   - replaces the two figure caption blockquotes with real image includes.
2. **pandoc** → `build/paper.tex`, with `preamble.tex` injected.
3. **tectonic** → `build/paper.pdf`, copied to `paper/paper.pdf`.

## Figures

Embedded as the **committed vector PDFs** (`figures/*.pdf`), not the PNGs, so they stay
resolution-independent in print. They are themselves reproducible from the committed result
arrays with no GPU:

```bash
pip install -r requirements-paper.txt && python paper/make_figures.py
```

## Build guards

The build refuses to produce a deposit PDF that is silently wrong. It fails if:

- **any glyph was dropped** — tectonic only *warns* on a missing character, so `≈0.33` could
  become `0.33` without erroring. Latin Modern lacks `α θ ≈ ⇒ → ∈ − × ⟨ ⟩`, so `preamble.tex`
  maps each one explicitly and the build greps the log for `Missing character` /
  `could not represent character`;
- **a figure failed to embed** (`Unable to load picture`, `not found on input line`);
- **no PDF was produced**;
- **the author line is missing** — `prepare_for_pdf.py` refuses to run without `--authors`,
  so the deposit cannot contain a placeholder;
- **internal scaffolding survived** — it asserts no `⟨ ⟩`, `RE-CHECK`, `[placeholder]`,
  `VERIFY-BEFORE-SUBMIT` or `DRAFT — v0` remains in the body.

## Verifying a build

```bash
python -c "
from pypdf import PdfReader
r = PdfReader('paper/paper.pdf'); print('pages', len(r.pages))
print('figures', sum(len(p.get('/Resources',{}).get('/XObject',{}) or {}) for p in r.pages))
"
```

Expect **9 pages** and **2 embedded form XObjects** (one per figure). `pypdf` is a
verification convenience only — the build itself does not need it.

## Note on the layout

Single-column A4, 11pt, 1in margins — a plain preprint look for a Zenodo/arXiv-style deposit,
not a venue template. Section numbering comes from the draft's own headings.
