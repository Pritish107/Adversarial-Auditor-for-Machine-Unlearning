#!/usr/bin/env bash
# Build the deposit-ready PDF from PAPER_DRAFT.md + the committed vector figures.
#
#   bash paper/build_pdf.sh "Name · Affiliation · contact"
#
# Toolchain (see paper/README.md for install): pandoc 3.10.1 + tectonic 0.16.9.
# tectonic is a self-contained XeTeX engine -- no system TeX install -- and fetches the
# LaTeX packages it needs on first run (cached thereafter).
set -euo pipefail

AUTHORS="${1:-}"
[ -n "$AUTHORS" ] || { echo "usage: bash paper/build_pdf.sh \"Name · Affiliation · contact\"" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p build

# Locate the tools: PATH first, then the user-local install dirs used by paper/README.md.
for d in "$LOCALAPPDATA/Programs/pandoc-3.10.1" "$LOCALAPPDATA/Programs/tectonic" \
         "$HOME/AppData/Local/Programs/pandoc-3.10.1" "$HOME/AppData/Local/Programs/tectonic"; do
  [ -d "$d" ] && PATH="$d:$PATH"
done
export PATH
command -v pandoc   >/dev/null || { echo "pandoc not found -- see paper/README.md" >&2; exit 127; }
command -v tectonic >/dev/null || { echo "tectonic not found -- see paper/README.md" >&2; exit 127; }

PY="${PYTHON:-../.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY=python

echo "[1/3] preparing deposit body from PAPER_DRAFT.md"
"$PY" prepare_for_pdf.py --authors "$AUTHORS"

echo "[2/3] pandoc -> LaTeX"
pandoc build/paper_body.md \
  --standalone \
  --from=markdown+tex_math_dollars \
  --to=latex \
  --output=build/paper.tex \
  --include-in-header=preamble.tex \
  --variable=documentclass:article \
  --variable=papersize:a4 \
  --variable=fontsize:11pt \
  --variable=geometry:"margin=0.95in" \
  --variable=linkcolor:black \
  --variable=urlcolor:black \
  --variable=colorlinks:true \
  --resource-path=.

echo "[3/3] tectonic -> PDF"
tectonic build/paper.tex --outdir build --print 2>&1 | grep -viE "^note: downloading" \
  | tee build/tectonic.log || true

# Fail loudly rather than deposit a PDF with silently-dropped glyphs or missing figures.
if grep -qiE "Missing character|could not represent character" build/tectonic.log; then
  echo "BUILD FAILED: glyphs were dropped --" >&2
  grep -iE "Missing character|could not represent character" build/tectonic.log | sort -u >&2
  exit 1
fi
if grep -qiE "Unable to load picture|not found on input line" build/tectonic.log; then
  echo "BUILD FAILED: a figure did not embed." >&2; exit 1
fi
[ -f build/paper.pdf ] || { echo "BUILD FAILED: no PDF produced." >&2; exit 1; }

cp build/paper.pdf paper.pdf
echo "built: paper/paper.pdf"
