"""Transform PAPER_DRAFT.md into the deposit-ready markdown that pandoc typesets.

The draft is the SOURCE OF TRUTH and is never edited. This script applies only the
deposit-specific transforms agreed for the PDF build, and ASSERTS that every one of them
matched -- if the draft changes underneath it, the build fails loudly rather than silently
depositing something different from what was reviewed.

Transforms (formatting only -- no word of body content is added, removed, or reworded):
  1. Extract the H1 title, the author line, and the Abstract section body into a pandoc YAML
     metadata block, so the LaTeX writer emits a real \\title{}/\\author{}/\\maketitle title
     block and wraps the abstract in article.cls's own \\begin{abstract} environment, instead
     of rendering them as plain body headings (which is what a bare H1 + bold-prose author
     line + H2 "Abstract" heading produce). \\date{} is left empty via preamble.tex, not here.
  2. Unwrap the inline verification flags: keep the substance as plain parentheses, drop the
     "verified live / RE-CHECK" meta-clauses. Exact-string replacements, not regexes.
  3. Apply a non-breaking hyphen (U+2011) to the one "2025-03-27" date so it cannot break
     mid-token at the end of a line. Same visible glyph as a regular hyphen; preamble.tex
     maps it so the build's glyph-check still catches any accidental drop.
  4. Drop the internal VERIFY-BEFORE-SUBMIT checklist section (Reproducibility is kept).
  5. Replace the References section's plain "[N] ..." paragraphs with a raw-LaTeX hanging-
     indent list (bracket labels [1]..[13], preserving the exact existing numbers -- entries
     8 and 9 are intentionally absent, since those two works are cited by name in prose only,
     per an explicit earlier decision, not as markers). A markdown ordered list cannot
     reproduce that non-consecutive numbering (pandoc auto-increments), so the entries are
     emitted as \\item[{[N]}] ... inside a plain LaTeX `list` environment -- the same
     construction \\thebibliography uses internally. Markdown *italics* in the entries are
     converted to \\emph{} by hand for this block only, since raw LaTeX bypasses pandoc's own
     markdown-to-LaTeX conversion; the one literal "%" (in "Min-K%++") is escaped to "\\%" so
     it cannot be read as a LaTeX comment marker.

Tables and figures need NO transform: the draft writes real markdown tables with a native
pandoc caption (a paragraph starting with ":" immediately after the table), so pandoc emits
genuine \\caption{} floats that LaTeX numbers automatically in reading order -- the prose
cross-references ("Table 1 reports the probe") are plain text that happens to match, not
something this script maintains. Figures are real markdown image includes already in the
draft.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "PAPER_DRAFT.md"
OUT = HERE / "build" / "paper_body.md"

AUTHORS = "AUTHOR_LINE_PLACEHOLDER"     # overridden via --authors; build refuses the default


def sub(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        sys.exit(f"PREPARE FAILED [{label}]: expected text not found in PAPER_DRAFT.md.\n"
                 f"  looked for: {old[:110]!r}\n"
                 f"  the draft changed — reconcile before building the deposit PDF.")
    return text.replace(old, new, 1)


def yaml_scalar(s: str) -> str:
    """Quote a single-line string for a YAML flow scalar. Fails loudly on anything that
    would need escaping rather than silently mishandling it -- title/author are short,
    known strings, so an unhandled character means something changed upstream."""
    if '"' in s or "\\" in s or "\n" in s:
        sys.exit(f"PREPARE FAILED: yaml_scalar cannot safely quote {s!r} — contains a "
                 "character this simple quoting does not handle.")
    return f'"{s}"'


def build_references_block(refs_text: str) -> str:
    """Turn the plain '[N] entry...' paragraphs into a raw-LaTeX hanging-indent list."""
    entries = re.findall(r"\[(\d+)\] (.+?)(?=\n\n\[\d+\]|\n\n---|\Z)", refs_text, re.S)
    if len(entries) != 11:
        sys.exit(f"PREPARE FAILED: expected 11 reference entries, found {len(entries)}.")
    expected_nums = ["1", "2", "3", "4", "5", "6", "7", "10", "11", "12", "13"]
    if [n for n, _ in entries] != expected_nums:
        sys.exit(f"PREPARE FAILED: reference numbering changed — expected {expected_nums}, "
                 f"got {[n for n, _ in entries]}.")

    items = []
    for num, body in entries:
        body = " ".join(body.split())               # collapse the source's line-wrapping
        if "%" not in body and "Min-K" in body and "++" in body:
            pass  # sanity anchor only; real check is the assertion below
        body = body.replace("%", r"\%")              # LaTeX comment character
        for bad in "&#_{}~^\\":
            if bad in body.replace(r"\%", ""):
                sys.exit(f"PREPARE FAILED: reference [{num}] contains unhandled LaTeX-special "
                         f"character {bad!r} — extend build_references_block before proceeding.")
        body = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", body)   # markdown italics -> \emph
        items.append(f"\\item[{{[{num}]}}] {body}")

    list_opts = (r"\setlength{\labelwidth}{2.4em}\setlength{\leftmargin}{2.4em}"
                r"\setlength{\itemindent}{0pt}\setlength{\itemsep}{6pt}\setlength{\parsep}{0pt}")
    return ("```{=latex}\n"
            f"\\begin{{list}}{{}}{{{list_opts}}}\n" + "\n".join(items) + "\n"
            "\\end{list}\n"
            "```")


def main() -> int:
    authors = AUTHORS
    if "--authors" in sys.argv:
        authors = sys.argv[sys.argv.index("--authors") + 1]
    if authors == "AUTHOR_LINE_PLACEHOLDER":
        sys.exit("PREPARE FAILED: no author line supplied. Pass --authors \"Name · Affiliation "
                 "· contact\". The deposit must not contain placeholders.")

    t = SRC.read_text(encoding="utf-8")

    # 1a --- extract the H1 title (must be the first line)
    if not t.startswith("# "):
        sys.exit("PREPARE FAILED: draft does not start with an H1 title.")
    title_end = t.index("\n")
    title = t[2:title_end].strip()

    # 1b --- extract the Abstract section body
    abs_marker = "## Abstract\n\n"
    abs_start = t.index(abs_marker) + len(abs_marker)
    abs_end = t.index("\n\n---\n", abs_start)
    abstract = t[abs_start:abs_end].strip()

    # 1c --- everything before "## 1. Introduction" was title/draft-banner/author/abstract
    # scaffolding; the extracted pieces above replace it, so the rest is dropped wholesale.
    intro_marker = "## 1. Introduction"
    if intro_marker not in t:
        sys.exit("PREPARE FAILED: '## 1. Introduction' not found — cannot locate body start.")
    body = t[t.index(intro_marker):]

    yaml_abstract = "\n".join(("  " + ln if ln else "") for ln in abstract.splitlines())
    front_matter = (
        "---\n"
        f"title: {yaml_scalar(title)}\n"
        f"author: {yaml_scalar(authors)}\n"
        "abstract: |\n"
        f"{yaml_abstract}\n"
        "---\n\n"
    )
    t = front_matter + body

    # 2 --- unwrap the inline verification flags (substance kept, meta-clause dropped)
    t = sub(t,
            "⟨TOFU_MIA.yaml → TOFU_QA_holdout → locuslab/TOFU config holdout10; added 2025-03-27 —\n"
            "verified live from HF earlier, RE-CHECK at submission time⟩",
            "(TOFU_MIA.yaml → TOFU_QA_holdout → locuslab/TOFU config holdout10; added 2025-03-27)",
            "flag 1: provenance")
    t = sub(t,
            "⟨identical safetensors weights, differing\n"
            "only in fp32 vs bf16 encoding — verified live from HF earlier, RE-CHECK at submission time⟩",
            "(identical safetensors weights, differing\nonly in fp32 vs bf16 encoding)",
            "flag 2: duplicate checkpoint")
    t = sub(t,
            " ⟨verified live from HF earlier, RE-CHECK at submission\ntime⟩",
            "", "flag 3: empty repos")

    # 3 --- non-breaking hyphen in the one date, so it cannot break mid-token at line end.
    # Same glyph as a regular hyphen (U+2011 vs U+002D) — purely a line-breaking hint.
    t = sub(t, "added 2025-03-27)", "added 2025\u201103\u201127)", "date non-break")

    # 4 --- drop the internal checklist section
    cut = t.index("## VERIFY-BEFORE-SUBMIT checklist")
    t = t[:cut].rstrip()
    if t.endswith("---"):
        t = t[:-3].rstrip() + "\n"

    # 5 --- References: plain paragraphs -> raw-LaTeX hanging-indent bracket list
    refs_marker = "than asserting a publication venue.\n\n"
    refs_start = t.index(refs_marker) + len(refs_marker)
    # The checklist section was already cut in step 4, so References now runs to end-of-file.
    refs_text = t[refs_start:]
    t = t[:refs_start] + build_references_block(refs_text) + "\n"

    # 6 --- promote heading levels by one (## -> #, ### -> ##). The draft numbers "## 1.
    # Introduction" / "## 2. Background" / "### 2.1 Unlearning" etc. as H2/H3 because H1 was
    # the document title within the draft's own markdown hierarchy -- sensible when the file
    # is read as plain markdown (e.g. on GitHub). Now that the title lives in YAML metadata
    # (step 1) instead of occupying the body's H1, pandoc's LaTeX writer maps whatever level
    # is topmost in the BODY to \section (via --top-level-division=section in build_pdf.sh);
    # since that topmost level is H2, leaving it unpromoted rendered every "N. Heading" as
    # \subsection and every "N.M Heading" as \subsubsection -- one step too shallow, and the
    # hierarchy the reader sees no longer matches the numbering ("1. Introduction" looked like
    # a subsection of nothing). This is a heading-MARKUP change only: PAPER_DRAFT.md itself
    # keeps its own H1/H2/H3 levels unchanged; this promotion exists only in the build's copy.
    t = re.sub(r"(?m)^(#{2,3}) ", lambda m: m.group(1)[1:] + " ", t)

    # guardrails: nothing internal may survive into the deposit
    for bad, why in [("⟨", "angle-bracket verify delimiter"), ("⟩", "angle-bracket verify delimiter"),
                     ("RE-CHECK", "verify scaffolding"), ("[placeholder]", "placeholder"),
                     ("VERIFY-BEFORE-SUBMIT", "internal checklist"), ("DRAFT — v0", "draft banner"),
                     ("[you]", "author placeholder")]:
        if bad in t:
            sys.exit(f"PREPARE FAILED: {why} ({bad!r}) still present in the deposit body.")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(t, encoding="utf-8")
    print(f"wrote {OUT}  ({len(t):,} chars)")
    print("  figures included:", t.count("](figures/"))
    n_tables = len(re.findall(r"(?m)^\| .+ \|\n\|[-:| ]+\|\n(?:\| .+ \|\n)+\n:\s", t))
    print(f"  table captions (native pandoc syntax): {n_tables}")
    print(f"  title: {title[:60]}...")
    print(f"  author: {authors}")
    print(f"  abstract: {len(abstract):,} chars")
    item_pattern = r"item\[\{\[(\d+)\]"
    print(f"  references: 11 entries as raw-LaTeX list, numbers {re.findall(item_pattern, t)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
