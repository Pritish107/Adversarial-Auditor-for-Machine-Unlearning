"""Transform PAPER_DRAFT.md into the deposit-ready markdown that pandoc typesets.

The draft is the SOURCE OF TRUTH and is never edited. This script applies only the
deposit-specific transforms agreed for the PDF build, and ASSERTS that every one of them
matched -- if the draft changes underneath it, the build fails loudly rather than silently
depositing something different from what was reviewed.

Transforms:
  1. Drop the internal "DRAFT - v0" scaffolding block.
  2. Substitute the real author / affiliation / contact.
  3. Unwrap the inline verification flags: keep the substance as plain parentheses, drop the
     "verified live / RE-CHECK" meta-clauses. Exact-string replacements, not regexes, so
     nothing unintended can be rewritten.
  4. Drop the internal VERIFY-BEFORE-SUBMIT checklist section (Reproducibility is kept).

Tables and figures need NO transform here: the draft writes real markdown tables with a
native pandoc caption (a paragraph starting with ":" immediately after the table), so pandoc
emits genuine \\caption{} floats that LaTeX numbers automatically in reading order -- the
prose cross-references ("Table 1 reports the probe", "Table 3 gives the full ladder") are
plain text that happens to match, not something this script maintains. Figures are real
markdown image includes (`![caption](figures/....pdf)`) already in the draft.
"""
from __future__ import annotations

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


def main() -> int:
    authors = AUTHORS
    if "--authors" in sys.argv:
        authors = sys.argv[sys.argv.index("--authors") + 1]
    if authors == "AUTHOR_LINE_PLACEHOLDER":
        sys.exit("PREPARE FAILED: no author line supplied. Pass --authors \"Name · Affiliation "
                 "· contact\". The deposit must not contain placeholders.")

    t = SRC.read_text(encoding="utf-8")

    # 1 + 2 --- strip the draft scaffolding block, insert the real author line
    start = t.index("> **DRAFT — v0.**")
    end = t.index("**Authors:**")
    t = t[:start] + t[end:]
    t = sub(t, "**Authors:** [you] · **Affiliation:** [placeholder] · **Contact:** [placeholder]",
            authors, "author line")

    # 3 --- unwrap the inline verification flags (substance kept, meta-clause dropped).
    # Figures and tables now live in the draft as real markdown, so they need no transform.
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

    # 4 --- drop the internal checklist section
    cut = t.index("## VERIFY-BEFORE-SUBMIT checklist")
    t = t[:cut].rstrip()
    if t.endswith("---"):
        t = t[:-3].rstrip() + "\n"

    # guardrails: nothing internal may survive into the deposit
    for bad, why in [("⟨", "angle-bracket verify delimiter"), ("⟩", "angle-bracket verify delimiter"),
                     ("RE-CHECK", "verify scaffolding"), ("[placeholder]", "placeholder"),
                     ("VERIFY-BEFORE-SUBMIT", "internal checklist"), ("DRAFT — v0", "draft banner")]:
        if bad in t:
            sys.exit(f"PREPARE FAILED: {why} ({bad!r}) still present in the deposit body.")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(t, encoding="utf-8")
    print(f"wrote {OUT}  ({len(t):,} chars)")
    print("  figures included:", t.count("](figures/"))
    # Tables use pandoc's native ": caption" syntax (a paragraph starting with ":" right
    # after the table), so LaTeX assigns real, auto-numbered \caption{} floats instead of
    # bold prose standing in for one -- this is what fixed the missing-"T" text-layer bug.
    import re
    n_tables = len(re.findall(r"(?m)^\| .+ \|\n\|[-:| ]+\|\n(?:\| .+ \|\n)+\n:\s", t))
    print(f"  table captions (native pandoc syntax): {n_tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
