"""Transform PAPER_DRAFT.md into the deposit-ready markdown that pandoc typesets.

The draft is the SOURCE OF TRUTH and is never edited. This script applies only the
deposit-specific transforms agreed for the PDF build, and ASSERTS that every one of them
matched -- if the draft changes underneath it, the build fails loudly rather than silently
depositing something different from what was reviewed.

Transforms:
  1. Drop the internal "DRAFT - v0" scaffolding block.
  2. Substitute the real author / affiliation / contact.
  3. Unwrap the four inline verification flags: keep the substance as plain parentheses,
     drop the "verified live / RE-CHECK" meta-clauses. These are exact-string replacements,
     not regexes, so nothing unintended can be rewritten.
  4. Renumber the Table captions into reading order (they appear 1, 4, 3, 2 in the draft).
     They are prose-summary notes rather than tabular floats, so they stay as blockquotes;
     the one prose cross-reference ("Table 1 reports the probe") keeps pointing at Table 1.
  5. Replace the two figure caption blockquotes with real image includes, so the committed
     vector PDFs are embedded at their referenced positions with the caption below.
  6. Drop the internal VERIFY-BEFORE-SUBMIT checklist section (Reproducibility is kept).
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

    # 3 --- unwrap the inline verification flags (substance kept, meta-clause dropped)
    t = sub(t,
            "⟨TOFU_MIA.yaml → TOFU_QA_holdout → locuslab/TOFU config holdout10; added 2025-03-27 —\n"
            "verified live from HF earlier, RE-CHECK at submission time⟩",
            "(TOFU_MIA.yaml → TOFU_QA_holdout → locuslab/TOFU config holdout10; added 2025-03-27)",
            "flag 1: provenance")
    t = sub(t,
            "⟨341/341 state-dict tensors aligned\n"
            "> — runtime assertion in exp2, not stored in the arrays; RE-CHECK on rerun⟩",
            "341/341 state-dict tensors aligned",
            "flag 2: tensor count")
    t = sub(t,
            "⟨identical safetensors weights, differing\n"
            "only in fp32 vs bf16 encoding — verified live from HF earlier, RE-CHECK at submission time⟩",
            "(identical safetensors weights, differing\nonly in fp32 vs bf16 encoding)",
            "flag 3: duplicate checkpoint")
    t = sub(t,
            " ⟨verified live from HF earlier, RE-CHECK at submission\ntime⟩",
            "", "flag 4: empty repos")

    # 4 --- tables into reading order: draft has 1 (S3), 4 (S3.1), 3 (S4.1), 2 (S4.3)
    t = sub(t, "> **Table 4.** Probe on both benchmarks", "> **Table 2.** Probe on both benchmarks",
            "table renumber 4->2")
    t = sub(t, "> **Table 2.** MIA-family map", "> **Table 4.** MIA-family map",
            "table renumber 2->4")

    # 5 --- real figure includes; strip the "**Figure N.**" prefix so LaTeX numbers them once
    t = sub(t,
            "> **Figure 1.** NLL distributions for forget10, holdout10, and retain90 under the never-trained\n"
            "> model. The holdout distribution is visibly shifted toward lower NLL (easier).\n"
            "> *[`paper/figures/fig1_difficulty_distributions.pdf`]*",
            "![NLL distributions for forget10, holdout10, and retain90 under the never-trained "
            "model. The holdout distribution is visibly shifted toward lower NLL (easier).]"
            "(figures/fig1_difficulty_distributions.pdf){width=68%}",
            "figure 1 include")
    t = sub(t,
            "> **Figure 2.** Residual-retention score vs α, under holdout10 (field default) and the\n"
            "> difficulty-matched reference, with decision thresholds and the missed-detection band shaded.\n"
            "> *[`paper/figures/fig2_missed_detection.pdf`]*",
            "![Residual-retention score vs α, under holdout10 (field default) and the "
            "difficulty-matched reference, with decision thresholds and the missed-detection "
            "band shaded.](figures/fig2_missed_detection.pdf){width=100%}",
            "figure 2 include")

    # 6 --- drop the internal checklist section
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
    print("  table captions:  ", sum(f"**Table {i}.**" in t for i in (1, 2, 3, 4)), "of 4, in order",
          [i for i in (1, 2, 3, 4) if f"**Table {i}.**" in t])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
