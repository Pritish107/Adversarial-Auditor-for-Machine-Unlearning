"""M3 reference-construction experiments (EXP-1, EXP-2, EXP-2b).

Research scripts, not library code: each answers one question about how the choice of
non-member reference set affects the audit, and each writes its per-example arrays to
`results/` so the reported tables reproduce with no GPU. See README.md.
"""
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
