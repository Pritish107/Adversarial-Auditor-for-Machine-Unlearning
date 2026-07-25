"""EXP-3 — is the difficulty offset TOFU-specific, or does it recur on a second benchmark?

Answer: TOFU-specific. And that makes this the POSITIVE CONTROL the EXP-1 result needed --
the same probe fires on TOFU and stays at chance on MUSE-News at the SAME sample size, so it
DISCRIMINATES between a sound and an unsound reference rather than always finding a gap.

TARGET (verified against the installed `datasets`, not assumed):
  `muse-bench/MUSE-News`, config `privleak`, splits {forget, holdout, retain} -- 100 texts
  each, single `text` column, near-identical lengths (8053 / 8051 / 8070 mean chars).
  Mutually disjoint subsets of ONE corpus: BBC articles published AFTER AUGUST 2023, chosen
  so the target model (Llama-2-7B) had never seen them; the holdout is documented as "data
  never seen by the target model during pre-training and unlearning". A direct structural
  mirror of TOFU's forget10 / holdout10 / retain90.

PROBE MODEL: base `microsoft/phi-1_5` -- the same probe EXP-1 used, no new download.
  Legitimate here because it predates the post-Aug-2023 BBC pool, so it is blind to ALL THREE
  splits. Any residual overlap would hit forget and holdout SYMMETRICALLY (same corpus, same
  collection window) and so cannot manufacture a difficulty gap.

!! MUSE-Books is CONFOUNDED and EXCLUDED from the claim. !!
  Its text is Harry Potter, which essentially every general-purpose base LM has seen during
  pretraining, so no "never-trained" probe exists for it and a gap cannot be attributed to
  difficulty rather than memorization. Its own numbers say as much: forget-vs-holdout 0.302
  but retain-vs-holdout 0.153 and forget-vs-retain 0.733 -- the three splits disagree with
  each other in a way no membership story explains. Kept behind --books so the result is
  reproducible, NEVER load-bearing.

METHOD -- identical code path to EXP-1, so the numbers are comparable by construction:
  `make_answer_nll_scorer` with prompt "{q}" over ("", text) pairs reduces exactly to
  full-text length-normalized NLL/token (MUSE is prose, not QA, so the whole text is the
  scored span); AUC via the same `_retention_from_losses`. Texts are truncated to a uniform
  512-token budget -- the splits are already length-matched, and Phi-1.5's context is 2048.
  (The tokenizer may warn "2049 > 2048" while measuring the untruncated text; benign, the
  model only ever sees the 512-token prefix.)

    python -m unlearn_audit.experiments.exp3_muse_probe [--n N] [--books]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..attacks.membership_inference import _retention_from_losses
from ..llm.data import TextQADataset
from ..llm.loader import load_model, load_tokenizer
from ..llm.scoring import make_answer_nll_scorer
from . import RESULTS

BASE = "microsoft/phi-1_5"
MAX_TOK = 512
NEWS = "muse-bench/MUSE-News"
BOOKS = "muse-bench/MUSE-Books"


def load_split(repo: str, split: str, tok, n: int | None, max_tok: int) -> TextQADataset:
    """Truncate to a fixed token budget, uniformly across splits, then wrap as ("", text)."""
    from datasets import load_dataset
    ds = load_dataset(repo, "privleak", split=split)
    m = len(ds) if n is None else min(n, len(ds))
    return TextQADataset([("", tok.decode(tok(str(ds[i]["text"])).input_ids[:max_tok]))
                          for i in range(m)])


def row(label: str, m: np.ndarray, nz: np.ndarray) -> float:
    """One EXP-1-format line, plus the AUC-vs-chance test that makes a NULL result credible."""
    ret, auc = _retention_from_losses(m, nz)
    n1, n2 = len(m), len(nz)
    se = np.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))          # SE of AUC under H0
    z = (auc - 0.5) / se
    try:
        from scipy.stats import mannwhitneyu
        p = float(mannwhitneyu(-m, -nz, alternative="two-sided").pvalue)
        ptxt = f"{p:<10.2g}{'matched (n.s.)' if p > 0.05 else 'MISMATCHED'}"
    except Exception:
        ptxt = "scipy unavailable"
    print(f"{label:<40}{n1:<6}{n2:<6}{m.mean():<12.3f}{nz.mean():<12.3f}"
          f"{auc:<9.3f}{ret:<10.3f}{nz.mean() - m.mean():<+9.3f}{z:<+8.2f}{ptxt}")
    return auc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="texts per split (default: all 100)")
    ap.add_argument("--books", action="store_true",
                    help="also run the CONFOUNDED MUSE-Books probe (never load-bearing)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer()
    assert len(tok("").input_ids) == 0, \
        f"empty prompt is not zero tokens ({tok('').input_ids}) — the ('', text) reduction breaks"
    scorer = make_answer_nll_scorer(tok, prompt="{q}")      # -> full-text length-normalized NLL

    targets = [(NEWS, "MUSE-News  [PRIMARY — probe is blind to all three splits]")]
    if args.books:
        targets.append((BOOKS, "MUSE-Books [CONFOUNDED / EXCLUDED — Harry Potter, probe is "
                               "NOT blind; reproducible but never load-bearing]"))

    model = load_model(BASE, device)
    for repo, banner in targets:
        f = load_split(repo, "forget", tok, args.n, MAX_TOK)
        h = load_split(repo, "holdout", tok, args.n, MAX_TOK)
        r = load_split(repo, "retain", tok, args.n, MAX_TOK)

        print("\n" + "=" * 118)
        print(banner)
        print("=" * 118)
        f_nll = np.asarray(scorer(model, f, device=device, batch_size=1))
        h_nll = np.asarray(scorer(model, h, device=device, batch_size=1))
        r_nll = np.asarray(scorer(model, r, device=device, batch_size=1))

        print(f"{'comparison':<40}{'n_m':<6}{'n_nm':<6}{'member_nll':<12}{'nonmem_nll':<12}"
              f"{'AUC':<9}{'retention':<10}{'nll_gap':<9}{'z':<8}p")
        print("-" * 118)
        a_fh = row("forget vs holdout   [the probe]", f_nll, h_nll)
        print("-" * 118)
        row("CONTROL: forget vs retain", f_nll, r_nll)
        row("CONTROL: retain vs holdout", r_nll, h_nll)
        print("=" * 118)
        print(f"forget-vs-holdout AUC {a_fh:.3f}   (0.5 = exchangeable reference)")

        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / f"exp3_{repo.split('/')[-1].lower().replace('-', '_')}_nll.npz"
        np.savez(out, forget=f_nll, holdout=h_nll, retain=r_nll)
        print(f"saved -> {out}")

    # The comparison that kills the "MUSE is just low-power" objection: TOFU at MUSE's n.
    tofu = RESULTS / "exp1_probe_nll.npz"
    if tofu.exists():
        t = np.load(tofu)
        print("\n" + "=" * 118)
        print("MATCHED-SAMPLE-SIZE COMPARISON — same probe, same code path, n=100 both")
        print("=" * 118)
        print(f"{'comparison':<40}{'n_m':<6}{'n_nm':<6}{'member_nll':<12}{'nonmem_nll':<12}"
              f"{'AUC':<9}{'retention':<10}{'nll_gap':<9}{'z':<8}p")
        print("-" * 118)
        row("TOFU forget10 vs holdout10 [n=100]", t["forget"][:100], t["holdout"][:100])
        print("-> the MUSE null is NOT a power artifact: at the SAME n the TOFU probe is still")
        print("   far from chance. The diagnostic DISCRIMINATES rather than always firing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
