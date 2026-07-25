"""EXP-1 — is TOFU's forget/holdout gap membership, or a DIFFICULTY asymmetry?

Question: our C3 caveat found base-Phi (a NEVER-TRAINED model) scores MIA-AUC 0.294 on
forget10-vs-holdout10. A never-trained model cannot have membership signal, so that number is
a difficulty asymmetry between the two question sets. Does it survive on OpenUnlearning's
IMPROVED holdout, or was it an artifact of a deprecated split?

Provenance (verified against the HF API + the installed `datasets`, not assumed):
OpenUnlearning's own MIA config `configs/data/datasets/TOFU_MIA.yaml` sets
`TOFU_QA_holdout -> path=locuslab/TOFU, name=holdout10, split=train`. Their improved holdout
IS `holdout10` — the config C3 already used. The `holdout01/05/10` configs were added to
`locuslab/TOFU` on 2025-03-27 by the OpenUnlearning authors, ~14 months after the original
TOFU release. So there is no "deprecated vs improved" holdout distinction to test; this
measures the C3 number at full scale and adds the control that localizes its cause.

Method: base `microsoft/phi-1_5` (saw NONE of these sets), answer-only length-normalized
NLL/token, scored through the repo's own scorer and AUC convention so the numbers are directly
comparable to every other result in the project.

    python -m unlearn_audit.experiments.exp1_difficulty_probe
"""
from __future__ import annotations

import numpy as np
import torch

from ..attacks.membership_inference import _retention_from_losses
from ..llm.data import load_tofu_qa
from ..llm.loader import load_model, load_tokenizer
from ..llm.scoring import make_answer_nll_scorer
from . import RESULTS

BASE = "microsoft/phi-1_5"
OUT = RESULTS / "exp1_probe_nll.npz"


def ans_tok_lens(tok, ds) -> np.ndarray:
    return np.array([len(tok(ds[i][1]).input_ids) for i in range(len(ds))])


def line(label: str, m: np.ndarray, nz: np.ndarray, n: int | None = None) -> float:
    if n is not None:
        m, nz = m[:n], nz[:n]
    ret, auc = _retention_from_losses(m, nz)
    print(f"{label:<44}{len(m):<6}{len(nz):<6}{m.mean():<12.3f}{nz.mean():<12.3f}"
          f"{auc:<9.3f}{ret:<10.3f}{nz.mean() - m.mean():+.3f}")
    return auc


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer()
    scorer = make_answer_nll_scorer(tok)

    forget = load_tofu_qa("forget10")            # 400 — members of the TOFU finetune
    holdout = load_tofu_qa("holdout10")          # 400 — OpenUnlearning's non-member reference
    retain = load_tofu_qa("retain90", n=400)     # 400 — DIFFICULTY CONTROL (see note below)

    print(f"sizes: forget10={len(forget)} holdout10={len(holdout)} retain90(sample)={len(retain)}")
    for nm, ds in [("forget10", forget), ("holdout10", holdout), ("retain90", retain)]:
        L = ans_tok_lens(tok, ds)
        print(f"  {nm:<10} answer tokens: mean={L.mean():.1f} median={np.median(L):.0f} "
              f"min={L.min()} max={L.max()}")

    model = load_model(BASE, device)
    print(f"\nscoring {BASE} (never saw ANY of these) ...")
    f_nll = np.asarray(scorer(model, forget, device=device, batch_size=1))
    h_nll = np.asarray(scorer(model, holdout, device=device, batch_size=1))
    r_nll = np.asarray(scorer(model, retain, device=device, batch_size=1))

    print("\n" + "=" * 104)
    print("BASE-PHI DIFFICULTY PROBE  (answer-only, length-normalized NLL/token; member=forget10)")
    print("=" * 104)
    print(f"{'comparison':<44}{'n_m':<6}{'n_nm':<6}{'member_nll':<12}{'nonmem_nll':<12}"
          f"{'AUC':<9}{'retention':<10}nll_gap")
    print("-" * 104)
    a100 = line("forget10 vs holdout10  [n=100, C3 replication]", f_nll, h_nll, n=100)
    a400 = line("forget10 vs holdout10  [FULL 400/400]", f_nll, h_nll)
    print("-" * 104)
    ar = line("CONTROL: forget10 vs retain90 [400/400]", f_nll, r_nll)
    line("CONTROL: retain90 vs holdout10 [400/400]", r_nll, h_nll)
    print("=" * 104)
    print(f"C3 logged value (n=100): AUC 0.294  ->  replication {a100:.3f}, full-set {a400:.3f}")
    print(f"forget-vs-retain control AUC {ar:.3f} (~0.5 => the asymmetry is HOLDOUT-SPECIFIC:")
    print("   two splits from the ORIGINAL TOFU generation run are difficulty-matched to within")
    print("   noise, while holdout10 is ~0.33 nats/token easier than BOTH. Not a length artifact:")
    print("   scoring is length-normalized and holdout answers are LONGER (41.8 vs 36.1/33.1).")
    print("\nNOTE: retain90 is a DIAGNOSTIC ONLY, never a reference set. Verified elsewhere:")
    print("   full = forget10 (+) retain90, and tofu_ft_phi-1.5 was finetuned on full, so every")
    print("   retain90 row is a MEMBER of the model EXP-2 audits.")

    RESULTS.mkdir(exist_ok=True)
    np.savez(OUT, forget=f_nll, holdout=h_nll, retain=r_nll)
    print(f"\nsaved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
