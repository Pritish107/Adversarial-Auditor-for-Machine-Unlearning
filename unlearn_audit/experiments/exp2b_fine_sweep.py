"""EXP-2b — fine sweep a in {0.75 .. 0.95} resolving EXP-2's open question (a).

(a) = does a REALISTICALLY-calibrated holdout10 auditor read "forgotten" where the
difficulty-matched auditor reads "RETENTION DETECTED"? That is a benchmark-induced MISSED
DETECTION — the unsafe direction for an auditor. EXP-2's coarse grid left it unresolved:
both curves cross the decision boundary inside (0.7, 1.0), which it never sampled.

+---------------------------------------------------------------------------------------------+
| !! THE ALPHA LADDER IS A SYNTHETIC PROBE, NOT AN UNLEARNING METHOD. See exp2 header. !!       |
+---------------------------------------------------------------------------------------------+

DECISION RULES REPORTED PER RUNG
  (1) R-matched, calibrated : retention > 0.0911  (null sd 0.036, FAR 0.050)  <- our auditor
  (2) R-holdout, degenerate : retention > 0.0     (that null is a point mass at 0)
  (3) R-holdout, FIELD RULE : AUC > 0.5           (MIA at/below chance => forgotten)
  (4) R-holdout, MATCHED-MARGIN CONTROL : retention > 0.0911 — the matched threshold
      transplanted onto holdout scores. Isolates the REFERENCE-SET effect from the THRESHOLD
      effect, blocking the rebuttal "you compared a calibrated auditor to an uncalibrated one".
      This is a constructed control, NOT field practice.

HONEST NOTE, stated up front: rules (2) and (3) are MATHEMATICALLY IDENTICAL here, because
retention = clip(2*(AUC-0.5),0,1) makes `retention > 0` <=> `AUC > 0.5`. The degenerate null's
threshold of 0 lands exactly on the field convention. So rule (3) is NOT a straw man — beating
it is beating real practice.

SCOPE (printed with the verdict, and inseparable from it): this tests ABSOLUTE-threshold
auditing. Gold-referenced / relative rules (MUSE PrivLeak, TOFU forget_quality) compare a
candidate to a retrain model on the SAME non-member set, so the offset is common-mode and
cancels — they are IMMUNE. The catch is that TOFU/Phi ships no valid gold retrain, so the
offset-robust rule cannot be run on this benchmark without training one first.

    python -m unlearn_audit.experiments.exp2b_fine_sweep
"""
from __future__ import annotations

import gc

import numpy as np
import torch

from ..attacks.membership_inference import _retention_from_losses
from ..llm.data import load_tofu_qa
from ..llm.guardrail import answer_accuracy
from ..llm.loader import load_tokenizer
from ..llm.scoring import make_answer_nll_scorer
from . import RESULTS
from .exp2_matched_reference import CALIPER, blend_into, caliper_match, cpu_state_dict

BASELINE = "locuslab/tofu_ft_phi-1.5"
BASE = "microsoft/phi-1_5"
FINE = [0.75, 0.80, 0.85, 0.90, 0.95]
THR_MATCHED = 0.0911            # from EXP-2's R-matched null at FAR 0.050

# base-Phi (gold substitute) AUCs, used for the relative-rule scope control
GOLD_AUC_HOLDOUT, GOLD_AUC_MATCHED = 0.332, 0.509

IN_EXP1 = RESULTS / "exp1_probe_nll.npz"
IN_EXP2 = RESULTS / "exp2_ladder.npz"
OUT = RESULTS / "exp2b_fine.npz"


def main() -> int:
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer()
    scorer = make_answer_nll_scorer(tok)
    forget, holdout = load_tofu_qa("forget10"), load_tofu_qa("holdout10")

    exp1 = np.load(IN_EXP1)
    bp_f, bp_h = exp1["forget"], exp1["holdout"]
    fi, hi = caliper_match(bp_f, bp_h, CALIPER)
    thr_m = float(np.quantile(np.load(IN_EXP2)["null_match"], 0.95))
    assert abs(thr_m - THR_MATCHED) < 1e-3, f"threshold drift vs EXP-2: {thr_m}"
    print(f"[setup] matched pairs n={len(fi)} | matched threshold {thr_m:.4f} (FAR 0.050)")
    print(f"[setup] residual imbalance {bp_h[hi].mean() - bp_f[fi].mean():+.3f} nats/token")
    print(f"[setup] AUDITING THE DIFFICULTY-MATCHED SUBPOPULATION of forget10 (n={len(fi)} of 400)\n")

    sd_b, sd_p = cpu_state_dict(BASELINE), cpu_state_dict(BASE)
    try:
        model = AutoModelForCausalLM.from_pretrained(BASELINE, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(BASELINE, torch_dtype=torch.float16)
    model = model.to(device).eval()

    rows = []
    for a in FINE:
        blend_into(model, sd_b, sd_p, a).eval()
        f = np.asarray(scorer(model, forget, device=device, batch_size=1))
        h = np.asarray(scorer(model, holdout, device=device, batch_size=1))
        acc = answer_accuracy(model, forget, tok, device)
        ret_h, auc_h = _retention_from_losses(f, h)
        ret_m, auc_m = _retention_from_losses(f[fi], h[hi])
        rows.append(dict(a=a, acc=acc, f=f, h=h, ret_h=ret_h, auc_h=auc_h,
                         ret_m=ret_m, auc_m=auc_m))
        print(f"[fine] a={a:<5} acc={acc:.3f} | holdout AUC={auc_h:.3f} ret={ret_h:.3f} "
              f"| matched AUC={auc_m:.3f} ret={ret_m:.3f}")

    del sd_b, sd_p, model
    gc.collect()
    torch.cuda.empty_cache()

    D = lambda b: "RETENTION DETECTED" if b else "forgotten (clean)"
    print("\n" + "=" * 126)
    print("EXP-2b FINE SWEEP — decisions under four rules   (rules (2) and (3) are identical: "
          "retention>0 <=> AUC>0.5)")
    print("=" * 126)
    print(f"{'a':<7}{'naive acc':<11}{'hold AUC':<10}{'hold ret':<10}{'match AUC':<11}"
          f"{'match ret':<11}{'(1) matched':<21}{'(3) holdout AUC>.5':<21}{'(4) hold@.091'}")
    print("-" * 126)
    flips3, flips4 = [], []
    for r in rows:
        d1, d3, d4 = r["ret_m"] > THR_MATCHED, r["auc_h"] > 0.5, r["ret_h"] > THR_MATCHED
        if d1 and not d3:
            flips3.append(r)
        if d1 and not d4:
            flips4.append(r)
        print(f"{r['a']:<7}{r['acc']:<11.3f}{r['auc_h']:<10.3f}{r['ret_h']:<10.3f}"
              f"{r['auc_m']:<11.3f}{r['ret_m']:<11.3f}{D(d1):<21}{D(d3):<21}{D(d4)}")
    print("=" * 126)

    print("\n(a) BENCHMARK-INDUCED MISSED DETECTION vs REALISTIC FIELD RULE (3) AUC>0.5:")
    for r in flips3:
        print(f"    *** FIRES at a={r['a']}: matched -> RETENTION DETECTED "
              f"(ret {r['ret_m']:.3f} > {THR_MATCHED:.3f}) | holdout10 -> forgotten "
              f"(AUC {r['auc_h']:.3f} <= 0.500). FALSE NEGATIVE.")
    print(f"    => (a) ANSWERED {'YES on %d/%d rungs' % (len(flips3), len(rows)) if flips3 else 'NO'}.")
    print("    Ground truth is unambiguous: every a<1 contains baseline weight, so retention IS")
    print("    present and the matched verdict is the correct one.")

    print("\n    same-margin CONTROL (4) — holdout scores at threshold 0.091 "
          "(isolates reference-set effect):")
    for r in flips4:
        print(f"    *** FIRES at a={r['a']}: matched ret {r['ret_m']:.3f} DETECTED | holdout ret "
              f"{r['ret_h']:.3f} forgotten — same threshold, only the reference set differs.")
    print(f"    => fires on {len(flips4)}/{len(rows)} rungs.")

    print("\n    retention shift (matched - holdout): "
          + ", ".join(f"a={r['a']}:{r['ret_m'] - r['ret_h']:+.3f}" for r in rows))

    print("\n" + "=" * 126)
    print("SCOPE CONTROL — relative (gold-referenced) rules are IMMUNE; report this WITH the flip")
    print("=" * 126)
    print(f"{'a':<7}{'hold AUC':<10}{'d_vs_gold':<12}{'match AUC':<11}{'d_vs_gold':<12}discrepancy")
    for r in rows:
        dh, dm = r["auc_h"] - GOLD_AUC_HOLDOUT, r["auc_m"] - GOLD_AUC_MATCHED
        print(f"{r['a']:<7}{r['auc_h']:<10.3f}{dh:<+12.3f}{r['auc_m']:<11.3f}{dm:<+12.3f}"
              f"{abs(dm - dh):.3f}")
    abs_spread = [r["auc_m"] - r["auc_h"] for r in rows]
    rel_spread = [(r["auc_m"] - GOLD_AUC_MATCHED) - (r["auc_h"] - GOLD_AUC_HOLDOUT) for r in rows]
    print(f"\n  reference-set spread: ABSOLUTE rule ~{np.mean(abs_spread):.3f} -> "
          f"RELATIVE rule ~{np.mean(rel_spread):.3f}  (order-of-magnitude collapse)")
    print("  The gold reference's AUC is depressed by the SAME difficulty gap as the candidate's,")
    print("  so subtracting cancels it. A gold-referenced rule (PrivLeak / forget_quality) would")
    print("  NOT flip. BUT that escape hatch is UNAVAILABLE on TOFU/Phi as shipped: the retain90")
    print("  checkpoint is a byte-identical duplicate and every unlearned repo is empty, so there")
    print("  is no valid gold retrain to reference without training one first.")

    RESULTS.mkdir(exist_ok=True)
    np.savez(OUT, alphas=np.array(FINE),
             **{f"f_{r['a']}": r["f"] for r in rows},
             **{f"h_{r['a']}": r["h"] for r in rows},
             ret_h=np.array([r["ret_h"] for r in rows]),
             ret_m=np.array([r["ret_m"] for r in rows]),
             auc_h=np.array([r["auc_h"] for r in rows]),
             auc_m=np.array([r["auc_m"] for r in rows]),
             acc=np.array([r["acc"] for r in rows]))
    print(f"\nsaved -> {OUT}")
    print("REMINDER: alpha ladder = SYNTHETIC retention gradient, NOT an unlearning method.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
