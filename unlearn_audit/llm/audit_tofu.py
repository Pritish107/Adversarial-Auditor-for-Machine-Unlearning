"""M2b headline: three-case text audit through the CORE auditor.

  baseline (tofu_ft_phi-1.5)   members present  -> should read RETENTION DETECTED
  guardrail (wrap baseline)    answer-acc ~0, NLL == baseline -> RETENTION DETECTED  [headline]
  base-Phi (microsoft/phi-1_5) never saw members -> the null; should read forgotten (clean)

member = forget10, non-member = holdout10. Same injected answer-NLL scorer + same LossMIA +
same audit() as CIFAR. base-Phi supplies the calibration null. No LLM training. Models loaded
one at a time to fit the 4060.

    python -m unlearn_audit.llm.audit_tofu
"""
from __future__ import annotations

import gc

import torch

from ..attacks import build_attack
from ..attacks.base import AttackContext
from ..attacks.membership_inference import bootstrap_gold_null
from ..audit.score import audit
from .data import load_tofu_qa
from .guardrail import TextGuardrail, answer_accuracy
from .loader import load_model, load_tokenizer
from .scoring import make_answer_nll_scorer

BASELINE = "locuslab/tofu_ft_phi-1.5"
BASE = "microsoft/phi-1_5"
N = 100
N_BOOT = 200
TARGET_FAR = 0.05


def _decision(rep):
    if rep.retention_detected is None:
        return "UNCALIBRATED"
    return "RETENTION DETECTED" if rep.retention_detected else "forgotten (clean)"


def _case(model, member, nonmember, retain, scorer, device, null):
    ctx = AttackContext(target_model=model, member_data=member, nonmember_data=nonmember,
                        retain_data=retain, device=device, batch_size=1, scorer=scorer)
    res = build_attack("loss_mia").run(ctx)
    rep = audit([res], aggregate="max", target_far=TARGET_FAR, null_scores=null)
    return res, rep


def _free(m):
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer()
    scorer = make_answer_nll_scorer(tok)
    member = load_tofu_qa("forget10", n=N)
    nonmember = load_tofu_qa("holdout10", n=N)
    retain = load_tofu_qa("retain90", n=N)
    rows = []

    # base-Phi: supplies the null AND is audited as the negative control.
    basephi = load_model(BASE, device)
    null = bootstrap_gold_null(basephi, member, nonmember, scorer=scorer, device=device,
                               batch_size=1, n_boot=N_BOOT, seed=0)
    bp_res, bp_rep = _case(basephi, member, nonmember, retain, scorer, device, null)
    bp_facc = answer_accuracy(basephi, member, tok, device)
    rows.append(("base-Phi null (microsoft/phi-1_5)", bp_facc, bp_res, bp_rep))
    _free(basephi)

    # baseline + guardrail (guardrail wraps baseline -> identical scoring path).
    baseline = load_model(BASELINE, device)
    bl_res, bl_rep = _case(baseline, member, nonmember, retain, scorer, device, null)
    bl_facc = answer_accuracy(baseline, member, tok, device)
    rows.append(("baseline (tofu_ft_phi-1.5)", bl_facc, bl_res, bl_rep))

    guard = TextGuardrail(baseline, forget_questions=[member[i][0] for i in range(len(member))])
    g_res, g_rep = _case(guard, member, nonmember, retain, scorer, device, null)
    g_facc = answer_accuracy(guard, member, tok, device)
    rows.append(("guardrail (wrap baseline)", g_facc, g_res, g_rep))
    _free(baseline)

    c = bp_rep.calibration
    far = "n/a" if c.false_alarm_rate is None else f"{c.false_alarm_rate:.3f}"
    print("=" * 92)
    print("TEXT AUDIT — TOFU / Phi-1.5, loss-MIA via the CORE auditor (training-free)")
    print("=" * 92)
    print(f"member=forget10 (n={N})  non-member=holdout10 (n={N})   "
          f"base-Phi null: n={len(null)} threshold={c.threshold:.3f} FAR={far}")
    print(f"{'case':<34}{'forget-ans-acc':<15}{'member_nll':<12}{'nonmem_nll':<12}"
          f"{'MIA-AUC':<9}{'retention':<10}decision")
    for label, facc, res, rep in rows:
        d = res.detail
        print(f"{label:<34}{facc:<15.3f}{d['member_loss_mean']:<12.3f}{d['nonmember_loss_mean']:<12.3f}"
              f"{d['auc']:<9.3f}{rep.retention_score:<10.3f}{_decision(rep)}")
    print("=" * 92)
    print("money-shot: guardrail member_nll / nonmem_nll / AUC must be BYTE-IDENTICAL to baseline")
    print("            (filter left scoring untouched) while forget-ans-acc collapsed to ~0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
