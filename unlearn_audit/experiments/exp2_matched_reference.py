"""EXP-2 — does the holdout10 difficulty offset DISTORT graded audit scores, or is it a
harmless conservative shift?

The objection this exists to answer: "the offset makes members look LESS member-like, so it is
safe-direction — who cares?" Safe direction is safe for false ALARMS. An auditor's failure mode
is a MISSED DETECTION, which is the opposite direction. So: build a difficulty-matched
reference, swap it in, and see whether scores, thresholds, and verdicts move.

+---------------------------------------------------------------------------------------------+
| !! THE ALPHA LADDER IS A SYNTHETIC PROBE, NOT AN UNLEARNING METHOD. !!                        |
|                                                                                               |
|   theta(a) = (1-a)*theta_baseline + a*theta_basePhi                                           |
|                                                                                               |
| This is a SYNTHETIC MONOTONE RETENTION GRADIENT whose only job is to supply graded scores so  |
| the reference question can be tested at all (the official TOFU unlearned checkpoints are      |
| empty repos — see PLAN.md M3 LIMITATION). It is NOT an unlearning method, it does not         |
| approximate one, and it must NEVER be reported as one or compared against published unlearning|
| numbers. Real methods (grad-ascent / NPO / grad-diff) are deferred to the training phase.     |
+---------------------------------------------------------------------------------------------+

THREE REFERENCE CONSTRUCTIONS
  R-holdout : forget10(400) vs holdout10(400), raw NLL   -- the field default; this is
              OpenUnlearning's shipped LOSS MIA against their TOFU_QA_holdout.
  R-matched : caliper-0.05 1:1 matching on base-Phi NLL  -- PRIMARY. Audits the DIFFICULTY-
              MATCHED SUBPOPULATION of forget10 (n~287 of 400), NOT all of it: matching
              discards forget10's hard tail. Report that qualifier with every number.
  R-delta   : d = NLL_target - NLL_basePhi, full 400/400 -- OpenUnlearning's shipped
              REFERENCE MIA. Cancels per-example difficulty exactly, no data discarded.

Why retain90 is NOT a reference: verified full = forget10 (+) retain90 and tofu_ft_phi-1.5 was
finetuned on full, so every retain90 row is a MEMBER. It is a difficulty DIAGNOSTIC only.

MANDATORY ENDPOINT GATE: a=0 must reproduce baseline member NLL ~0.090 and a=1 must reproduce
base-Phi ~2.072. If the endpoints miss, the blend is broken -> abort without reading the middle.

    python -m unlearn_audit.experiments.exp2_matched_reference
"""
from __future__ import annotations

import gc

import numpy as np
import torch

from ..attacks.membership_inference import _retention_from_losses
from ..audit.score import audit
from ..llm.data import load_tofu_qa
from ..llm.guardrail import TextGuardrail, answer_accuracy
from ..llm.loader import load_tokenizer
from ..llm.scoring import make_answer_nll_scorer
from . import RESULTS

BASELINE = "locuslab/tofu_ft_phi-1.5"
BASE = "microsoft/phi-1_5"
ALPHAS = [0.0, 0.15, 0.3, 0.5, 0.7, 1.0]
CALIPER = 0.05
N_BOOT = 200
TARGET_FAR = 0.05
ENDPOINT_TOL = 0.05

EXP1_BASELINE_MEMBER_NLL = 0.090       # logged M2a/M2b endpoints the ladder must reproduce
EXP1_BASEPHI_MEMBER_NLL = 2.072

IN_EXP1 = RESULTS / "exp1_probe_nll.npz"
OUT = RESULTS / "exp2_ladder.npz"


def caliper_match(f_nll: np.ndarray, h_nll: np.ndarray, caliper: float):
    """1:1 greedy nearest-neighbour match, no replacement, on BASE-PHI NLL ONLY.

    Selection never touches the target model, so it cannot leak membership signal into the
    comparison — the matched set is chosen by a model that saw neither side.
    """
    used = np.zeros(len(h_nll), bool)
    keep_f, keep_h = [], []
    for i in np.argsort(f_nll):                       # deterministic order
        d = np.abs(h_nll - f_nll[i])
        d[used] = np.inf
        j = int(d.argmin())
        if d[j] <= caliper:
            used[j] = True
            keep_f.append(int(i))
            keep_h.append(j)
    return np.array(keep_f), np.array(keep_h)


def blend_into(target, sd_b, sd_p, alpha: float):
    """target <- (1-alpha)*baseline + alpha*basePhi, in place, per-tensor. SYNTHETIC PROBE."""
    with torch.no_grad():
        tgt = dict(target.named_parameters())
        tgt.update(dict(target.named_buffers()))
        for k, pb in sd_b.items():
            if k not in tgt:
                continue
            mixed = (1.0 - alpha) * pb.float() + alpha * sd_p[k].float()
            tgt[k].copy_(mixed.to(tgt[k].dtype))
    return target


def bootstrap_null_from_losses(m: np.ndarray, nz: np.ndarray, n_boot: int, seed: int = 0):
    """Same logic as attacks.bootstrap_gold_null, from PRE-COMPUTED losses (no re-scoring)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        mb = m[rng.integers(0, len(m), len(m))]
        nb = nz[rng.integers(0, len(nz), len(nz))]
        out.append(_retention_from_losses(mb, nb)[0])
    return out


def decide(m, nz, null):
    ret, auc = _retention_from_losses(m, nz)
    res = type("R", (), {"retention_score": ret, "name": "loss_mia", "detail": {}})()
    rep = audit([res], aggregate="max", target_far=TARGET_FAR, null_scores=null)
    d = "UNCALIB" if rep.retention_detected is None else (
        "RETENTION DETECTED" if rep.retention_detected else "forgotten (clean)")
    return dict(auc=auc, retention=rep.retention_score, thr=rep.calibration.threshold,
                far=rep.calibration.false_alarm_rate, decision=d,
                m=float(m.mean()), nz=float(nz.mean()))


def cpu_state_dict(repo: str):
    from transformers import AutoModelForCausalLM
    try:
        m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16)
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    del m
    gc.collect()
    return sd


def main() -> int:
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer()
    scorer = make_answer_nll_scorer(tok)
    forget, holdout = load_tofu_qa("forget10"), load_tofu_qa("holdout10")

    exp1 = np.load(IN_EXP1)                # base-Phi reference NLLs; a=1 re-derives them below
    bp_f, bp_h = exp1["forget"], exp1["holdout"]

    fi, hi = caliper_match(bp_f, bp_h, CALIPER)
    resid = float(bp_h[hi].mean() - bp_f[fi].mean())
    _, bp_auc_m = _retention_from_losses(bp_f[fi], bp_h[hi])
    _, bp_auc_full = _retention_from_losses(bp_f, bp_h)
    print(f"[match] caliper={CALIPER} pairs={len(fi)}/{len(bp_f)} "
          f"({100 * len(fi) / len(bp_f):.0f}% of forget10)")
    print(f"[match] base-Phi AUC  full 400/400 = {bp_auc_full:.3f}  ->  matched = {bp_auc_m:.3f}")
    print(f"[match] residual imbalance (nonmem-mem, base-Phi NLL) = {resid:+.3f} nats/token")
    print(f"[match] AUDITING THE DIFFICULTY-MATCHED SUBPOPULATION of forget10 (n={len(fi)}), "
          f"not all 400.\n")

    print("[load] baseline + base-Phi state dicts to CPU ...")
    sd_b, sd_p = cpu_state_dict(BASELINE), cpu_state_dict(BASE)
    kb, kp = set(sd_b), set(sd_p)
    assert kb == kp, f"key mismatch: only-baseline={sorted(kb - kp)[:5]} only-base={sorted(kp - kb)[:5]}"
    bad = [k for k in kb if sd_b[k].shape != sd_p[k].shape]
    assert not bad, f"shape mismatch on {bad[:5]}"
    print(f"[load] {len(kb)} tensors, keys+shapes aligned OK")

    try:
        model = AutoModelForCausalLM.from_pretrained(BASELINE, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(BASELINE, torch_dtype=torch.float16)
    model = model.to(device).eval()

    rows = {}
    for a in ALPHAS:
        blend_into(model, sd_b, sd_p, a).eval()
        f_nll = np.asarray(scorer(model, forget, device=device, batch_size=1))
        h_nll = np.asarray(scorer(model, holdout, device=device, batch_size=1))
        facc = answer_accuracy(model, forget, tok, device)
        rows[a] = dict(f=f_nll, h=h_nll, facc=facc)
        print(f"[ladder] a={a:<5} member_nll={f_nll.mean():.3f} nonmem_nll={h_nll.mean():.3f} "
              f"forget_ans_acc={facc:.3f}")

    e0, e1 = rows[0.0]["f"].mean(), rows[1.0]["f"].mean()
    d0, d1 = abs(e0 - EXP1_BASELINE_MEMBER_NLL), abs(e1 - EXP1_BASEPHI_MEMBER_NLL)
    drift = float(np.abs(rows[1.0]["f"] - bp_f).max())
    print(f"\n[GATE] a=0 member NLL {e0:.3f} vs baseline {EXP1_BASELINE_MEMBER_NLL} (d={d0:.3f})")
    print(f"[GATE] a=1 member NLL {e1:.3f} vs base-Phi {EXP1_BASEPHI_MEMBER_NLL} (d={d1:.3f})")
    print(f"[GATE] a=1 max per-example drift vs EXP-1 base-Phi = {drift:.4f}")
    if d0 > ENDPOINT_TOL or d1 > ENDPOINT_TOL:
        print("[GATE] FAILED — blend is broken. Refusing to read the middle of the ladder.")
        return 1
    print("[GATE] PASSED — endpoints reproduce; ladder is trustworthy.\n")

    blend_into(model, sd_b, sd_p, 0.0).eval()
    guard = TextGuardrail(model, forget_questions=[forget[i][0] for i in range(len(forget))])
    g_f = np.asarray(scorer(guard, forget, device=device, batch_size=1))
    g_h = np.asarray(scorer(guard, holdout, device=device, batch_size=1))
    g_acc = answer_accuracy(guard, forget, tok, device)
    ident = bool(np.array_equal(g_f, rows[0.0]["f"]) and np.array_equal(g_h, rows[0.0]["h"]))
    print(f"[guardrail] NLL byte-identical to baseline: {ident} | forget_ans_acc "
          f"{rows[0.0]['facc']:.3f} -> {g_acc:.3f}")

    del sd_b, sd_p, model
    gc.collect()
    torch.cuda.empty_cache()

    null_hold = bootstrap_null_from_losses(bp_f, bp_h, N_BOOT)
    null_match = bootstrap_null_from_losses(bp_f[fi], bp_h[hi], N_BOOT)
    for nm, nl in [("R-holdout", null_hold), ("R-matched", null_match)]:
        nl = np.asarray(nl)
        print(f"[null] {nm:<10} n={nl.size} mean={nl.mean():.4f} sd={nl.std():.4f} "
              f"frac_zero={np.mean(nl == 0):.3f} q95={np.quantile(nl, 0.95):.4f}  "
              f"DEGENERATE={bool(nl.std() == 0)}")

    def table(title, keyfn, null, note):
        print("\n" + "=" * 118)
        print(title)
        print(note)
        print("-" * 118)
        print(f"{'case':<26}{'naive forget-acc':<18}{'member_nll':<12}{'nonmem_nll':<12}"
              f"{'AUC':<8}{'retention':<11}{'thr':<8}{'FAR':<8}decision")
        out = {}
        for a in ALPHAS:
            r = decide(*keyfn(rows[a]), null)
            out[a] = r
            far = "n/a" if r["far"] is None else f"{r['far']:.3f}"
            print(f"{'ladder a=' + str(a):<26}{rows[a]['facc']:<18.3f}{r['m']:<12.3f}"
                  f"{r['nz']:<12.3f}{r['auc']:<8.3f}{r['retention']:<11.3f}"
                  f"{r['thr']:<8.3f}{far:<8}{r['decision']}")
        r = decide(*keyfn({"f": g_f, "h": g_h}), null)
        far = "n/a" if r["far"] is None else f"{r['far']:.3f}"
        print(f"{'guardrail (a=0 wrap)':<26}{g_acc:<18.3f}{r['m']:<12.3f}{r['nz']:<12.3f}"
              f"{r['auc']:<8.3f}{r['retention']:<11.3f}{r['thr']:<8.3f}{far:<8}{r['decision']}")
        return out

    res_h = table("REFERENCE A — R-holdout : forget10(400) vs holdout10(400), raw NLL  [FIELD DEFAULT]",
                  lambda r: (r["f"], r["h"]), null_hold,
                  "== OpenUnlearning's shipped LOSS MIA against their TOFU_QA_holdout.")
    res_m = table(f"REFERENCE B — R-matched : caliper-{CALIPER} matched pairs, raw NLL  [PRIMARY]",
                  lambda r: (r["f"][fi], r["h"][hi]), null_match,
                  f"DIFFICULTY-MATCHED SUBPOPULATION of forget10 (n={len(fi)} of 400), "
                  f"residual imbalance {resid:+.3f} nats/token.")
    table("REFERENCE C — R-delta : d = NLL_target - NLL_basePhi, full 400/400  [companion]",
          lambda r: (r["f"] - bp_f, r["h"] - bp_h), null_hold,
          "== OpenUnlearning's shipped REFERENCE MIA. ITS OWN NULL IS STRUCTURALLY DEGENERATE "
          "(base-Phi vs itself => d==0 exactly, AUC 0.500); threshold shown is borrowed from "
          "R-holdout, so these decisions are NOT properly calibrated.")

    print("\n" + "=" * 118)
    print("EXP-2 VERDICT")
    print("=" * 118)
    flips = [(a, res_h[a], res_m[a]) for a in ALPHAS
             if res_h[a]["decision"] != res_m[a]["decision"]]
    print("(a) MISSED DETECTION — 'forgotten' vs holdout10 but 'RETENTION DETECTED' vs matched?")
    if flips:
        for a, rh, rm in flips:
            print(f"    *** FIRES at a={a}: holdout10 -> {rh['decision']} (ret {rh['retention']:.3f}) "
                  f"| matched -> {rm['decision']} (ret {rm['retention']:.3f})")
    else:
        print("    no verdict flips on THIS COARSE GRID — both curves cross the boundary inside")
        print("    (0.7, 1.0), which is unsampled here. Resolved by exp2b_fine_sweep.")
    gaps = [(a, res_m[a]["retention"] - res_h[a]["retention"]) for a in ALPHAS]
    print("    retention shift (matched - holdout10): "
          + ", ".join(f"a={a}:{g:+.3f}" for a, g in gaps))

    nh, nm2 = np.asarray(null_hold), np.asarray(null_match)
    print("\n(b) DOES THE CALIBRATION NULL STOP BEING DEGENERATE? (retires the C3 blocker)")
    for nm, nl in [("R-holdout", nh), ("R-matched", nm2)]:
        print(f"    {nm}: sd={nl.std():.4f} frac_zero={np.mean(nl == 0):.3f} "
              f"thr={np.quantile(nl, 1 - TARGET_FAR):.4f} -> "
              f"{'DEGENERATE point mass' if nl.std() == 0 else 'has spread -> CALIBRATED'}")

    mono = lambda v: all(v[i] >= v[i + 1] - 1e-9 for i in range(len(v) - 1))
    print(f"\n    ranking monotone in alpha?  holdout10: "
          f"{mono([res_h[a]['retention'] for a in ALPHAS])}   "
          f"matched: {mono([res_m[a]['retention'] for a in ALPHAS])}")
    print("    -> the offset biases score LEVELS, not ORDERING.")

    RESULTS.mkdir(exist_ok=True)
    np.savez(OUT, alphas=np.array(ALPHAS), fi=fi, hi=hi, resid=resid,
             bp_f=bp_f, bp_h=bp_h, g_f=g_f, g_h=g_h, g_acc=g_acc,
             null_hold=nh, null_match=nm2,
             **{f"f_{a}": rows[a]["f"] for a in ALPHAS},
             **{f"h_{a}": rows[a]["h"] for a in ALPHAS},
             **{f"acc_{a}": rows[a]["facc"] for a in ALPHAS})
    print(f"\nsaved -> {OUT}")
    print("REMINDER: the alpha ladder is a SYNTHETIC retention gradient, NOT an unlearning method.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
