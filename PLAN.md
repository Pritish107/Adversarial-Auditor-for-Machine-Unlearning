# PLAN — Adversarial Auditor for Machine Unlearning

Running tracker of milestones, decisions, and open questions. Source of truth for *goals*
is [PROJECT.md](PROJECT.md); this file tracks *execution state*.

Last updated: 2026-07-16 (M1 plumbing: CIFAR switch + guardrail + first calibration).

---

## Milestones

### M0 — v0 hello-world (image classifier)  ← IN PROGRESS
One unlearning setup + ONE membership-inference attack producing a forgetting score, with a
clean calibration seam. Prototype = small CNN on MNIST (CIFAR-10 is a config knob). No LLMs.

- [x] Repo scaffold: `data / models / unlearn / attacks / audit / report` + tests + README
- [x] `.venv` + pinned deps (torch cu124, torchvision, numpy, sklearn, pyyaml, pytest)
- [x] `data.py` — MNIST load + deterministic forget/retain/test split (sizes are knobs)
- [x] `models.py` — small CNN + train/eval + per-sample loss
- [x] `unlearn/` — Unlearner interface + gradient-ascent baseline + gold retrain reference
- [x] `attacks/` — Attack interface + loss-based membership inference
- [x] `audit/` — retention aggregation + forgetting score + calibration STUB seam
- [x] `report.py` + `cli.py` — end-to-end runner + readable before/after report
- [x] tests — interface contracts + tiny-subset smoke test
- [x] Run full pipeline; confirm it runs end-to-end; first commit (f9bd9a6)
- [x] Sanity-check: MNIST MIA signal was ~chance (AUC 0.507) -> switched default to CIFAR-10

### M1 — v0 hardening / labeled eval seeds  ← plumbing DONE
- [x] Switched default to CIFAR-10 for a natural train/test gap (real loss-MIA signal)
- [x] Three ground-truth-labeled cases: gold (truly forgotten) / gradient_ascent (crude) /
      guardrail (fake output filter, weights intact) + baseline control
- [x] Guardrail fake-unlearner: frozen weights + perfect-hash detector + output-label filter;
      `evaluate()` split so accuracy uses `predict`, loss uses raw `forward`
- [x] First real calibration: bootstrap the gold model -> null -> threshold at target FAR
- [x] ORDERING RESULT (CIFAR-10, seed 0) — HOLDS:
        gold      retention=0.012  MIA-AUC=0.506  -> forgotten (clean)
        grad_asc  retention=0.297  MIA-AUC=0.648  -> RETENTION DETECTED
        guardrail retention=0.462  MIA-AUC=0.731  -> RETENTION DETECTED  (== baseline exactly)
      Headline: guardrail forget-ACC=0.000 (naive eval: "perfectly forgotten") but MIA-AUC
      equals the un-unlearned baseline -> auditor catches 100% residual retention.
      Calibration: threshold=0.066, FAR=0.050 (target met). Baseline train/test gap ~43pt.
- [ ] Detection AUC harness (clean vs. tampered) across more cases
- [ ] Multi-seed gold null (removes the calibration caveat below)

### M2 — LLM / TOFU  (the architecture-preservation test)
- [ ] Add TOFU behind the SAME data abstraction (verify current HF name/config vs. installed
      `datasets` version — do NOT guess the API; download only when we reach this milestone)
- [ ] Small open LLM (GPT-2 / Pythia-scale) + a real unlearning baseline
- [ ] Reuse the SAME `Attack` interface for loss-MIA on text (per-sequence token NLL)
- [ ] Confirm: did the interface survive? Log any minimal change needed BEFORE refactoring.

### M3 — calibrated score + attack battery
- [ ] Real calibration: null distribution from clean cases -> threshold at target FAR
- [ ] Attack #2: relearning-speed probe (uses `AttackContext.retrain_fn` seam)
- [ ] Attack #3: activation probing
- [ ] Combined calibrated forgetting score with stated false-alarm rate

### M4 — evaluation + paper
- [ ] Full eval across methods/benchmarks; ablations; ranking correlation; robustness
- [ ] Write-up, limitations, repo polish

---

## Key design decisions (v0)

- **Flat package layout** (`unlearn_audit/` at root) so `python -m unlearn_audit.cli` runs
  with no editable install — one less Windows failure mode.
- **Score convention:** attacks report `retention_score ∈ [0,1]` (higher = more residual
  retention). Aggregate = `max` (fail if ANY attack recovers info). `forgetting = 1 - retention`.
- **`AttackContext` is deliberately over-general** (member/non-member data, optional
  reference model, optional `retrain_fn`) so relearning-speed / activation probing / the LLM
  setting reuse it without interface churn.
- **Calibration is now real (first-cut)**: gold model bootstrapped -> null -> threshold at
  the `(1 - target_far)` quantile -> per-case decision + measured FAR. Falls back to an
  explicit uncalibrated result (`far=None`) only when no null is supplied — never a fake number.
- **CIFAR-10 default** (natural memorization gap -> real MIA signal); MNIST kept only as the
  fast smoke-test config (`configs/test_tiny.yaml`).

## Named caveats (have the answer written — reviewers will ask)

### C1 — Threat-model boundary of the guardrail result
The guardrail is caught **only because the auditor accesses the model's SCORING function
(per-sample loss), not just the filtered output labels.** Our guardrail models output-LABEL
filtering: it corrupts `predict()` (the shown answer) while `forward()` logits stay intact,
so loss-MIA still sees the membership trace. **A guardrail that also corrupted the logits
(not just the argmax) would defeat loss-MIA specifically.** This is precisely the motivation
for the audit being a **battery, not one attack**: M3's activation-probing and relearning-speed
probes read signals a logit-corrupting filter cannot cheaply fake. State this boundary
explicitly in the paper next to the guardrail result — it is the first reviewer question.

### C2 — Calibration FAR is a lower bound (optimistic)
The v0 null is bootstrapped from **one** gold retrain, so it captures member/non-member
**sampling** variance but **not model/seed** variance. Real truly-forgotten models vary
run-to-run; ignoring that makes the reported FAR **optimistically low — a lower bound** on
the true false-alarm rate. Honest version = multiple gold retrains at different seeds pooled
into the null. v0 FAR is labeled "first-cut" in the report accordingly.

## Open questions

1. **MIA signal strength on MNIST** — gradient-ascent may leave only a weak membership gap on
   an easy dataset. If the before/after drop is tiny, bump the default to CIFAR-10. (Decide
   after first real run.)
2. **Forgetting-score direction** — reporting both `retention` and `forgetting = 1 - retention`.
   Confirm which the paper leads with. (Leaning: auditor's native output is *retention*, since
   the auditor's job is to disprove forgetting.)
3. **Gold as calibration anchor** — gold retrain is currently just displayed; M1 promotes it to
   the seed of the null distribution for real calibration.
4. **PROJECT.md attack-2 ordering** — spec lists relearning-speed as attack #2 (before
   activation probing). Interface already accommodates it via `retrain_fn`.
