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

## ★ v0 HEADLINE RESULT (for the paper — pull straight from here)

**guardrail−gold MIA-AUC gap = 0.220 ± 0.030 (N=5 seeds), gold 0/5 false positives,
ordering (gold < gradient_ascent < guardrail) preserved 5/5.**
CIFAR-10, small CNN, loss-MIA. Per-seed gap: [0.225, 0.171, 0.225, 0.225, 0.254].
Seed-aware calibration: pooled null of 5 independent gold retrains, threshold 0.074, FAR 0.050.
Interpretation: a guardrail fake-unlearner with forget-accuracy ~0.01 (naive eval calls it
forgotten) is caught with MIA-AUC identical to the un-unlearned baseline, while the truly-
forgotten gold sits at chance (AUC 0.498 ± 0.024). This is the "auditor catches retention
that standard evals declare forgotten" claim, with error bars.

---

### M1 — v0 hardening / labeled eval seeds  ← DONE
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
- [x] MULTI-SEED (N=5) hardening — see ★ headline above. Per-case mean±std, gap distribution,
      ranking stability (5/5), and seed-aware pooled-null calibration all via
      `python -m unlearn_audit.multiseed`. Ordering robust; gap tight; separation perfect.
- [ ] Detection AUC harness (clean vs. tampered) across more cases — deferred to post-v0

### M2 — LLM / TOFU  (the architecture-preservation test)

**M2a (de-risking spike) — verified facts:**
- Deps: `transformers==5.14.1`, `datasets==5.0.0`, `accelerate==1.14.0` installed + pinned.
- TOFU dataset id = **`locuslab/TOFU`** (canonical; 100k downloads). **Plain JSON data repo, NO
  loading script** → datasets 5.0.0 loads it (verified empirically). 18 configs; schema is
  `{question, answer}` strings; each config is a single `train` split. Forget/retain are
  **paired configs** (`forget10` + `retain90`), not splits. `holdout*` = same-distribution
  never-trained QAs = the natural MIA non-member set.
- Model = **`locuslab/tofu_ft_phi-1.5`** (Phi-1.5, 1.3B, ~2.6 GB, fits the 4060). It is
  **TOFU-finetuned** (members baked in), so the NLL check needs no training. The smaller
  `tofu_ft_gpt2` / `tofu_ft_pythia-1.4b` do NOT exist (404-verified); Llama-2-7B is the only
  other official finetune (13.5 GB, too big for the 4060 now).
- [x] M2a step 4 — GREEN LIGHT. Per-token (length-normalized) answer-only NLL, n=100 each:
      forget10 (member) mean=0.090 vs holdout10 (non-member) mean=2.123 -> gap +2.03,
      ZERO overlap (forget max 0.418 < holdout min 0.548), descriptive AUC=1.000. The Phi-1.5
      finetune memorized the forget answers (~0.09 NLL/token); holdout unseen (~2.1). Signal is
      huge -> big baseline for M2b's unlearning to erode. Length-normalized + answer-only span
      (identical template both sets), so NOT a length artifact.

**M2a GOTCHA (must handle in M2b code):** TOFU finetune repos ship NO tokenizer files.
`AutoTokenizer.from_pretrained("locuslab/tofu_ft_phi-1.5")` does NOT error — it silently
returns a GPT2Tokenizer with **vocab_size=0** (broken). Load the tokenizer from the BASE
model `microsoft/phi-1_5` (CodeGenTokenizer, vocab 50257). Also: the MODEL repo id
(`locuslab/tofu_ft_phi-1.5`) and the DATA repo id (`locuslab/TOFU`) are distinct — don't
cross them in `load_dataset`.

**M2b RUNWAY — ⚠️ RETRACTED, THE PRE-BAKED CHECKPOINTS ARE BROKEN (verified 2026-07-25):**
The earlier "whole trio exists pre-baked, no LLM training" note was WRONG. Verified against HF:
- `tofu_ft_phi-1.5` (members/baseline): **REAL, works** (reproduced the spike through LossMIA).
- `tofu_ft_retain90_phi-1.5` (intended GOLD): **byte-identical DUPLICATE of the full model**
  (raw-disk max abs weight diff = 0.0; stored fp32 vs baseline bf16, same values). Memorized
  forget10 identically -> USELESS as a truly-forgotten reference.
- `phi_grad_ascent_* / phi_grad_diff_* / phi_KL_* / phi_idk_*` (ALL forget fractions) and the
  llama2 unlearned repos: **EMPTY** — contain only `.gitattributes` + `README.md`, NO weights.

Consequence: only the baseline model is usable. A valid GOLD/null and the UNLEARNED models to
audit must come from elsewhere. Gold substitute under evaluation = base `microsoft/phi-1_5`
(never saw forget10 or holdout10 -> should score ~chance). Unlearned models will likely have to
be PRODUCED locally (grad-ascent/etc. on Phi-1.5) — which reintroduces (light) LLM training —
OR the no-training first cut uses the training-free guardrail strawman only.

**DECIDED (2026-07-25):** M2b scope = the TRAINING-FREE guardrail headline, mirroring CIFAR:
baseline (members) + guardrail wrapper (frozen weights, output filter deflects forget10) +
base-Phi null. Reproduces the exact validated CIFAR result in text — forget-accuracy -> 0 but
per-token NLL untouched -> MIA-AUC ~1.0 -> RETENTION DETECTED — through the same auditor. That
IS the cross-modal generalization claim. No unlearning checkpoints needed.

- [x] Gold/null substitute = base `microsoft/phi-1_5` (never saw forget10 OR holdout10). Doubles
      as the auditor's NEGATIVE CONTROL: proves the audit reads FORGOTTEN on a never-memorized
      model, not always "retention". Validated by: forget10-vs-holdout10 AUC ~0.5 via same path.
- [ ] M2b build: text guardrail (predict-path deflects on forget10; forward/NLL untouched);
      run baseline + guardrail + base-Phi-null through the real `audit()`; three-case text result.

**M3 LIMITATION (log now, state in paper):** the "official" locuslab TOFU-unlearned Phi
checkpoints (`phi_grad_ascent_* / phi_grad_diff_* / phi_KL_* / phi_idk_*`) are EMPTY repos —
weights were never uploaded, so they CANNOT be downloaded or reproduced. M3's real unlearning
methods must therefore be TRAINED BY US with our own (known, logged) hyperparameters. Upside:
more trustworthy/reproducible than the corrupted official repos. Cost: our unlearned-model
results are NOT directly comparable to published TOFU leaderboard numbers — a stated scoping
limitation, not a flaw.

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

### C2 — Calibration is seed-aware now, but the FAR tail is still limited by N
Status: **seed-aware (pools N independent gold retrains); FAR tail still limited by N=5
true draws.** The multi-seed harness pools the gold models from all N seeds — each a genuinely
independent retrain (different init AND different forget-set draw) — into one null, so the null
now captures **model/seed variance**, not just sampling variance. That is a real improvement,
**not a fix**: the tail quantile at FAR=0.05 still rests on only **N=5 truly-independent draws**,
each contributing 200 *correlated* bootstrap resamples. So n=1000 null points is **not** 1000
independent observations — the effective independent sample size for the tail is ~5. The FAR
is far better grounded than the single-gold v0 number, but the 5th-percentile tail is still
coarse; more independent gold seeds (not more bootstrap resamples) are what tighten it.

Earlier single-gold state (superseded, kept for context): bootstrapping ONE gold captured
sampling variance only, making the FAR an optimistic lower bound.

### C3 — LLM null (base-Phi) carries a text-difficulty offset (two faces)
Base `microsoft/phi-1_5` on forget10-vs-holdout10 scores **MIA-AUC 0.294** (member NLL 2.071 >
non-member 1.684). It never saw either set, so this is NOT membership — it is a **difficulty
asymmetry between the two TOFU question sets**: forget10 answers are intrinsically ~0.39
nats/token harder for an untrained model. The offset runs in the SAFE direction (makes forget10
look *less* member-like → cannot cause false alarms). Two faces:
  * As a NEGATIVE CONTROL ("does the auditor say forgotten?"): **passes cleanly, retention 0.**
  * As a CALIBRATION NULL ("what threshold is forgotten?"): **0.294 is worse than optimistic —
    it is DEGENERATE.** Base-Phi's AUC is so far below 0.5 that every bootstrap clips to
    retention 0, so the null is a **point mass at 0** -> threshold 0. With the old `>=` decision
    rule this flagged the null itself (base-Phi mislabeled "detected", FAR reported 1.000 — the
    calibrator honestly signalling its own null was degenerate).

**Operator fix (adopted):** decision rule is now **strict `>`** (flag what EXCEEDS the forgotten
baseline, not what equals it), and FAR = `mean(null > threshold)`. This makes base-Phi a USABLE
null for the guardrail HEADLINE (base-Phi 0 -> forgotten, FAR 0; baseline/guardrail 1.0 ->
detected; signal towers at 1.0 so the point-mass null suffices). It does **NOT** make base-Phi a
GOOD calibration null: a point-mass-at-0 null has no spread and **cannot calibrate a GRADED
result**. For M3's real unlearning methods (which may land at partial retention, e.g. ~0.3), a
**difficulty-matched non-member set is MANDATORY, not optional** — the operator fix is not a
substitute for it. (forget10/holdout10 are not difficulty-matched: base-Phi AUC 0.294.)

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
