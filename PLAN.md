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
- [x] M2b build: text guardrail (predict-path deflects on forget10; forward/NLL untouched);
      run baseline + guardrail + base-Phi-null through the real `audit()`; three-case text result.
      (Landed in d45075d — see that commit message for the three-case table.)

**M3 LIMITATION (log now, state in paper):** the "official" locuslab TOFU-unlearned Phi
checkpoints (`phi_grad_ascent_* / phi_grad_diff_* / phi_KL_* / phi_idk_*`) are EMPTY repos —
weights were never uploaded, so they CANNOT be downloaded or reproduced. M3's real unlearning
methods must therefore be TRAINED BY US with our own (known, logged) hyperparameters. Upside:
more trustworthy/reproducible than the corrupted official repos. Cost: our unlearned-model
results are NOT directly comparable to published TOFU leaderboard numbers — a stated scoping
limitation, not a flaw.

### M3 — calibrated score + attack battery

**M3 EXP-1 (2026-07-25) — does the C3 difficulty asymmetry survive OpenUnlearning's IMPROVED
holdout? ANSWER: THE QUESTION DISSOLVES — they are the SAME dataset — and the asymmetry is REAL,
holdout-specific, and confirmed at full scale.**

*Dataset provenance (verified, no guessing — same discipline as M2a):*
- OpenUnlearning's TOFU MIA config `configs/data/datasets/TOFU_MIA.yaml` defines exactly two
  entries: `TOFU_QA_forget` -> **path `locuslab/TOFU`, name `forget10`, split `train`**, and
  `TOFU_QA_holdout` -> **path `locuslab/TOFU`, name `holdout10`, split `train`**.
  **Their improved holdout IS `locuslab/TOFU`/`holdout10` — the very config we already used in C3.**
- The org `huggingface.co/open-unlearning` publishes only TWO datasets, `open-unlearning/eval`
  (450+ checkpoint eval JSONs) and `open-unlearning/idk`. **There is no separate holdout repo.**
- `holdout01/05/10` were added to `locuslab/TOFU` on **2025-03-27 by pratyushmaini** (uploaded as
  `.jsonl`, renamed to `.json` two minutes later) — ~14 months AFTER the original TOFU release.
  They are the OpenUnlearning-era "new holdout datasets replicating the TOFU generation setup",
  not a legacy artifact. **So there is no "deprecated raw holdout10" vs "improved holdout"
  distinction to test — holdout10 is the improved one.**
- Verified against the installed `datasets==5.0.0`: 18 configs; `holdout01/05/10` present with
  n=40/200/400, schema `{question, answer}`. `forget10` n=400 and `holdout10` n=400 are size-matched.

*Result — base `microsoft/phi-1_5` (never saw forget10, holdout10 OR retain90), answer-only
length-normalized NLL/token, repo's own scorer + AUC convention (member=forget10):*

    comparison                            n_m  n_nm  mem_nll  nonmem_nll  AUC     retention  nll_gap
    forget10 vs holdout10  [n=100 repro]  100  100   2.071    1.684       0.294   0.000      -0.387
    forget10 vs holdout10  [FULL 400/400] 400  400   2.072    1.748       0.332   0.000      -0.324
    CONTROL forget10 vs retain90 [400]    400  400   2.072    2.106       0.514   0.029      +0.034
    CONTROL retain90 vs holdout10 [400]   400  400   2.106    1.748       0.310   0.000      -0.357

- **The C3 n=100 number reproduced EXACTLY (0.294).** At full 400/400 it is **AUC 0.332**
  (nonmember NLL 1.748 vs member 2.072, gap **-0.324 nats/token**). The n=100 slice slightly
  overstated the effect; the full-set value is the one to quote. Still **materially off 0.5**.
- **-> STRONG/GENERAL BRANCH CONFIRMED.** The non-matched-reference problem persists on
  OpenUnlearning's own improved holdout — this is not an artifact of a stale split, and it is
  the reference set the field's leading unlearning benchmark harness actually uses for MIA.

*NEW — the control isolates the cause (this is the paper-grade part):*
- **`forget10` vs `retain90` under base-Phi: AUC 0.514, NLL gap +0.034 nats/token — essentially
  perfect chance.** Two TOFU splits from the SAME original generation run are difficulty-matched
  to within noise.
- **`retain90` vs `holdout10`: AUC 0.310** — i.e. holdout10 is offset from the TOFU training
  distribution by the same amount (-0.357) as forget10 is (-0.324).
- **Conclusion: the asymmetry is NOT "forget10 is intrinsically easy" — it is HOLDOUT-SPECIFIC.
  `holdout10` is systematically ~0.33 nats/token EASIER for a never-trained model than ANY
  original-TOFU split.** The later regeneration (different GPT-4 vintage/prompt state, 2025 vs
  2024) did not reproduce the original difficulty distribution. Not a length artifact either:
  scoring is length-normalized and answer-only, and holdout answers are LONGER (41.8 tokens mean
  vs 36.1 forget10 / 33.1 retain90), which would push the bias the other way if anything.
- Direction is still SAFE (makes members look *less* member-like -> no false alarms), so the M2b
  binary headline stands unchanged. But it means **any graded MIA number computed against
  holdout10 — including OpenUnlearning's own published TOFU MIA/PrivLeak-style metrics — carries
  this ~0.33 nat/token reference offset.** That is a finding about the benchmark, not just us.
- **Actionable for M3:** `retain90` is empirically difficulty-matched to `forget10` (AUC 0.514)
  and is the defensible non-member pool for graded results — but it was TRAINED ON by
  `tofu_ft_phi-1.5`, so it is NOT a valid non-member for that model. The matched-reference
  construction M3 needs is therefore: hold out a slice of the *original* TOFU distribution from
  our own finetune (we must train anyway — see M3 LIMITATION below), rather than adopting
  `holdout10`. DECISION DEFERRED — reporting only, per the EXP-1 brief.
- Repro: `unlearn_audit/experiments/exp1_difficulty_probe.py` (per-example NLLs in
  `experiments/results/exp1_probe_nll.npz`).

**M3 EXP-2 (2026-07-25) — does the holdout10 offset DISTORT graded scores, or is it a harmless
conservative shift? ANSWER: it is NOT harmless. It makes the auditor UNCALIBRATABLE and
under-reports graded retention by up to 3.3x at the decision boundary.**

*Why this experiment existed:* the reviewer objection to EXP-1 is "the offset is safe-direction,
so who cares". Safe direction is safe for false ALARMS. An auditor's failure mode is a MISSED
DETECTION, which is the opposite direction — hence this test.

*⚠️ THE INTERPOLATION LADDER IS A SYNTHETIC PROBE, NOT AN UNLEARNING METHOD.*
`theta(a) = (1-a)*theta_baseline + a*theta_basePhi` is a synthetic MONOTONE RETENTION GRADIENT
built solely to supply graded scores so the reference question can be tested. It must NEVER be
presented as an unlearning method or compared to one. Real methods (grad-ascent / NPO /
grad-diff) are deferred to the training phase. Endpoint gate (mandatory, PASSED): a=0
reproduced baseline member NLL 0.084 (logged 0.090); a=1 reproduced base-Phi 2.072 with
**max per-example drift 0.0000** vs the EXP-1 cache — i.e. a=1 independently re-derives the
reference rather than trusting it. 341/341 state-dict tensors key+shape aligned.

*Three reference constructions.* `retain90` is NOT usable — verified `full` = `forget10` (+)
`retain90` (4000 = 400 + 3600, disjoint, both subsets of `full`) and `tofu_ft_phi-1.5` was
finetuned on `full`, so every retain90 row is a MEMBER. It stays a difficulty DIAGNOSTIC only.
  - **R-holdout**: forget10(400) vs holdout10(400), raw NLL = the field default.
  - **R-matched** (PRIMARY): caliper-0.05 1:1 greedy matching on **base-Phi NLL only** (selection
    never touches the target, so no membership leaks into pair construction). n=287 of 400 (72%),
    residual imbalance **+0.015 nats/token**, base-Phi AUC 0.332 -> **0.509**. This audits the
    **DIFFICULTY-MATCHED SUBPOPULATION of forget10, not all 400** — matching discards forget10's
    hard tail (member base-Phi NLL 2.072 -> 1.844). State that qualifier with every number.
  - **R-delta**: d = NLL_target - NLL_basePhi, full 400/400.

*Ladder (retention scores; naive forget-answer-accuracy in brackets):*

    a      [naive acc]   R-holdout   R-matched   R-delta
    0.00   [0.989]       0.997       0.996       0.999
    0.15   [0.964]       0.993       0.991       0.997
    0.30   [0.875]       0.973       0.981       0.994
    0.50   [0.731]       0.733       0.878       0.985
    0.70   [0.640]       0.162       0.541       0.970
    1.00   [0.563]       0.000       0.019       0.000
    guardrail [0.000]    0.997       0.996       0.999   (NLL byte-identical to a=0: verified True)

**(b) FIRED — RETIRES THE C3 BLOCKER.** Difficulty-matching converts an UNCALIBRATABLE auditor
into a calibrated one:
  - R-holdout null (base-Phi bootstrap, n=200): **sd=0.0000, frac_zero=1.000, threshold=0.000**
    -> degenerate point mass. Exactly the C3 pathology. **No threshold at any target FAR exists.**
  - R-matched null: **sd=0.0358, frac_zero=0.320, threshold=0.0911, FAR=0.050** (target met)
    -> real spread, genuinely calibrated.
  This is the clean methods contribution: C3 said "a point-mass null cannot calibrate a GRADED
  result"; difficulty-matching is the construction that fixes it, with receipts.

**EXP-1 DELIVERABLE, QUANTIFIED — which of OpenUnlearning's six MIAs the offset corrupts.**
The offset corrupts **raw-score MIAs (LOSS, ZLib, MinK, MinK++)** and leaves **Reference**
intact, because per-example reference normalization cancels the difficulty term exactly.
Receipt at a=0.7 (the decision boundary): retention **0.162** (LOSS vs holdout10) vs **0.541**
(LOSS vs matched) vs **0.970** (Reference/delta) — a **3.3x under-estimate** by the field-default
path, at precisely the operating point where the forgotten/not decision is made.

**THIRD FINDING (new M3 constraint): the Reference-MIA null is STRUCTURALLY degenerate.**
Scoring base-Phi under R-delta compares it to itself, so d == 0 for every example, AUC = 0.500
exactly, retention 0, zero variance. So R-delta gives excellent SCORES but **cannot be calibrated
with base-Phi as the null** — the decisions in its table borrow R-holdout's threshold and are
NOT properly calibrated. **Calibrating a Reference MIA REQUIRES a null model DISTINCT from the
reference model.** This is a hard requirement on M3's design, not a nuance.

**FOURTH FINDING: the offset biases score LEVELS, not ORDERING.** Retention is monotone in a
under BOTH references (True/True). Reassuring for leaderboard-style method RANKING; NOT
reassuring for any threshold-based forgotten/not decision, which is what an auditor emits.

**(a) FIRED — benchmark-induced MISSED DETECTION over a ~ 0.80–0.93 (EXP-2b fine sweep).**
The coarse grid left this unresolved (no flips over a in {0,.15,.3,.5,.7,1}) because both curves
cross the decision boundary inside (0.7, 1.0), which was unsampled. Fine sweep resolves it:

    a      naive acc  hold AUC  hold ret  match AUC  match ret  (1) matched  (3) AUC>0.5  (4) hold@.091
    0.75   0.622      0.520     0.040     0.720      0.439      DETECTED     DETECTED     forgotten
    0.80   0.608      0.467     0.000     0.669      0.339      DETECTED     forgotten    forgotten
    0.85   0.594      0.423     0.000     0.623      0.245      DETECTED     forgotten    forgotten
    0.90   0.582      0.386     0.000     0.580      0.160      DETECTED     forgotten    forgotten
    0.95   0.573      0.355     0.000     0.541      0.081      forgotten    forgotten    forgotten
    retention shift (matched - holdout): +0.400, +0.339, +0.245, +0.160, +0.081

- **Fires on 3/5 rungs against rule (3)** and 4/5 against control (4). Matched auditor's decision
  boundary sits at a ~ 0.94, holdout10's at a ~ 0.76: **the offset costs ~1/5 of the detectable
  range.**
- **Ground truth is unambiguous by construction:** every a < 1 is a blend containing baseline
  weight, so retention IS present and the matched auditor is correct — the holdout10 verdict is a
  true FALSE NEGATIVE, not a disagreement between two defensible readings.
- **Control (4) isolates the cause to the REFERENCE SET, not the calibration:** transplanting the
  matched threshold (0.091) onto holdout scores still flips on 4/5 rungs. Same margin, only the
  reference set differs. This blocks the obvious rebuttal ("you compared a calibrated auditor to
  an uncalibrated one").
- **Rules (2) and (3) are mathematically IDENTICAL:** retention = clip(2*(AUC-0.5),0,1), so
  `retention > 0` <=> `AUC > 0.5`. The degenerate null's threshold of 0 lands exactly on the
  field's "MIA at/below chance = forgotten" convention. So the threshold-0 default was never a
  straw man — beating it IS beating real practice.
- Side observation: at a = 0.8–0.9 holdout AUC sits BELOW 0.5 (0.467/0.423/0.386). A one-sided
  auditor reads that as clean; a two-sided one would flag it, but as OVER-forgetting — the right
  alarm for the wrong reason.

**★ SCOPE — THIS TRAVELS WITH THE (a) CLAIM EVERYWHERE. NEVER STATE THE FLIP WITHOUT IT. ★**
The offset corrupts **ABSOLUTE-THRESHOLD** MIA auditing — the "AUC ~ 0.5 means forgotten"
convention, which is what standard practice and OUR auditor both do. **Gold-referenced /
relative rules (MUSE PrivLeak, TOFU forget_quality) are IMMUNE**: they compare a candidate to a
retrain model on the SAME non-member set, so the difficulty offset is **common-mode and cancels**.
Measured directly (candidate AUC minus gold AUC, per reference set):

    a      d_vs_gold (holdout)   d_vs_gold (matched)   discrepancy
    0.75   +0.188                +0.211                0.023
    0.80   +0.135                +0.160                0.026
    0.85   +0.091                +0.114                0.022
    0.90   +0.054                +0.071                0.017
    0.95   +0.023                +0.032                0.008

Reference-set spread collapses from **~0.20 (absolute) to ~0.02 (relative)** — an order of
magnitude. **A gold-referenced rule would NOT flip.**

**BUT the escape hatch is UNAVAILABLE on TOFU/Phi as shipped:** a relative rule needs a valid gold
retrain, and this benchmark provides none — `tofu_ft_retain90_phi-1.5` is a byte-identical
duplicate of the full model and every `phi_grad_ascent_*/grad_diff_*/KL_*/idk_*` repo is empty
(see M2b RUNWAY + M3 LIMITATION). So on this benchmark you cannot run the offset-robust rule
without first training your own gold. That is the honest, publishable framing: *the offset breaks
the auditing mode the benchmark actually supports, and the mode that would survive it is the one
the benchmark cannot supply.*

- Repro: `unlearn_audit/experiments/` (exp1 / exp2 / exp2b + README, result `.npz` committed
  so every table above reproduces with no GPU).

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
