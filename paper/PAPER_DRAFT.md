# The Reference Is the Bug: Difficulty-Mismatched Non-Member Sets Break Membership-Inference Auditing of Unlearning

> **DRAFT — v0.** Prose written from committed results. **All numeric values have been
> verified against the committed `.npz` arrays** (38/38 array-derivable values match, using
> the same code path the experiments used). Values that cannot be re-derived from the arrays
> remain marked ⟨verified live from HF earlier — RE-CHECK at submission time⟩ and must be
> re-confirmed before submission, since upstream repositories can drift.
> Figures are generated from the committed arrays (`paper/figures/`).
> Author list, affiliations, and acknowledgements are placeholders.

**Authors:** [you] · **Affiliation:** [placeholder] · **Contact:** [placeholder]

---

## Abstract

Membership-inference attacks (MIAs) are the standard instrument for auditing whether a model
has truly unlearned specified data, and every MIA-based audit depends on a *non-member
reference set* assumed to be exchangeable with the forget set in all respects except
membership. **That property is required but not automatic — it depends on how the reference
was constructed, and it is essentially never checked.** We give a cheap probe that checks it,
and show it discriminates. Scoring a benchmark's forget and holdout sets with a model that
was never trained on either isolates pure difficulty, since such a model has no membership
signal to exploit. On TOFU, the most widely used LLM unlearning benchmark, the designated
holdout **fails**: it is systematically *easier* than the forget set by ≈0.33 nats/token
(AUC 0.332, p < 1e-15). On MUSE-News the same probe at the same sample size **passes**
(AUC 0.522, p = 0.59, not significant), so the probe distinguishes a sound reference from an
unsound one rather than always reporting a gap. The two differ in construction, which gives a
concrete rule: **partition one corpus, do not regenerate a reference later** — MUSE-News
splits a single collection pass, whereas TOFU's holdout was regenerated separately, months
after the original release. We then show the TOFU offset has three consequences for
MIA-based auditing.
First, it renders absolute-threshold auditing **uncalibratable**: the holdout-derived null
distribution collapses to a degenerate point mass, admitting no valid decision threshold at
any target false-alarm rate. Second, on a controlled synthetic retention gradient with
unambiguous ground truth, the offset produces **missed detections** — a difficulty-matched
auditor flags residual retention across a band where a holdout-based auditor following
standard practice reports clean forgetting. Third, it corrupts raw-score MIAs (LOSS, ZLib,
Min-K, Min-K++) while leaving reference-normalized MIAs intact. The offset-robust
alternative — gold-referenced (relative) rules — is immune by common-mode cancellation, but
is *unavailable on TOFU as shipped*: the released retain checkpoint is a byte-identical
duplicate of the full finetuned model, and the released unlearned checkpoints are empty
repositories. We provide a difficulty-matched reference construction that restores
calibration, and release all experiments with GPU-free reproducibility from committed result
arrays.

---

## 1. Introduction

Machine unlearning aims to remove the influence of specific training data from a trained
model, motivated by privacy regulation and the right to erasure. A central, unresolved
difficulty is *verification*: given a model that claims to have forgotten a set of data, how
do we confirm it actually did? Membership-inference attacks are the field's default answer.
An MIA assigns each sample a membership score, and an audit declares forgetting successful
when the forget set becomes statistically indistinguishable from a *non-member reference* —
canonically, a held-out set the model never trained on.

The entire construction rests on an unstated assumption: that the non-member reference is
*exchangeable* with the forget set in everything except membership. If the reference is
intrinsically easier or harder than the forget set for reasons unrelated to training, the
audit's calibration is corrupted at the source — before any attack, any threshold, any
model. To our knowledge, no prior work checks whether this assumption holds on the benchmarks
the field actually uses.

We check it. On TOFU — a synthetic-author QA benchmark and the most cited LLM unlearning
testbed — it fails: the designated holdout set is measurably easier than the forget set for a
model that has seen neither. Because the probe model has no membership information, the gap
is *pure difficulty*, not leakage. On MUSE-News, at the same sample size, the same probe finds
nothing (§3.1) — so this is a diagnostic that discriminates, not one that fires everywhere,
and the property it tests is one benchmarks can and do get right. We then trace what the TOFU
offset does to MIA-based auditing:

- It makes absolute-threshold auditing **uncalibratable** (§4.1): the holdout-based null is a
  degenerate point mass with no threshold at any false-alarm rate.
- It causes **missed detections** (§4.2): on a controlled retention gradient, standard-practice
  holdout auditing reports "forgotten" where a difficulty-matched auditor correctly detects
  residual retention.
- It **corrupts raw-score MIAs but not reference-normalized ones** (§4.3), which lets us say
  precisely which of the field's shipped attacks inherit the bias.

We are careful about scope (§5). The failure is specific to *absolute-threshold* auditing —
the "MIA at chance ⇒ forgotten" convention that is standard practice. Gold-referenced
*relative* rules are immune, because the difficulty offset is common-mode across candidate
and reference and cancels under subtraction. But the offset-robust mode is *unavailable on
TOFU as shipped*: there is no valid gold retrain to reference, because the released retain
checkpoint duplicates the full finetuned model and the released unlearned checkpoints are
empty. The offset breaks the auditing mode the benchmark supports, and the mode that would
survive it is the one the benchmark cannot supply.

Our contributions:

- **C1.** A measured, provenance-verified difficulty offset in TOFU's holdout set, localized
  to the holdout (not the forget set) and shown not to be a length artifact (§3) — together
  with a **positive control** on MUSE-News, where the identical probe at the identical sample
  size returns chance, establishing that the check discriminates and yielding the construction
  rule *partition one corpus, do not regenerate a reference later* (§3.1).
- **C2.** A demonstration that the offset makes absolute-threshold auditing uncalibratable,
  and a difficulty-matched reference that restores calibration (§4.1, §6).
- **C3.** A controlled demonstration that the offset causes missed detections, with a
  same-margin control isolating the cause to the reference set (§4.2).
- **C4.** A map of which MIAs inherit the bias — raw-score corrupted, reference-normalized
  intact — tied to the specific attacks shipped by the standard evaluation framework (§4.3).
- **C5.** A difficulty-matched reference construction and a fully GPU-free-reproducible
  artifact release (§6).

---

## 2. Background and Related Work

**MIA-based unlearning auditing.** The standard audit treats forgetting as the erasure of a
membership signal: if an attacker cannot distinguish forget-set samples from non-members, the
data is deemed forgotten. Loss- or perplexity-thresholding is the canonical black-box attack;
numerous variants exist. All share the reliance on a non-member reference.

**Meta-evaluation of unlearning metrics.** Recent work evaluates whether unlearning *metrics*
faithfully reflect knowledge and resist stress tests. This line assesses metrics *against* a
reference and holdout, and explicitly notes that MIA-based privacy metrics assume access to
clean i.i.d. holdouts or oracle retrain models. Our finding is upstream: we show the reference
those metrics consume is itself miscalibrated on the standard benchmark, which propagates into
the shipped MIA metrics (§4.3).

**Critiques of the MIA paradigm.** A parallel line argues that *failed* MIA does not imply
true forgetting, because unlearned samples occupy a feature region distinct from genuine
non-members, and proposes training-free estimators as replacements. This critique concerns the
*estimator*; it inherits the assumption of a clean member/non-member reference. Our finding is
orthogonal and prior: the reference itself carries a difficulty offset, which corrupts
estimators built on top of it regardless of their sophistication.

**Relative (gold-referenced) rules.** Some evaluations compare a candidate model to a gold
retrain model on the same reference set. We show in §5 that these are robust to the difficulty
offset — but require a valid gold retrain that the benchmark does not provide.

*Positioning in one line each.* Against meta-evaluation frameworks: they evaluate metrics
against a reference; we show the reference is the problem. Against paradigm critiques: they fix
the estimator; we show the input to the estimator is miscalibrated. Against relative rules:
they are offset-robust but unavailable on the benchmark as shipped.

---

## 3. A Difficulty Offset in the Standard Reference (EXP-1)

**Probe design.** To isolate difficulty from membership, we score TOFU's forget and holdout
sets with a model that was never trained on *any* TOFU data (base Phi-1.5). Such a model
cannot exploit membership; any systematic score gap between forget and holdout is therefore
attributable to intrinsic difficulty alone. We use answer-only, length-normalized negative
log-likelihood (NLL) per token as the difficulty measure, and report separability as ROC-AUC
between forget and holdout NLLs.

**Provenance (pre-empting "wrong data").** The holdout we test is `holdout10` from the
canonical TOFU dataset — the same configuration the standard evaluation framework's MIA
config points to as its non-member set, added to the benchmark by its maintainers as the
"improved" holdout. It is not a deprecated split; it is the current field-standard reference.
⟨TOFU_MIA.yaml → TOFU_QA_holdout → locuslab/TOFU config holdout10; added 2025-03-27 —
verified live from HF earlier, RE-CHECK at submission time⟩

**Result.** Table 1 reports the probe. Against the never-trained model, forget10 and holdout10
separate at AUC 0.332 — far from the 0.5 that exchangeable sets would yield. The two
controls localize the defect: forget10 vs retain90, two splits from the *original* TOFU
generation run, separate at AUC 0.514 (matched to noise), while retain90 vs holdout10
separates at 0.310. The holdout is ≈0.33 nats/token easier than *both* original
splits. The defect is in the holdout, not the forget set.

**Not a length artifact.** Scoring is length-normalized, and holdout answers are on average
*longer* than forget answers (41.8 vs 36.1 mean tokens), which would bias separability in
the opposite direction. The offset persists against this bias.

> **Table 1.** Difficulty-probe AUC under a never-trained model (base Phi-1.5), answer-only
> length-normalized NLL. forget10–holdout10 0.332; forget10–retain90 0.514 (control,
> matched); retain90–holdout10 0.310.

> **Figure 1.** NLL distributions for forget10, holdout10, and retain90 under the never-trained
> model. The holdout distribution is visibly shifted toward lower NLL (easier).
> *[`paper/figures/fig1_difficulty_distributions.pdf`]*

### 3.1 Positive control: the probe passes a soundly-constructed reference (EXP-3)

A probe that reports a gap on every benchmark would be worthless — the TOFU result would say
more about our difficulty measure than about TOFU. We therefore ran the identical probe,
through the identical code path, on a second benchmark whose reference is built differently.

**MUSE-News** (`muse-bench/MUSE-News`, config `privleak`) supplies `forget`, `holdout` and
`retain` as three mutually disjoint subsets of a *single* corpus — BBC articles published
after August 2023, collected in one pass so that the target model had never seen them. Its
holdout is documented as data never seen during pre-training or unlearning. We probe it with
the same never-trained model (base Phi-1.5), which predates that collection window and is
therefore blind to all three splits; any residual overlap would affect forget and holdout
symmetrically and so could not manufacture a gap.

**The probe passes.** forget vs holdout gives **AUC 0.522 (p = 0.59, not significant)**, an
offset of +0.039 nats/token — an order of magnitude smaller than TOFU's and in the opposite
direction. Controls agree: forget vs retain 0.499 (p = 0.98), retain vs holdout 0.522
(p = 0.59). Every MUSE-News comparison is statistically indistinguishable from chance.

**This is not a power artifact, and the matched-sample-size comparison is what shows it.**
MUSE-News supplies 100 texts per split. Restricting TOFU to the *same* n = 100 still yields
**AUC 0.294 (p = 4.8e-07)** — far from chance. Same probe, same code, same sample size: one
benchmark fails decisively, the other passes cleanly. The diagnostic discriminates.

**What separates them is construction, not subject matter.** MUSE-News partitions one corpus
collected in a single pass; TOFU's holdout was regenerated separately, roughly fourteen months
after the original release, and did not reproduce the original generation's difficulty
distribution. Hence the rule we propose: *partition one corpus; do not regenerate a reference
later.* Where regeneration is unavoidable, this probe is a cheap check — one forward pass per
example with an off-the-shelf never-trained model, no training and no target-model access.

> **Table 4.** Probe on both benchmarks, identical method. TOFU forget10–holdout10 AUC 0.332
> (n = 400, p < 1e-15) and 0.294 at n = 100 (p = 4.8e-07); MUSE-News forget–holdout AUC 0.522
> (n = 100, p = 0.59, n.s.), controls 0.499 and 0.522.

*Excluded: MUSE-Books.* Its text is Harry Potter, which essentially every general-purpose base
LM has seen in pretraining, so no never-trained probe exists for it and any gap is
uninterpretable — memorization cannot be separated from difficulty. Its numbers confirm the
diagnosis rather than the finding: forget–holdout 0.302, but retain–holdout 0.153 and
forget–retain 0.733, i.e. the three splits disagree with one another in a way no membership
account explains. We report it for completeness and rest nothing on it.

---

## 4. Consequences for Auditing (EXP-2, EXP-2b)

To study how the offset affects graded auditing, we need models with *known, varying* amounts
of residual retention. Because the benchmark's released unlearned checkpoints are unavailable
(§5), we construct a controlled synthetic retention gradient by interpolating in weight space
between the finetuned baseline and the never-trained model:
θ(α) = (1−α)·θ_baseline + α·θ_base, for α ∈ [0, 1]. This yields a monotone ladder of
partial memorization with unambiguous ground truth: every α < 1 contains baseline weight, so
residual retention is present by construction, and a correct auditor must detect it.

> **This gradient is a synthetic probe, not an unlearning method.** It exists solely to supply
> graded retention with known ground truth. It must not be compared to published unlearning
> numbers. Real-method validation is deferred (§7). Endpoint sanity was enforced: α=0 reproduces
> the baseline member NLL and α=1 reproduces the never-trained reference exactly (0.084
> vs logged 0.090; 2.072 with max per-example drift 0.0000; ⟨341/341 state-dict tensors aligned
> — runtime assertion in exp2, not stored in the arrays; RE-CHECK on rerun⟩).

### 4.1 The offset makes absolute-threshold auditing uncalibratable (C2)

An absolute-threshold auditor calibrates a decision threshold by bootstrapping a null
distribution of the retention score under truly-forgotten conditions, then flags retention
above the (1−FAR) quantile. Using holdout10 as the non-member reference, this null collapses
to a **degenerate point mass** (standard deviation 0.0000), because the difficulty offset
drives the truly-forgotten score identically to a floor. No valid threshold exists at any
target false-alarm rate.

Replacing holdout10 with a difficulty-matched reference (§6) restores a well-defined null
(standard deviation 0.0358), a usable threshold (0.0911), and the target false-alarm rate
(0.050). Difficulty matching is what makes the auditor calibratable at all.

> **Table 3.** Calibration under holdout10 vs difficulty-matched reference: null sd
> 0.0000 vs 0.0358, threshold 0 vs 0.0911, FAR — vs 0.050.

### 4.2 The offset causes missed detections (C3)

Figure 2 plots residual-retention score against α under both references, with decision
thresholds. The flip is **directly observed at the sampled rungs α ∈ {0.80, 0.85, 0.90}**;
taking linear crossings between rungs, the **missed-detection band spans α ≈ 0.77–0.94** (the
holdout-based auditor stops detecting at α ≈ 0.77, the difficulty-matched auditor at
α ≈ 0.94). Across this band, the difficulty-matched auditor reports RETENTION DETECTED while a
holdout-based auditor following the standard "AUC ≈ 0.5 ⇒ forgotten" convention reports clean
forgetting. Ground truth is unambiguous (retention is present for all α < 1), so these are
genuine missed detections induced by the reference, not by chance.

**Same-margin control (isolating the cause).** One might object that the matched auditor is
calibrated while the holdout auditor is not, confounding reference choice with calibration. We
control for this by transplanting the matched threshold onto holdout-referenced scores: same
decision margin, only the reference set differs. The flip persists (4/5 rungs in the fine
sweep), isolating the cause to the **reference set**, not the calibration. The offset costs
approximately one-fifth of the detectable retention range (holdout boundary α≈0.77 vs
matched boundary α≈0.94).

> **Figure 2.** Residual-retention score vs α, under holdout10 (field default) and the
> difficulty-matched reference, with decision thresholds and the missed-detection band shaded.
> *[`paper/figures/fig2_missed_detection.pdf`]*

### 4.3 Which MIAs inherit the bias (C4)

The offset corrupts *raw-score* MIAs — those that threshold an absolute membership score
(LOSS, ZLib, Min-K, Min-K++) — because the reference's artificial easiness suppresses apparent
separability. *Reference-normalized* MIAs, which subtract a per-example reference-model score,
are unaffected, because the offset is common-mode and cancels. At α=0.7, the residual-retention
readout is 0.162 for a raw LOSS attack against holdout10, 0.541 for the same attack against
the matched reference, and 0.970 for the reference-normalized attack — a 3.3× under-report
by the field-default raw-score/holdout combination.

We note a constraint discovered in constructing the reference-normalized comparison: computing
it against the *same* model used as the difficulty null yields an identically-zero score by
construction (a self-comparison), so calibrating a reference-normalized MIA requires a null
model *distinct* from the reference model. This is a design constraint for any future
reference-normalized auditor built on this benchmark.

> **Table 2.** MIA-family map: raw-score attacks (LOSS/ZLib/Min-K/Min-K++) inherit the offset;
> reference-normalized attacks do not. α=0.7 receipt: 0.162 / 0.541 / 0.970.

**Ordering vs levels.** Across the ladder, residual-retention scores are monotone in α under
*both* references: the offset biases score *levels*, not *ordering*. Leaderboard-style relative
ranking of methods is therefore comparatively robust; threshold-based forgotten/not decisions
are not.

---

## 5. Scope and the Missing Escape Hatch

**The failure is specific to absolute-threshold auditing.** Our results concern the standard
"MIA at chance ⇒ forgotten" convention, which thresholds an absolute membership score against a
non-member reference. This is what standard practice — and our auditor — does.

**Relative rules are immune.** Gold-referenced rules (e.g., comparing a candidate to a retrain
model on the same non-member set) are robust to the difficulty offset: the offset depresses the
candidate's and the gold's scores by the same amount, so subtraction cancels it. Empirically,
the reference-set-induced spread collapses from ≈0.20 under absolute rules to ≈0.02 under a
relative rule — an order of magnitude.

**But the escape hatch is unavailable on TOFU as shipped.** A relative rule requires a valid
gold retrain to reference. TOFU/Phi provides none: the released retain checkpoint is a
byte-identical duplicate of the full finetuned model ⟨identical safetensors weights, differing
only in fp32 vs bf16 encoding — verified live from HF earlier, RE-CHECK at submission time⟩,
and the released unlearned checkpoints (`phi_grad_ascent_*`, `phi_grad_diff_*`, `phi_KL_*`,
`phi_idk_*`) are empty repositories ⟨verified live from HF earlier, RE-CHECK at submission
time⟩. There is no gold retrain to reference without training one. The offset breaks the
auditing mode the benchmark supports, and the mode that would survive it is the one the
benchmark cannot supply.

---

## 6. A Difficulty-Matched Reference (C5)

We restore calibration by constructing a difficulty-matched non-member reference. For each
forget-set example, we select a distinct holdout example whose *never-trained-model* NLL is
within a caliper of the forget example's, using only the null model's scores — never the target
model's — so no membership signal leaks into the construction. With a caliper of 0.05, this
retains 287 of 400 pairs and drives the never-trained separability from AUC 0.332 to
0.509 — matched to noise, the same neighborhood as the original-split control (0.514).

**Honest cost.** Matching discards the forget set's hard tail, so the matched auditor evaluates
a *difficulty-matched subpopulation* of the forget set (n=287/400), not the whole set, with a
residual imbalance of +0.015 nats/token that we report alongside every matched result. This
qualifier travels with every matched number in the paper.

---

## 7. Limitations

**Synthetic retention gradient.** The missed-detection demonstration (§4.2) uses weight-space
interpolation, not real unlearning methods. The *reference mismatch itself* (§3) is measured on
real data with a real, never-trained probe; the gradient only supplies a graded axis with known
ground truth to show the mismatch has consequences under controlled conditions. Validating the
missed-detection result on real unlearning methods (grad-ascent, NPO, etc.) requires training
those methods and a gold retrain — deferred to future work, and the natural extension of this
paper.

**The offset is TOFU-specific; what generalizes is the check, not the defect.** We tested a
second benchmark and it passed (§3.1), so we do not claim reference mismatch is widespread. We
claim the weaker and better-evidenced thing: a difficulty-matched reference is required but
not automatic, it depends on construction, and it is cheap to verify. Two benchmarks is still
two — whether WMDP or other references pass is untested, and one clean benchmark is not
evidence that most are clean.

**One probe-model family.** Both benchmarks were probed with base Phi-1.5. A difficulty
measure is model-dependent in principle, so a second probe family (Pythia, GPT-2-scale, a
larger base model) would strengthen the claim that these are properties of the *data* rather
than of one tokenizer and one pretraining mix. The controls make this unlikely to be an
artifact — under the same probe, TOFU's two original-generation splits are matched (0.514)
while its regenerated holdout is not (0.332) — but it remains untested.

**Prose versus QA.** The two benchmarks also differ in format: TOFU is short QA scored over
the answer span, MUSE-News is long-form prose scored over a fixed 512-token prefix. That the
probe behaves sensibly on both is reassuring, but format and construction are not fully
disentangled by two data points.

**Matched subpopulation.** The restored-calibration results audit a difficulty-matched
subpopulation of the forget set, not its entirety.

**Difficulty proxy.** The never-trained model provides a difficulty proxy and a null substitute,
not a true gold retrain; a trained gold retrain would sharpen both the matching and the
relative-rule analysis.

---

## 8. Conclusion

Membership-inference auditing of unlearning is only as trustworthy as its non-member reference,
and that reference is almost never checked. On the field's most-used LLM unlearning benchmark,
the standard reference carries a measurable difficulty offset that renders absolute-threshold
auditing uncalibratable and induces missed detections, corrupting the raw-score MIAs the field
ships. The offset-robust alternative is unavailable on the benchmark as delivered. Auditing
research should validate the *reference*, not only the estimator: an audit calibrated against a
warped yardstick reports a warped result, however sophisticated the attack.

---

## Reproducibility

All experiments and result arrays are released. Every table and figure re-derives from the
committed arrays without a GPU (verified): `exp1_probe_nll.npz`, `exp2_ladder.npz`,
`exp2b_fine.npz`, `exp3_muse_news_nll.npz`, `exp3_muse_books_nll.npz`, with rerun
instructions in the experiments README.

---

## VERIFY-BEFORE-SUBMIT checklist — status

**Verified against the committed arrays (38/38 match, same code path as the experiments):**

- [x] EXP-1: forget–holdout AUC 0.332; forget–retain 0.514; retain–holdout 0.310
- [x] EXP-1: ≈0.33 nats/token offset (exact: −0.324 vs forget, −0.357 vs retain);
      answer-length 41.8 vs 36.1 (and retain 33.1) — token lengths recomputed from
      tokenizer + dataset, not stored in the arrays
- [x] Ladder endpoints: α=0 → 0.084 (vs logged 0.090); α=1 → 2.072; drift 0.0000
- [x] §4.1 nulls: holdout sd 0.0000 / thr 0; matched sd 0.0358 / thr 0.0911 / FAR 0.050
- [x] §4.2 flip observed at rungs {0.80, 0.85, 0.90}; interpolated band α ≈ 0.77–0.94;
      fires 3/5 (field rule); same-margin control 4/5; boundaries 0.77 vs 0.94;
      range cost 0.185 (≈ one-fifth)
- [x] §4.3 α=0.7 receipt 0.162 / 0.541 / 0.970; 3.34× under-report; monotone under both refs
- [x] §5 spread 0.196 → 0.019 (≈0.20 → ≈0.02)
- [x] §6 caliper 0.05; 287/400 pairs; matched AUC 0.509; residual imbalance +0.015
- [x] §3.1 EXP-3 positive control: MUSE-News forget–holdout AUC 0.522, p 0.59 (n.s.), gap
      +0.039; controls 0.499 (p 0.98) and 0.522 (p 0.59); TOFU at matched n=100 AUC 0.294,
      p 4.8e-07; MUSE-Books excluded as contamination-confounded (0.302 / 0.153 / 0.733)

**NOT re-derivable from the arrays — RE-CHECK at submission time (upstream can drift):**

- [ ] Provenance: TOFU_MIA.yaml holdout config; 2025-03-27 add date
- [ ] §5 retain checkpoint byte-identical duplicate of the full finetuned model
- [ ] §5 unlearned repos (`phi_grad_ascent_*` / `grad_diff_*` / `KL_*` / `idk_*`) empty
- [ ] §4 ladder "341/341 state-dict tensors aligned" — runtime assertion, re-check on rerun

**Editorial:**

- [x] Interpolation-ladder "synthetic, not unlearning" caveat appears in §4 box, §4 intro, §7
- [x] Scope (absolute vs relative, escape hatch unavailable) appears in §1 and §5
