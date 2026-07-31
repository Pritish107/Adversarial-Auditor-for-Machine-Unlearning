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

We check it. On TOFU [1] — a synthetic-author QA benchmark and the most cited LLM unlearning
testbed — it fails: the designated holdout set is measurably easier than the forget set for a
model that has seen neither. Because the probe model has no membership information, the gap
is *pure difficulty*, not leakage. On MUSE-News [2], at the same sample size, the same probe finds
nothing (§5.1) — so this is a diagnostic that discriminates, not one that fires everywhere,
and the property it tests is one benchmarks can and do get right. We then trace what the TOFU
offset does to MIA-based auditing:

- It makes absolute-threshold auditing **uncalibratable** (§6.1): the holdout-based null is a
  degenerate point mass with no threshold at any false-alarm rate.
- It causes **missed detections** (§6.2): on a controlled retention gradient, standard-practice
  holdout auditing reports "forgotten" where a difficulty-matched auditor correctly detects
  residual retention.
- It **corrupts raw-score MIAs but not reference-normalized ones** (§6.3), which lets us say
  precisely which of the field's shipped attacks inherit the bias.

We are careful about scope (§7). The failure is specific to *absolute-threshold* auditing —
the "MIA at chance ⇒ forgotten" convention that is standard practice. Gold-referenced
*relative* rules are immune, because the difficulty offset is common-mode across candidate
and reference and cancels under subtraction. But the offset-robust mode is *unavailable on
TOFU as shipped*: there is no valid gold retrain to reference, because the released retain
checkpoint duplicates the full finetuned model and the released unlearned checkpoints are
empty. The offset breaks the auditing mode the benchmark supports, and the mode that would
survive it is the one the benchmark cannot supply.

Our contributions:

- **C1.** A measured, provenance-verified difficulty offset in TOFU's holdout set, localized
  to the holdout (not the forget set) and shown not to be a length artifact (§5) — together
  with a **positive control** on MUSE-News, where the identical probe at the identical sample
  size returns chance, establishing that the check discriminates and yielding the construction
  rule *partition one corpus, do not regenerate a reference later* (§5.1).
- **C2.** A demonstration that the offset makes absolute-threshold auditing uncalibratable,
  and a difficulty-matched reference that restores calibration (§6.1, §8).
- **C3.** A controlled demonstration that the offset causes missed detections, with a
  same-margin control isolating the cause to the reference set (§6.2).
- **C4.** A map of which MIAs inherit the bias — raw-score corrupted, reference-normalized
  intact — tied to the specific attacks shipped by the standard evaluation framework (§6.3).
- **C5.** A difficulty-matched reference construction and a fully GPU-free-reproducible
  artifact release (§8).

---

## 2. Background and Preliminaries

This section fixes notation and states the audit procedure we analyse. Nothing here is novel;
it is written out because the failure we identify lives in an assumption that is usually left
implicit.

### 2.1 Unlearning

Let $\mathcal{D}$ be a training corpus and let a model $\theta_{\text{full}}$ be obtained by
training on $\mathcal{D}$. A *forget request* designates a subset $F \subset \mathcal{D}$ to be
removed; the complement $R = \mathcal{D} \setminus F$ is the *retain set*. An unlearning
algorithm is a map

$$\mathcal{U} : (\theta_{\text{full}}, F, R) \mapsto \theta_{\text{un}}$$

whose output should behave as though $F$ had never been in the training corpus. The reference
point for "as though" is the *gold retrain* $\theta_{\text{retrain}}$, trained from scratch on
$R$ alone. Exact retraining is the definition of success but is usually too expensive to run,
which is precisely why approximate unlearning methods — and the auditing problem — exist.

Two further ingredients matter for auditing. A *holdout* set $H$ consists of samples drawn
from the same distribution as $\mathcal{D}$ but never included in training, and therefore
never seen by $\theta_{\text{full}}$, $\theta_{\text{un}}$, or $\theta_{\text{retrain}}$. The
triple $(F, R, H)$ — forget, retain, holdout — is the standard evaluation scaffolding shipped
by unlearning benchmarks [1, 2].

### 2.2 Membership inference

A membership-inference attack equips a model with a real-valued *membership score*
$s(x; \theta)$ intended to be systematically lower (more member-like) on training members than
on non-members. The canonical black-box instantiation thresholds the model's loss on the
sample [3]; variants normalise by a compression estimate of the sample's intrinsic
complexity [4], aggregate only the least-likely tokens [5, 6], or calibrate against a
reference model's score on the same sample [7].

Rather than fix a threshold, an audit measures *separability*. Given member scores
$\{s(x)\}_{x \in F}$ and non-member scores $\{s(x)\}_{x \in H}$, the attack's power is the
area under the ROC curve of $-s$ against the membership label,

$$\mathrm{AUC} = \Pr\big[\, s(x_F) < s(x_H) \,\big] + \tfrac{1}{2}\Pr\big[\, s(x_F) = s(x_H) \,\big],
\qquad x_F \sim F,\; x_H \sim H .$$

$\mathrm{AUC} = 0.5$ means the two sets are indistinguishable to the attack.

### 2.3 The audit and the retention score

An audit converts separability into a decision. We use the retention score

$$\rho \;=\; \mathrm{clip}\big(2(\mathrm{AUC} - 0.5),\, 0,\, 1\big) \in [0, 1],$$

so $\rho = 0$ when the forget set is indistinguishable from the reference and $\rho = 1$ under
perfect separability. The clipping encodes an asymmetry: $\mathrm{AUC} < 0.5$ means members
look *less* member-like than non-members, which is not evidence of retention, so it floors at
zero rather than being read as a negative result.

Because $\rho$ is a statistic of finite samples, a raw value cannot be read as a verdict. The
audit therefore calibrates: it constructs a *null distribution* $\mathcal{N}$ of $\rho$ under
the truly-forgotten condition, sets the decision threshold $\tau$ at the $(1 - \text{FAR})$
quantile of $\mathcal{N}$ for a target false-alarm rate, and reports

$$\texttt{RETENTION DETECTED} \iff \rho > \tau .$$

The comparison is strict: a candidate is flagged only when it *exceeds* the forgotten
baseline, not when it merely equals it. In our setting $\mathcal{N}$ is obtained by
bootstrap-resampling the member and non-member scores of a model known to be free of the
forget set, and the realised false-alarm rate is reported as $\mathrm{FAR} =
\Pr_{\mathcal{N}}[\rho > \tau]$ rather than assumed.

### 2.4 The exchangeability assumption

Everything above depends on a premise that is rarely stated. Write the score of a sample as a
difficulty term plus a membership term,

$$s(x; \theta) \;=\; d(x) \;+\; m(x; \theta),$$

where $d$ captures how hard $x$ is intrinsically — for a language model, how predictable its
tokens are to *any* model of that class — and $m$ captures the reduction attributable to
having trained on $x$. An MIA audit attributes the entire $F$-versus-$H$ score gap to $m$. That
attribution is valid only if the difficulty terms are matched in distribution:

$$d(x_F) \;\stackrel{d}{=}\; d(x_H), \qquad x_F \sim F,\; x_H \sim H. \tag{$\star$}$$

We call $(\star)$ the **exchangeability assumption**. If it fails — if the reference is
intrinsically easier or harder than the forget set — then the measured AUC confounds
difficulty with membership, and every quantity built on top of it (the retention score, the
null, the threshold, the verdict) inherits the confound. No amount of sophistication in the
*estimator* repairs a violation of $(\star)$, because the violation is in the estimator's
input.

The contribution of this paper is to observe that $(\star)$ is directly testable at negligible
cost, to test it on the benchmarks the field actually uses, and to trace the consequences
where it fails.

---

## 3. Related Work

**Benchmarks: TOFU.** TOFU [1] is the most-cited testbed for LLM unlearning. It consists of
synthetic question–answer pairs about 200 fictitious authors, generated so that the content
cannot appear in any pretraining corpus — an elegant design that isolates the effect of
finetuning. The benchmark ships forget/retain partitions at several fractions
(`forget01/05/10` with matched `retain99/95/90`) together with finetuned target models. Its
headline finding is that no baseline unlearning method it evaluated achieved effective
forgetting. Our work does not dispute any TOFU result; it examines a component added to the
benchmark later — the `holdout` splits that serve as the non-member reference for MIA-based
evaluation — and shows that this component violates $(\star)$.

**Benchmarks: MUSE.** MUSE [2] evaluates unlearning along six axes, including verbatim
memorisation, knowledge memorisation, privacy leakage, utility preservation, scalability, and
sustainability. Its two corpora are news articles and books. Relevant here is its
construction discipline: for the news corpus it collects a single pool of BBC articles
published after August 2023 — chosen so the target model's pretraining could not have seen
them — and partitions that one pool into disjoint forget, retain, and holdout subsets. We use
MUSE-News as a positive control and find its reference satisfies $(\star)$; §5.1 argues the
difference from TOFU is exactly this construction discipline.

**Meta-evaluation frameworks.** A recent line of work standardises unlearning evaluation,
asking whether the *metrics themselves* are faithful and robust, and consolidates many
algorithms and metrics behind one interface. This work is closest in spirit to ours and
supplies the concrete MIA implementations we analyse in §6.3; it also states that MIA-based
privacy metrics presuppose clean i.i.d. holdouts or oracle retrain models. Our finding sits
one level upstream: the presupposed clean holdout is itself miscalibrated, so the
miscalibration propagates into the shipped metrics rather than being screened off by them.

**Critiques of the MIA paradigm.** A parallel line argues that a *failed* MIA does not license
the conclusion that forgetting occurred, since unlearned samples occupy a feature-space region
distinct from genuine non-members, and proposes training-free estimators of the non-member
mixture proportion instead. That critique targets the *estimator*: it accepts a clean
reference and argues the inference drawn from it is unsound. Ours is orthogonal and prior in
the pipeline — the reference itself carries a difficulty offset, which corrupts any estimator
built on it. The two compose rather than compete: a difficulty-matched reference is a
precondition for the estimator-level fix to mean anything.

**Membership-inference attacks.** The loss-threshold attack [3] established the overfitting–
leakage link and remains the canonical black-box baseline. Later work sharpened the statistic,
not the reference: zlib normalisation divides loss by a compression estimate, targeting cheap
generic text [4]; Min-K% scores only the least-likely tokens, on the premise that memorised
text lacks low-probability outliers [5]; Min-K%++ adds a theoretically motivated next-token
normalisation [6]. All four are *raw-score* attacks — they threshold an absolute score against
a non-member reference, and so inherit any violation of $(\star)$ directly. The
reference-model line instead calibrates per-example against a second model's score on the
*same* sample [7], which is what makes it structurally immune to a common-mode offset (§6.3).

**Relative (gold-referenced) rules.** Rather than comparing a candidate to an absolute chance
level, some evaluations compare it to a gold retrain model measured on the same reference set
[1, 2]. We show in §7 that these rules are robust to a difficulty offset by common-mode
cancellation — and that this escape hatch is nonetheless unavailable on TOFU as shipped,
because no valid gold retrain is actually distributed.

*Positioning in one line each.* Against meta-evaluation frameworks: they evaluate metrics
against a reference; we show the reference is the problem. Against paradigm critiques: they fix
the estimator; we show the input to the estimator is miscalibrated. Against relative rules:
they are offset-robust but unavailable on the benchmark as shipped.

---

## 4. Experimental Setup

Every experiment in this paper is a scoring pass: no model is trained, and no gradient is
taken. This section specifies the data, models, and procedures precisely enough to reproduce
each number.

### 4.1 Datasets and splits

**TOFU** is loaded from the canonical dataset repository, whose configurations are single
`train` splits of `{question, answer}` string pairs. We use `forget10` (400 pairs, the 10%
forget partition), `retain90` (3600 pairs, its complement), and `holdout10` (400 pairs, the
designated non-member reference). We verified that `forget10` and `retain90` are disjoint and
that their union is exactly the full 4000-pair corpus the target model was finetuned on, and
that `holdout10` is disjoint from that corpus. Where a control requires a size-matched retain
sample we take the first 400 pairs of `retain90`.

**MUSE-News** is loaded from its benchmark repository under the `privleak` configuration,
which supplies `forget`, `holdout`, and `retain` as three splits of 100 raw-text passages
each. The three are near-identical in length (8053 / 8051 / 8070 mean characters), which is a
consequence of their being partitions of a single collection pass rather than independently
generated sets.

**Provenance.** The holdout we test is the configuration that the standard evaluation
framework's MIA configuration points to as its non-member set — not a deprecated split, but
the current field-standard reference, added to the benchmark by its maintainers as the
improved holdout.
⟨TOFU_MIA.yaml → TOFU_QA_holdout → locuslab/TOFU config holdout10; added 2025-03-27 —
verified live from HF earlier, RE-CHECK at submission time⟩

### 4.2 Models

The **target** model is the publicly released TOFU-finetuned Phi-1.5 checkpoint: a 1.3B-parameter
decoder-only transformer [10] finetuned on the full 4000-pair TOFU corpus, so all of
`forget10` and `retain90` are training members for it. It is distributed in bfloat16 across
341 parameter tensors.

The **probe / null** model is the corresponding *base* Phi-1.5 [10], never finetuned on TOFU
and never trained on MUSE-News. Its role is central: having seen neither the forget set nor
the reference set, its membership term $m$ is identically zero on both, so any score gap it
exhibits is attributable to the difficulty term $d$ alone. It is a measuring instrument for
$(\star)$, not a model under audit. For MUSE-News the same base model serves as probe; it
predates the post-August-2023 collection window, so it is blind to all three splits, and any
residual overlap would fall on forget and holdout symmetrically and so could not manufacture a
gap.

A practical note: the finetuned repositories ship no tokenizer files, and loading one from
them silently yields a broken vocabulary rather than raising — the tokenizer must be loaded
from the base model.

### 4.3 The difficulty score: answer-only, length-normalised NLL

All scores are per-token negative log-likelihood over the *answer span only*, under a prompt
template identical for members and non-members. For a pair $(q, a)$:

1. Form the prompt $P(q)$ from a fixed template and tokenize it; let $\ell$ be its token count.
2. Tokenize $P(q) \Vert a$ to obtain the full sequence $x_1, \dots, x_T$.
3. Run one forward pass and read the next-token log-probabilities
   $\log p_\theta(x_t \mid x_{<t})$.
4. Report
   $$s(q,a;\theta) \;=\; -\frac{1}{T-\ell}\sum_{t=\ell+1}^{T} \log p_\theta(x_t \mid x_{<t}).$$

Two properties of this definition matter. It is **answer-only**: the question tokens are
context and are not scored, so a question that is easy or hard to predict cannot move the
score. It is **length-normalised**: dividing by the answer's token count means a long answer
is not automatically scored as more surprising than a short one. Both choices remove obvious
confounds that would otherwise be mistaken for the effect we are measuring.

MUSE-News is prose rather than question–answer pairs, so the analogous span is the whole
passage. We obtain this through the identical code path by setting the prompt to the empty
string, which reduces the definition above to a full-text length-normalised NLL, and we
truncate every passage to a uniform 512-token prefix — the splits are already length-matched,
and the probe model's context is 2048 tokens.

Separability is computed with the same ROC-AUC routine in every experiment, so numbers from
different sections are directly comparable by construction.

### 4.4 The difficulty-matched reference

Given base-model scores on the forget and holdout sets, we construct a matched reference by
1:1 nearest-neighbour matching without replacement under a caliper:

1. Sort the forget examples by base-model NLL (deterministic order).
2. For each forget example in turn, find the unmatched holdout example whose base-model NLL is
   closest.
3. Accept the pair if the absolute difference is within the caliper $c = 0.05$ nats/token;
   otherwise leave the forget example unmatched.

Selection uses **only the probe model's scores** — never the target model's — so no membership
information can leak into the construction of the reference. The result is reported with its
cost: matching discards forget examples whose difficulty has no counterpart in the holdout, so
the matched audit evaluates a difficulty-matched *subpopulation* of the forget set, and we
report the residual imbalance alongside every matched number.

### 4.5 The synthetic retention gradient

> **This gradient is a synthetic probe, not an unlearning method.** It exists solely to supply
> graded retention with known ground truth. It must not be compared to published unlearning
> numbers. Real-method validation is deferred (§10).

To study graded auditing we require models with known, varying amounts of residual retention.
The benchmark's released unlearned checkpoints are unavailable (§7), so we construct a
controlled gradient by linear interpolation in weight space between the finetuned target and
the base model,

$$\theta(\alpha) \;=\; (1-\alpha)\,\theta_{\text{baseline}} \;+\; \alpha\,\theta_{\text{base}},
\qquad \alpha \in [0,1],$$

evaluated at $\alpha \in \{0, 0.15, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1\}$. Ground
truth is unambiguous by construction: every $\alpha < 1$ retains a non-zero share of the
finetuned weights, so residual retention is present and a correct auditor must detect it.

The construction is gated rather than trusted. We assert that the two checkpoints share all
341 parameter tensors by name and shape before blending, and we require the endpoints to
reproduce the models they interpolate: $\alpha = 0$ must recover the finetuned member NLL and
$\alpha = 1$ must recover the base model's. Both hold — 0.084 against a logged 0.090, and
2.072 with maximum per-example drift 0.0000 against the independently computed base-model
scores (341/341 state-dict tensors aligned). Had either endpoint missed, the middle of the
ladder would not have been read at all.

### 4.6 Hardware and reproducibility

All scoring runs on a single consumer GPU (8 GB), in half precision, with batch size one; the
full set of experiments is a few tens of minutes of compute. Every per-example score is
committed as a `.npz` array alongside the code, and every table and figure in this paper is
regenerated from those arrays with no GPU and no model download. The figure-generation and
verification scripts are part of the released artifact.

---

## 5. A Difficulty Offset in the Standard Reference (EXP-1)

**Probe design.** To isolate difficulty from membership, we score TOFU's forget and holdout
sets with a model that was never trained on *any* TOFU data (base Phi-1.5 [10]). Such a model
cannot exploit membership; any systematic score gap between forget and holdout is therefore
attributable to intrinsic difficulty alone. We use answer-only, length-normalized negative
log-likelihood (NLL) per token as the difficulty measure, and report separability as ROC-AUC
between forget and holdout NLLs.

**Result.** Table 1 reports the probe. Against the never-trained model, forget10 and holdout10
separate at AUC 0.332 — far from the 0.5 that exchangeable sets would yield. The two
controls localize the defect: forget10 vs retain90, two splits from the *original* TOFU
generation run, separate at AUC 0.514 (matched to noise), while retain90 vs holdout10
separates at 0.310. The holdout is ≈0.33 nats/token easier than *both* original
splits. The defect is in the holdout, not the forget set.

**Not a length artifact.** Scoring is length-normalized, and holdout answers are on average
*longer* than forget answers (41.8 vs 36.1 mean tokens), which would bias separability in
the opposite direction. The offset persists against this bias.

| comparison | $n$ | member NLL | non-member NLL | gap | AUC |
|:---|---:|---:|---:|---:|---:|
| forget10 vs holdout10 | 400 / 400 | 2.072 | 1.748 | −0.324 | **0.332** |
| *control:* forget10 vs retain90 | 400 / 400 | 2.072 | 2.106 | +0.034 | 0.514 |
| *control:* retain90 vs holdout10 | 400 / 400 | 2.106 | 1.748 | −0.357 | 0.310 |

: Difficulty probe on TOFU under a never-trained model (base Phi-1.5), answer-only
length-normalised NLL per token. A reference satisfying $(\star)$ would give AUC 0.5.

![NLL distributions for forget10, holdout10, and retain90 under the never-trained model. The holdout distribution is visibly shifted toward lower NLL (easier); forget10 and retain90 nearly coincide, which is the control result.](figures/fig1_difficulty_distributions.pdf){width=58%}

### 5.1 Positive control: the probe passes a soundly-constructed reference (EXP-3)

A probe that reports a gap on every benchmark would be worthless — the TOFU result would say
more about our difficulty measure than about TOFU. We therefore ran the identical probe,
through the identical code path, on a second benchmark whose reference is built differently.

**MUSE-News** [2] (`muse-bench/MUSE-News`, config `privleak`) supplies `forget`, `holdout` and
`retain` as three mutually disjoint subsets of a *single* corpus — BBC articles published
after August 2023, collected in one pass so that the target model [11] had never seen them. Its
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

| benchmark | comparison | $n$ | AUC | $p$ | reference verdict |
|:---|:---|---:|---:|---:|:---|
| TOFU | forget10 vs holdout10 | 400 / 400 | **0.332** | < 1e-15 | violates $(\star)$ |
| TOFU | forget10 vs holdout10 | 100 / 100 | **0.294** | 4.8e-07 | violates $(\star)$ |
| TOFU | *control:* forget10 vs retain90 | 400 / 400 | 0.514 | n.s. | consistent with $(\star)$ |
| MUSE-News | forget vs holdout | 100 / 100 | **0.522** | 0.59 | consistent with $(\star)$ |
| MUSE-News | *control:* forget vs retain | 100 / 100 | 0.499 | 0.98 | consistent with $(\star)$ |
| MUSE-News | *control:* retain vs holdout | 100 / 100 | 0.522 | 0.59 | consistent with $(\star)$ |

: The same probe on both benchmarks. The matched-sample-size row is the one that
rules out low power as an explanation for the MUSE-News null.

![The probe discriminates. Identical instrument, identical code path: TOFU's reference fails at both sample sizes while every MUSE-News comparison sits at chance. The n = 100 TOFU row rules out low power as the explanation for the MUSE-News null.](figures/fig4_benchmark_comparison.pdf){width=78%}

*Excluded: MUSE-Books.* **Its forget–holdout AUC of 0.302 superficially resembles TOFU's 0.332
and must not be read as corroboration** — the controls show its splits are incoherent:
retain–holdout is 0.153 and forget–retain is 0.733, i.e. the three splits disagree with one
another in a way no membership account explains. The cause is contamination: its text is Harry
Potter, which essentially every general-purpose base LM has seen in pretraining, so no
never-trained probe exists for it and any gap is uninterpretable — memorization cannot be
separated from difficulty. Its numbers confirm that diagnosis, not our finding. We report it
for completeness and rest nothing on it.

---

## 6. Consequences for Auditing (EXP-2, EXP-2b)

We now trace what the offset does to a graded audit, using the synthetic retention gradient of
§4.5. Table 3 gives the full ladder under all three scorings; the subsections take the three
consequences in turn.

| $\alpha$ | naive acc. | $\rho$ vs holdout10 | $\rho$ vs matched | $\rho$ reference-norm. |
|---:|---:|---:|---:|---:|
| 0.00 | 0.989 | 0.997 | 0.996 | 0.999 |
| 0.15 | 0.964 | 0.993 | 0.991 | 0.997 |
| 0.30 | 0.875 | 0.973 | 0.981 | 0.994 |
| 0.50 | 0.731 | 0.733 | 0.878 | 0.985 |
| 0.70 | 0.640 | **0.162** | **0.541** | **0.970** |
| 0.75 | 0.622 | 0.040 | 0.439 | — |
| 0.80 | 0.608 | 0.000 | 0.339 | — |
| 0.85 | 0.594 | 0.000 | 0.245 | — |
| 0.90 | 0.582 | 0.000 | 0.160 | — |
| 0.95 | 0.573 | 0.000 | 0.081 | — |
| 1.00 | 0.563 | 0.000 | 0.019 | 0.000 |

: The full retention ladder. *Naive acc.* is teacher-forced answer accuracy on the
forget set — the quantity a naive evaluation would read. Retention is reported under the
field-default holdout reference, under the difficulty-matched reference, and under a
reference-normalised (per-example) scoring. The ladder is a **synthetic probe, not an
unlearning method**.

### 6.1 The offset makes absolute-threshold auditing uncalibratable (C2)

An absolute-threshold auditor calibrates a decision threshold by bootstrapping a null
distribution of the retention score under truly-forgotten conditions, then flags retention
above the (1−FAR) quantile. Using holdout10 as the non-member reference, this null collapses
to a **degenerate point mass** (standard deviation 0.0000), because the difficulty offset
drives the truly-forgotten score identically to a floor. No valid threshold exists at any
target false-alarm rate.

Replacing holdout10 with a difficulty-matched reference (§8) restores a well-defined null
(standard deviation 0.0358), a usable threshold (0.0911), and the target false-alarm rate
(0.050). Difficulty matching is what makes the auditor calibratable at all.

| reference | null s.d. | threshold $\tau$ | realised FAR | calibratable? |
|:---|---:|---:|---:|:---|
| holdout10 (field default) | 0.0000 | 0 | — | **no** — point mass |
| difficulty-matched | 0.0358 | 0.0911 | 0.050 | yes |

: Calibration under the two references. The holdout-derived null has no spread, so
no quantile exists to threshold at — the auditor cannot be calibrated, at any target FAR.

![The two calibration nulls. Every bootstrap draw from the holdout-referenced null lands at exactly zero, so no quantile exists to place a threshold at; the difficulty-matched null has genuine spread and supports a threshold at the target false-alarm rate.](figures/fig3_calibration_nulls.pdf){width=58%}

### 6.2 The offset causes missed detections (C3)

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

![Residual-retention score vs α, under holdout10 (field default) and the difficulty-matched reference, with decision thresholds and the missed-detection band shaded. The verdict strip below panel (b) shows the two references returning opposite verdicts across the band.](figures/fig2_missed_detection.pdf){width=92%}

### 6.3 Which MIAs inherit the bias (C4)

The offset corrupts *raw-score* MIAs — those that threshold an absolute membership score
(LOSS [3], ZLib [4], Min-K [5], Min-K++ [6]) — because the reference's artificial easiness suppresses apparent
separability. *Reference-normalized* MIAs [7], which subtract a per-example reference-model score,
are unaffected, because the offset is common-mode and cancels. At α=0.7, the residual-retention
readout is 0.162 for a raw LOSS attack against holdout10, 0.541 for the same attack against
the matched reference, and 0.970 for the reference-normalized attack — a 3.3× under-report
by the field-default raw-score/holdout combination.

We note a constraint discovered in constructing the reference-normalized comparison: computing
it against the *same* model used as the difficulty null yields an identically-zero score by
construction (a self-comparison), so calibrating a reference-normalized MIA requires a null
model *distinct* from the reference model. This is a design constraint for any future
reference-normalized auditor built on this benchmark.

| attack family | members | scoring | inherits offset? | $\rho$ at $\alpha=0.7$ |
|:---|:---|:---|:---|---:|
| raw score, holdout reference | LOSS, ZLib, Min-K, Min-K++ | absolute vs reference | **yes** | 0.162 |
| raw score, matched reference | same attacks, matched set | absolute vs reference | mitigated | 0.541 |
| reference-normalised | Reference | per-example difference | **no** | 0.970 |

: MIA-family map. Raw-score attacks threshold an absolute score against the
reference and inherit the offset; the reference-normalised family cancels it per example. The
$\alpha = 0.7$ column is the receipt quoted above.

![Retention vs α under the three scorings. The raw-score attack against the field-default holdout collapses earliest, the same attack against a difficulty-matched reference degrades later, and the reference-normalised scoring — which cancels the offset per example — holds highest throughout.](figures/fig5_mia_family.pdf){width=78%}

**Ordering vs levels.** Across the ladder, residual-retention scores are monotone in α under
*both* references: the offset biases score *levels*, not *ordering*. Leaderboard-style relative
ranking of methods is therefore comparatively robust; threshold-based forgotten/not decisions
are not.

---

## 7. Scope and the Missing Escape Hatch

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

## 8. A Difficulty-Matched Reference (C5)

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

## 9. Discussion

**Benchmark construction is an auditing decision.** The difference between the two benchmarks
we probe is not subject matter, model family, or scale; it is whether the non-member reference
was cut from the same cloth as the forget set. MUSE-News partitions one collection pass and
passes; TOFU's holdout was regenerated separately, later, and fails. Regeneration looks
innocuous — same format, same generator, same topic — and is exactly where $(\star)$ silently
breaks, since a generator's output distribution is not stable across time, prompt revisions, or
model versions. The rule follows directly: **partition one corpus; do not regenerate a
reference later.** Where regeneration is unavoidable, the offset is measurable before any
auditing is done, at the cost of one forward pass per example with an off-the-shelf model.

**Why this matters for the right to erasure.** Unlearning is motivated by data-protection
regimes in which a subject may demand removal of their data, and an audit is what converts
"we ran an unlearning method" into "the data is gone." A miscalibrated reference undermines
that conversion in the direction that matters: the offset is *conservative* for false alarms —
it makes members look less member-like, so it cannot manufacture a spurious accusation of
retention — but an auditor's purpose is to *catch* retention, and the same conservatism becomes
a missed detection, certifying clean a model that has demonstrably retained the data. In a
compliance setting that asymmetry protects the wrong party.

**Threat model.** This is not an adversarial finding — nobody built TOFU's holdout to defeat
auditing, and we found the offset with a probe that assumes no adversary. But the consequence
is exploitable: a provider free to propose the reference their own unlearning is audited
against has a lever that requires no tampering with the model, leaving its weights, outputs,
and training logs above suspicion. The reference set should therefore be fixed by the auditor
or the benchmark, never by the audited party, and validated rather than assumed — the same
discipline applied to a control group in any other experimental science.

**What a practitioner should do.** Before trusting an MIA-based unlearning number: (i) score
the forget and reference sets with a model that saw neither and check the AUC is
indistinguishable from 0.5, localising the defect before any auditing is attempted;
(ii) prefer reference-normalised attacks, which cancel a per-example difficulty offset
structurally rather than relying on the sets being matched; (iii) where a graded verdict is
needed, verify the calibration null has non-zero spread — a degenerate null means the
reference is doing no work, and a threshold derived from it is worse than none. None of this
requires training, gradients, or access to the audited model's internals.

**How far the rule generalises.** The mechanism — an unchecked difficulty mismatch between the
forget set and its reference — is the standard hazard of an unmatched control group, and is not
specific to language models, NLL, or unlearning: it appears wherever a membership decision
compares two sets assumed, rather than shown, to be comparable. What is specific here is that
the check is unusually cheap, since a never-trained model of the same family is an
off-the-shelf difficulty instrument. We do not claim reference mismatch is widespread — we
tested two benchmarks and one passed. We claim the weaker, better-supported thing: matching is
required, not automatic, depends on construction, and should be verified rather than presumed.

---

## 10. Limitations

**Synthetic retention gradient.** The missed-detection demonstration (§6.2) uses weight-space
interpolation, not real unlearning methods. The *reference mismatch itself* (§5) is measured on
real data with a real, never-trained probe; the gradient only supplies a graded axis with known
ground truth to show the mismatch has consequences under controlled conditions. Validating the
missed-detection result on real unlearning methods (grad-ascent, NPO [12], etc.) requires training
those methods and a gold retrain — deferred to future work, and the natural extension of this
paper.

**The offset is TOFU-specific; what generalizes is the check, not the defect.** We tested a
second benchmark and it passed (§5.1), so we do not claim reference mismatch is widespread. We
claim the weaker and better-evidenced thing: a difficulty-matched reference is required but
not automatic, it depends on construction, and it is cheap to verify. Two benchmarks is still
two — whether WMDP [13] or other references pass is untested, and one clean benchmark is not
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

## 11. Conclusion

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

## References

Venues are stated only where confirmed; everything else is cited as an arXiv preprint rather
than asserting a publication venue.

[1] P. Maini, Z. Feng, A. Schwarzschild, Z. C. Lipton, and J. Z. Kolter. *TOFU: A Task of
Fictitious Unlearning for LLMs.* arXiv preprint arXiv:2401.06121, 2024.

[2] W. Shi, J. Lee, Y. Huang, S. Malladi, J. Zhao, A. Holtzman, D. Liu, L. Zettlemoyer,
N. A. Smith, and C. Zhang. *MUSE: Machine Unlearning Six-Way Evaluation for Language Models.*
arXiv preprint arXiv:2407.06460, 2024. (Source of the PrivLeak metric.)

[3] S. Yeom, I. Giacomelli, M. Fredrikson, and S. Jha. *Privacy Risk in Machine Learning:
Analyzing the Connection to Overfitting.* In 31st IEEE Computer Security Foundations Symposium
(CSF), pages 268–282, 2018. arXiv:1709.01604.

[4] N. Carlini, F. Tramèr, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts,
T. Brown, D. Song, Ú. Erlingsson, A. Oprea, and C. Raffel. *Extracting Training Data from
Large Language Models.* In 30th USENIX Security Symposium, 2021. arXiv:2012.07805.

[5] W. Shi, A. Ajith, M. Xia, Y. Huang, D. Liu, T. Blevins, D. Chen, and L. Zettlemoyer.
*Detecting Pretraining Data from Large Language Models.* arXiv preprint arXiv:2310.16789, 2023.

[6] J. Zhang, J. Sun, E. Yeats, Y. Ouyang, M. Kuo, J. Zhang, H. F. Yang, and H. Li.
*Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language Models.*
arXiv preprint arXiv:2404.02936, 2024.

[7] N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, and F. Tramèr. *Membership Inference
Attacks From First Principles.* arXiv preprint arXiv:2112.03570, 2021.

[10] Y. Li, S. Bubeck, R. Eldan, A. Del Giorno, S. Gunasekar, and Y. T. Lee. *Textbooks Are
All You Need II: phi-1.5 technical report.* arXiv preprint arXiv:2309.05463, 2023.

[11] H. Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models.* arXiv preprint
arXiv:2307.09288, 2023.

[12] R. Zhang, L. Lin, Y. Bai, and S. Mei. *Negative Preference Optimization: From Catastrophic
Collapse to Effective Unlearning.* arXiv preprint arXiv:2404.05868, 2024.

[13] N. Li et al. *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning.*
arXiv preprint arXiv:2403.03218, 2024.

---

## VERIFY-BEFORE-SUBMIT checklist — status

**Verified against the committed arrays (38/38 match, same code path as the experiments):**

- [x] EXP-1: forget–holdout AUC 0.332; forget–retain 0.514; retain–holdout 0.310
- [x] EXP-1: ≈0.33 nats/token offset (exact: −0.324 vs forget, −0.357 vs retain);
      answer-length 41.8 vs 36.1 (and retain 33.1) — token lengths recomputed from
      tokenizer + dataset, not stored in the arrays
- [x] Ladder endpoints: α=0 → 0.084 (vs logged 0.090); α=1 → 2.072; drift 0.0000
- [x] §6.1 nulls: holdout sd 0.0000 / thr 0; matched sd 0.0358 / thr 0.0911 / FAR 0.050
- [x] §6.2 flip observed at rungs {0.80, 0.85, 0.90}; interpolated band α ≈ 0.77–0.94;
      fires 3/5 (field rule); same-margin control 4/5; boundaries 0.77 vs 0.94;
      range cost 0.185 (≈ one-fifth)
- [x] §6.3 α=0.7 receipt 0.162 / 0.541 / 0.970; 3.34× under-report; monotone under both refs
- [x] §7 spread 0.196 → 0.019 (≈0.20 → ≈0.02)
- [x] §8 caliper 0.05; 287/400 pairs; matched AUC 0.509; residual imbalance +0.015
- [x] §5.1 EXP-3 positive control: MUSE-News forget–holdout AUC 0.522, p 0.59 (n.s.), gap
      +0.039; controls 0.499 (p 0.98) and 0.522 (p 0.59); TOFU at matched n=100 AUC 0.294,
      p 4.8e-07; MUSE-Books excluded as contamination-confounded (0.302 / 0.153 / 0.733)
- [x] Table 3 ladder + naive accuracy + reference-normalised column: all recomputed from
      the committed arrays by `paper/make_figures.py`

**NOT re-derivable from the arrays — RE-CHECK at submission time (upstream can drift):**

- [ ] Provenance: TOFU_MIA.yaml holdout config; 2025-03-27 add date
- [ ] §7 retain checkpoint byte-identical duplicate of the full finetuned model
- [ ] §7 unlearned repos (`phi_grad_ascent_*` / `grad_diff_*` / `KL_*` / `idk_*`) empty
- [ ] §4.5 ladder "341/341 state-dict tensors aligned" — runtime assertion, re-check on rerun

**Editorial:**

- [x] Interpolation-ladder "synthetic, not unlearning" caveat appears in §4.5, §6, §10
- [x] Scope (absolute vs relative, escape hatch unavailable) appears in §1 and §7
