# M3 reference-construction experiments

Three experiments on one question: **does the choice of non-member reference set change what
the auditor concludes?** Each writes its per-example arrays to `results/`, which are committed,
so every table below reproduces **without a GPU**.

> ### ⚠️ The α interpolation ladder is a SYNTHETIC PROBE, not an unlearning method
> EXP-2 and EXP-2b build models by blending weights:
> `θ(α) = (1−α)·θ_baseline + α·θ_basePhi`.
> This is a **synthetic monotone retention gradient** whose only purpose is to supply graded
> scores so the reference question can be tested at all — the official TOFU unlearned
> checkpoints are empty repos (see PLAN.md "M3 LIMITATION"). It is **not** an unlearning
> method, does not approximate one, and must **never** be reported as one or compared against
> published unlearning numbers. Real methods (grad-ascent / NPO / grad-diff) belong to the
> training phase.

## Shared setup

- Member set: `locuslab/TOFU` config `forget10` (400 QAs), baked into `locuslab/tofu_ft_phi-1.5`.
- Non-member reference: `holdout10` (400) — verified to be **OpenUnlearning's own** MIA holdout
  (`configs/data/datasets/TOFU_MIA.yaml` → `path=locuslab/TOFU, name=holdout10`).
- Null / gold substitute: base `microsoft/phi-1_5`, which saw none of these sets.
- Score: answer-only, length-normalized NLL/token via the repo's `make_answer_nll_scorer`, run
  through the repo's own `LossMIA` AUC convention — so these numbers are directly comparable to
  every other result in the project.
- `retain90` is a difficulty **diagnostic only, never a reference**: `full` = `forget10` ⊎
  `retain90` and the finetune trained on `full`, so every retain90 row is a **member**.

## EXP-1 — `exp1_difficulty_probe.py`

**Shows:** the forget/holdout gap seen by a never-trained model is a **difficulty asymmetry**,
not membership — and it is **holdout-specific**.

```
python -m unlearn_audit.experiments.exp1_difficulty_probe
```

Expected (base-Phi, answer-only NLL/token):

| comparison | member | non-mem | AUC | gap |
|---|---|---|---|---|
| forget10 vs holdout10 (n=100, C3 replication) | 2.071 | 1.684 | 0.294 | −0.387 |
| forget10 vs holdout10 (full 400/400) | 2.072 | 1.748 | **0.332** | −0.324 |
| *control* forget10 vs retain90 | 2.072 | 2.106 | **0.514** | +0.034 |
| *control* retain90 vs holdout10 | 2.106 | 1.748 | 0.310 | −0.357 |

Two splits from the original TOFU generation run are matched to within noise (0.514); holdout10
is ~0.33 nats/token easier than **both**. Not a length artifact — scoring is length-normalized
and holdout answers are *longer* (41.8 vs 36.1 / 33.1 mean tokens).

## EXP-2 — `exp2_matched_reference.py`

**Shows:** the offset is not harmless. It makes the auditor **uncalibratable**, and it
under-reports graded retention by up to 3.3× at the decision boundary.

```
python -m unlearn_audit.experiments.exp2_matched_reference     # ~25 min on a 4060
```

Builds three references — `R-holdout` (field default = OpenUnlearning's LOSS MIA), `R-matched`
(caliper-0.05 pairs matched on base-Phi NLL, n=287), `R-delta` (= their Reference MIA) — and
sweeps α ∈ {0, .15, .3, .5, .7, 1}.

**Endpoint gate (hard fail):** α=0 must reproduce baseline member NLL ≈0.090 and α=1 ≈2.072. If
they miss, the script exits 1 rather than reporting the middle of a broken blend. Observed:
0.084 / 2.072, with max per-example drift **0.0000** vs EXP-1 — α=1 re-derives the reference
rather than trusting the cache. State dicts are asserted key- and shape-aligned (341/341).

Retention by α: `R-holdout` 0.997, 0.993, 0.973, 0.733, **0.162**, 0.000 · `R-matched` 0.996,
0.991, 0.981, 0.878, **0.541**, 0.019 · `R-delta` 0.999, 0.997, 0.994, 0.985, **0.970**, 0.000.

Key outputs:
- **Calibration:** `R-holdout` null is a **degenerate point mass** (sd 0.0000, threshold 0) —
  no threshold exists at any target FAR. `R-matched` null has real spread (sd 0.0358,
  threshold 0.0911, FAR 0.050). Difficulty-matching makes the auditor calibratable.
- **MIA-family map:** the offset corrupts **raw-score** MIAs (LOSS, ZLib, MinK, MinK++) and
  leaves **Reference** intact — per-example normalization cancels the difficulty term.
- **Reference-MIA null is structurally degenerate:** base-Phi scored under `R-delta` compares to
  itself, so Δ≡0 and AUC=0.500 exactly. Calibrating a Reference MIA **requires a null model
  distinct from the reference model**.
- Retention is monotone in α under both references: the offset biases score **levels**, not
  **ordering**.

## EXP-2b — `exp2b_fine_sweep.py`

**Shows:** a benchmark-induced **missed detection** — the unsafe direction for an auditor.

```
python -m unlearn_audit.experiments.exp2b_fine_sweep           # ~15 min on a 4060
```

Sweeps α ∈ {0.75, .8, .85, .9, .95}, the interval EXP-2's coarse grid skipped, and reports each
rung under four decision rules. Rules (2) and (3) are **mathematically identical**
(`retention > 0` ⟺ `AUC > 0.5`), so the degenerate default lands exactly on the field
convention — beating it is beating real practice, not a straw man.

| α | hold AUC | match ret | (1) matched | (3) AUC>0.5 | (4) hold@0.091 |
|---|---|---|---|---|---|
| 0.75 | 0.520 | 0.439 | DETECTED | DETECTED | forgotten |
| 0.80 | 0.467 | 0.339 | **DETECTED** | *forgotten* | *forgotten* |
| 0.85 | 0.423 | 0.245 | **DETECTED** | *forgotten* | *forgotten* |
| 0.90 | 0.386 | 0.160 | **DETECTED** | *forgotten* | *forgotten* |
| 0.95 | 0.355 | 0.081 | forgotten | forgotten | forgotten |

Fires on 3/5 rungs against rule (3), 4/5 against control (4). The matched auditor's boundary is
α≈0.94, holdout10's is α≈0.76 — the offset costs ~1/5 of the detectable range. Ground truth is
unambiguous: every α<1 contains baseline weight, so retention **is** present. Control (4)
transplants the matched threshold onto holdout scores and still flips, isolating the cause to
the **reference set** rather than the calibration.

### ★ Scope — never state the flip without it

The script prints this control alongside the verdict. The offset corrupts
**absolute-threshold** MIA auditing (the "AUC ≈ 0.5 means forgotten" convention, which is what
this auditor and standard practice both do). **Gold-referenced / relative rules — MUSE's
PrivLeak, TOFU's forget_quality — are immune:** they compare a candidate to a retrain model on
the *same* non-member set, so the offset is common-mode and cancels. Measured, reference-set
spread collapses from **~0.20 (absolute) to ~0.02 (relative)**.

The catch, and why the finding still stands: a relative rule needs a valid gold retrain, and
**TOFU/Phi ships none** — `tofu_ft_retain90_phi-1.5` is a byte-identical duplicate of the full
model and every `phi_grad_ascent_*` / `grad_diff_*` / `KL_*` / `idk_*` repo is empty. The offset
breaks the auditing mode the benchmark actually supports, and the mode that would survive it is
the one the benchmark cannot supply.

## EXP-3 — `exp3_muse_probe.py`

**Shows:** the offset is **TOFU-specific** — and that makes this the **positive control** EXP-1
needed. The same probe fires on TOFU and stays at chance on MUSE-News at the *same* sample
size, so it discriminates a sound reference from an unsound one rather than always finding a
gap.

```
python -m unlearn_audit.experiments.exp3_muse_probe [--n N] [--books]   # ~3 min on a 4060
```

Target is `muse-bench/MUSE-News`, config `privleak`, splits `{forget, holdout, retain}` (100
texts each) — three mutually disjoint subsets of **one** corpus, BBC articles published after
August 2023, collected in a single pass. Probe is the same base `microsoft/phi-1_5`, which
predates that window and is therefore blind to all three splits. MUSE is prose, so the scored
span is the whole text (uniform 512-token prefix) rather than an answer; the reduction is the
repo scorer with prompt `"{q}"` over `("", text)` pairs, so the number is computed by the same
function TOFU's was.

| comparison | member | non-mem | AUC | gap | p |
|---|---|---|---|---|---|
| forget vs holdout | 2.994 | 3.034 | **0.522** | +0.039 | 0.59 (n.s.) |
| *control* forget vs retain | 2.994 | 2.992 | 0.499 | −0.003 | 0.98 (n.s.) |
| *control* retain vs holdout | 2.992 | 3.034 | 0.522 | +0.042 | 0.59 (n.s.) |
| **TOFU at matched n=100** | 2.071 | 1.684 | **0.294** | −0.387 | **4.8e-07** |

The last row is the point: the MUSE null is not a power artifact. The script prints it
automatically whenever `exp1_probe_nll.npz` is present.

**Construction rule this yields:** *partition one corpus; do not regenerate a reference later.*
MUSE-News splits a single collection pass and passes; TOFU's holdout was regenerated ~14 months
after release and fails.

> **`--books` is CONFOUNDED and never load-bearing.** MUSE-Books is Harry Potter, which
> essentially every general-purpose base LM saw in pretraining, so no never-trained probe
> exists and a gap cannot be attributed to difficulty rather than memorization. Its own numbers
> show it: forget–holdout 0.302, but retain–holdout 0.153 and forget–retain 0.733 — the three
> splits disagree with each other in a way no membership account explains. Kept for
> reproducibility, excluded from every claim.

## `results/`

| file | contents |
|---|---|
| `exp1_probe_nll.npz` | base-Phi per-example NLL for forget10 / holdout10 / retain90 |
| `exp2_ladder.npz` | per-α NLLs + accuracies, matched-pair indices, both bootstrap nulls, guardrail arrays |
| `exp2b_fine.npz` | per-α NLLs, AUCs, retentions, accuracies for the fine sweep |
| `exp3_muse_news_nll.npz` | base-Phi per-example NLL for MUSE-News forget / holdout / retain |
| `exp3_muse_books_nll.npz` | same for MUSE-Books — **confounded, excluded from all claims** |
