# Project: An Adversarial Auditor for Machine Unlearning ("Did the model *really* forget?")

> Paste this file into a new chat as context. It describes the project, the research
> question, the plan, and where I need help. I'm building this to publish a research paper.

---

## One-line summary

Build the strongest possible **auditor** that tries to prove a model did **not** unlearn
data it claims to have forgotten, and output a **calibrated "forgetting score"** for any
(model, forget-set) pair.

## Why this problem

Machine unlearning (making a trained model "forget" specific data, e.g. for the right to
erasure / GDPR) is a fast-growing area, but recent surveys flag the same core gap:
**verification**. There is no reliable, standardized way to confirm that unlearning actually
worked. Proposing a *positive* certificate ("this model provably forgot X") is very hard.

**The key insight / tractable slice:** flip the problem. Instead of proving forgetting,
build an adversary that tries to **disprove** it. A strong negative result — "here is
evidence the 'forgotten' data is still recoverable" — is achievable, useful, and publishable,
even where a positive guarantee is out of reach. This reframes an open problem into a
concrete, buildable auditor.

## Research question

> Given only a model that claims to have unlearned a forget-set (and, ideally, its
> pre-unlearning counterpart), can we reliably detect residual retention of that data,
> quantify it as a calibrated score, and rank unlearning methods by how well they truly forget?

## Core idea

An **audit battery** of complementary attacks, each probing a different signature of
residual knowledge. The model fails the audit if *any* attack recovers the "forgotten"
information above a calibrated threshold.

Candidate attacks to implement (start with 2–3, add more):
1. **Membership inference** — can we still tell the forget-set was in training?
2. **Activation steering / probing** — do internal activations still encode the forgotten
   content, even if outputs look clean?
3. **Relearning-speed probe** — does the model re-learn the forget-set suspiciously fast
   (implying it never truly left)? This is a strong, underused signal.
4. **Contrastive / adversarial decoding** — coax out suppressed-but-present knowledge.
5. **Logit-difference vs. pre-unlearning model** — where a base model is available.

Output: a single **calibrated forgetting score** with a stated false-alarm rate, not just
a binary pass/fail.

## Scope (to keep it a focused first paper)

- Small open models (e.g. GPT-2 / Pythia / a ~1–3B open LLM, or even classifiers first).
- Existing unlearning benchmarks as the testbed (see below).
- 2–3 attacks in the audit battery for v1; frame more as future work.
- One clean headline result: *the auditor catches residual retention that standard
  post-unlearning evaluations declare "successfully forgotten."*

## Datasets / benchmarks to use

Look up and evaluate on established unlearning benchmarks (verify current versions):
- **TOFU** (synthetic author profiles — clean forget/retain split, very commonly used)
- **MUSE** (six-way unlearning evaluation for language models)
- **WMDP** (hazardous-knowledge unlearning benchmark)
- Optionally a simple image classifier + CIFAR/MNIST forget-set for a fast prototype baseline.

## Baselines / comparison points

- The unlearning methods' **own** reported evaluation metrics (show they over-claim).
- Naive membership-inference-only auditing (show the battery beats any single attack).
- Guardrail / output-filtering "unlearning" (show it's easily defeated by the auditor).

## Evaluation metrics

- **Detection AUC**: clean (truly forgotten) vs. tampered (residual retention) cases.
- **Calibration**: measured false-alarm rate of the forgetting score.
- **Ranking correlation**: does the auditor correctly rank unlearning methods by forgetting quality?
- **Robustness**: performance as retention is made progressively stealthier.

## Deliverables

1. Open-source auditor library (clean API + CLI + readable report).
2. A labeled evaluation suite of "truly forgotten" vs. "residual retention" cases.
3. Paper: problem framing, method, results, honest limitations.

## Rough timeline (target ~8–10 weeks)

- Wk 1–2: read core papers; reproduce one unlearning method + one benchmark eval.
- Wk 3–4: implement attacks 1–2 (membership inference, relearning-speed probe).
- Wk 5–6: add attack 3, build the combined calibrated score.
- Wk 7: run full evaluation across methods/benchmarks; ablations.
- Wk 8–10: write-up, limitations, polish repo, submit.

## Papers / prior work to read first (verify exact titles/authors when searching)

- A recent **survey of machine unlearning** (e.g. "From Forgetting to Future," 2026) —
  for the verification-gap framing.
- Position paper arguing **"machine unlearning" is overused in LLMs** — for honest scoping.
- Work on **extracting unlearned information via activation steering**.
- **Contrastive-decoding**-based knowledge extraction from unlearned models.
- **MNEME / sparse model diffing** for unlearning side-effects — related, and a possible
  extension.
- Benchmark papers: **TOFU**, **MUSE**, **WMDP**.

## Target venues

- Primary: **SaTML** (Secure & Trustworthy ML) — great fit for auditing/verification.
- Workshops at **ICLR / NeurIPS** (SoLaR / SafeGenAI / trustworthy-ML).
- arXiv preprint first, regardless.

## Tech stack

Python, PyTorch, Hugging Face Transformers, single-GPU (Colab Pro / one consumer GPU is
enough for the small-model scope). Standard ML tooling; no large-scale training required.

## Where I want help (in the new chat)

- Sanity-check the framing and make sure the "negative certificate" angle is genuinely novel
  and not already covered.
- Help me pick the exact first model + benchmark + 2 attacks for a minimal working prototype.
- Help design the calibrated forgetting score and its false-alarm-rate estimation.
- Later: help write and structure the paper.

## Status

Starting from scratch. Nothing built yet. Next concrete step: reproduce one unlearning
method on TOFU (or a classifier prototype) and run a single membership-inference attack
against it as the "hello world" of the auditor.
