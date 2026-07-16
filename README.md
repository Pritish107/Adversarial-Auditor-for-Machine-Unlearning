# Adversarial Auditor for Machine Unlearning

> Did the model *really* forget? This project builds an **auditor** that tries to
> **disprove** that a model unlearned its forget-set, and turns the evidence into a
> **calibrated forgetting score**. See [PROJECT.md](PROJECT.md) for the full research spec
> and [PLAN.md](PLAN.md) for milestones and current status.

## Status: v0 (image-classifier prototype)

The v0 auditor runs the full loop on a small CNN (CIFAR-10 by default):

```
train baseline  ->  {unlearning methods}  ->  membership-inference attack  ->  calibrated forgetting score  ->  report
```

It audits three ground-truth-labeled cases against a gold retrain-from-scratch reference,
with a **real (first-cut) calibration**: the gold model is bootstrapped into a null
distribution, giving a decision threshold at a target false-alarm rate.

**Headline result (CIFAR-10, seed 0):** a *guardrail* fake-unlearner (frozen weights + an
output filter) shows **forget-accuracy 0.000** — a naive eval calls it perfectly forgotten —
yet the auditor reads **MIA-AUC 0.731, identical to the un-unlearned baseline**, and flags
full residual retention. Gold sits at chance (AUC 0.506); crude gradient-ascent lands in
between (0.648). See [PLAN.md](PLAN.md) for the full table and the two honest caveats
(threat-model boundary; FAR is a lower bound).

LLM/TOFU is Milestone 2 — the interfaces here are designed to carry over unchanged.

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install numpy scikit-learn pyyaml pytest
```

(CPU-only? Drop the `--index-url` line and just `pip install torch torchvision`.)

## Run

```bash
.venv/Scripts/python.exe -m unlearn_audit.cli --config configs/default.yaml
```

`configs/default.yaml` is CIFAR-10 (first run downloads ~170MB). Sizes, epochs, the
forget-set, the method list, and calibration are all config knobs — e.g. `--dataset mnist`
for a fast, weaker-signal run.

## Tests

```bash
.venv/Scripts/python.exe -m pytest
```

`test_interfaces.py` guards the Attack/Unlearner contracts (no download); `test_smoke.py`
runs the whole pipeline on a tiny MNIST subset.

## Architecture

```
unlearn_audit/
  data.py            # forget/retain/test split behind one DataBundle abstraction
  models.py          # small CNN + train/eval + per-sample loss
  unlearn/           # Unlearner interface + gradient_ascent + retrain (gold)   [registry]
  attacks/           # Attack interface + membership_inference (loss MIA)       [registry]
  audit/             # score aggregation + calibration seam (stub in v0)
  report.py          # readable before/after text report
  pipeline.py        # wires train -> unlearn -> audit
  cli.py             # entry point
```

**Design intent:** attacks and unlearners are drop-in modules behind shared interfaces, so
adding attack #2 (relearning-speed) / #3 (activation probing) or swapping in an LLM is a new
module, not a core rewrite. `attacks/base.py::AttackContext` is deliberately general (member/
non-member data, optional reference model, optional retrain hook) for exactly that reason.

**Score convention:** every attack reports `retention_score` in `[0,1]` (higher = more
residual retention = worse forgetting). The audit aggregates with `max` (fail if *any* attack
recovers info); `forgetting_score = 1 - retention_score`.
