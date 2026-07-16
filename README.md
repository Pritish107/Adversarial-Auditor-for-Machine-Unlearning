# Adversarial Auditor for Machine Unlearning

> Did the model *really* forget? This project builds an **auditor** that tries to
> **disprove** that a model unlearned its forget-set, and turns the evidence into a
> **calibrated forgetting score**. See [PROJECT.md](PROJECT.md) for the full research spec
> and [PLAN.md](PLAN.md) for milestones and current status.

## Status: v0 (image-classifier prototype)

The v0 "hello world" runs the full auditor loop on a small CNN:

```
train baseline  ->  unlearn (honest baseline)  ->  ONE membership-inference attack  ->  forgetting score  ->  report
```

with a gold retrain-from-scratch model as the "truly forgotten" reference and a clean
**calibration seam** stubbed for later. LLM/TOFU is Milestone 2 — the interfaces here are
designed to carry over unchanged.

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

Everything in `configs/default.yaml` is tiny by default (seconds-to-minutes). Dataset,
sizes, and the forget-set are config knobs — e.g. `--dataset cifar10` for a stronger signal.

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
