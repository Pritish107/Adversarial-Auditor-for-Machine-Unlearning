# PLAN — Adversarial Auditor for Machine Unlearning

Running tracker of milestones, decisions, and open questions. Source of truth for *goals*
is [PROJECT.md](PROJECT.md); this file tracks *execution state*.

Last updated: 2026-07-16.

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
- [ ] Run full pipeline; confirm a real before/after retention drop; first commit
- [ ] Sanity-check: does MNIST give a meaningful MIA signal, or bump default to CIFAR-10?

### M1 — v0 hardening / labeled eval seeds
- [ ] Clean vs. tampered cases (truly-forgotten gold vs. residual-retention) as labeled data
- [ ] Detection AUC harness (clean vs. tampered), first real calibration from gold null
- [ ] Second unlearning method to test ranking

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
- **Calibration is an honest stub**, not a fake number: it returns `calibrated=False`,
  `far=None`, and raises `NotImplementedError` if handed a null distribution — the real path
  is wired but explicitly unbuilt until M3.
- **MNIST default** for instant iteration; `--dataset cifar10` when we want stronger signal.

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
