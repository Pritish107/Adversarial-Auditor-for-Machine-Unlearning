# The Reference Is the Bug

**Difficulty-Mismatched Non-Member Sets Break Membership-Inference Auditing of Unlearning**

📄 **Paper (Zenodo, open access):** https://doi.org/10.5281/zenodo.21740682

Machine unlearning audits verify "forgetting" by comparing forgotten data against a
non-member reference set the model never trained on — assuming the two are *exchangeable*
in everything except membership. This project shows that assumption fails on **TOFU**, the
most-cited LLM unlearning benchmark.

## The finding

Measured with a model that never trained on either set (so any gap is pure difficulty, not
memorization), TOFU's designated holdout is systematically **easier** than the forget set —
**AUC 0.332** where exchangeable sets should give 0.5 (p < 1e-15). Consequences:

- **Absolute-threshold auditing becomes uncalibratable** — the holdout-derived null collapses
  to a degenerate point mass, with no valid decision threshold at any target false-alarm rate.
- **Missed detections** — on a controlled retention gradient, a difficulty-matched auditor
  flags residual retention where the standard holdout-based rule reports clean forgetting.
- **A model that still remembers can pass the forgetting test.**

The same probe **passes** on a second benchmark (**MUSE-News**, AUC 0.522, n.s.) — so it
discriminates a sound reference from an unsound one rather than flagging everything. The two
differ in one design choice: MUSE partitions a single collection pass; TOFU regenerated its
holdout separately, months later. The rule: **partition one corpus, don't regenerate a
reference.**

A difficulty-matched reference construction (included) restores calibration.

## Reproducibility

Every table and figure in the paper regenerates from committed result arrays — **no GPU
required**. See [`unlearn_audit/experiments/`](unlearn_audit/experiments/) for the
experiments and rerun instructions.

```bash
python paper/make_figures.py     # regenerates all 5 figures from the committed .npz arrays
```

Re-running the experiments themselves *does* need a GPU and downloads model weights
(~45 min total on an 8 GB card); regenerating the figures and tables from the committed
arrays does not.

## Repository

```
unlearn_audit/
  experiments/          # the paper's experiments + committed result arrays
    exp1_difficulty_probe.py     # EXP-1: the difficulty offset on TOFU
    exp2_matched_reference.py    # EXP-2: matched reference, calibration, retention ladder
    exp2b_fine_sweep.py          # EXP-2b: fine sweep across the missed-detection band
    exp3_muse_probe.py           # EXP-3: MUSE-News positive control
    results/                     # 5 .npz arrays — every paper number derives from these
    README.md                    # per-experiment rerun instructions + expected output
  attacks/              # Attack interface + loss-based membership inference
  audit/                # retention-score aggregation + calibration
  unlearn/              # Unlearner interface + gradient-ascent / retrain / guardrail
  llm/                  # TOFU & MUSE data, Phi-1.5 loading, answer-NLL scoring
  data.py  models.py  pipeline.py  report.py  cli.py  multiseed.py  utils.py

paper/
  paper.pdf             # the compiled paper (17 pp.)
  PAPER_DRAFT.md        # source of truth for the text
  figures/              # 5 figures, .pdf (vector, for LaTeX) + .png
  make_figures.py       # regenerates every figure from the committed arrays
  build_pdf.sh          # PAPER_DRAFT.md -> paper.pdf (pandoc + tectonic)
  prepare_for_pdf.py  preamble.tex  README.md    # build toolchain + its documentation

tests/                  # interface contracts + smoke tests
configs/                # YAML configs for the CLI
PROJECT.md  PLAN.md     # original research spec + full development log
```

## Setup

```bash
git clone https://github.com/Pritish107/Adversarial-Auditor-for-Machine-Unlearning.git
cd Adversarial-Auditor-for-Machine-Unlearning

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# torch/torchvision MUST come from the PyTorch index first: requirements.txt pins the
# CUDA 12.4 builds (torch==2.6.0+cu124), which are not published on PyPI.
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

pip install -r requirements-paper.txt   # matplotlib + scipy, for figure regeneration
```

**CPU-only / no GPU.** Install the CPU builds from PyPI, then the remaining pins directly —
`requirements.txt` cannot be used as-is here, because it hard-pins the `+cu124` builds:

```bash
pip install torch==2.6.0 torchvision==0.21.0
pip install numpy==2.4.4 scikit-learn==1.9.0 pyyaml==6.0.3 pytest==9.1.1 \
            transformers==5.14.1 datasets==5.0.0 accelerate==1.14.0
pip install -r requirements-paper.txt
```

Torch is required even on the no-GPU path — the scoring and audit code imports it — but the
CPU build is enough to regenerate every figure and table.

Run the tests with:

```bash
python -m pytest
```

## Citation

If you use this work, please cite the Zenodo record: https://doi.org/10.5281/zenodo.21740682
