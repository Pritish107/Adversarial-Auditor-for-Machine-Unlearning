"""Smoke test: the whole pipeline runs on a tiny MNIST subset and produces sane numbers.

Downloads MNIST on first run (small). Marked slow-ish but still seconds on CPU.
"""
from __future__ import annotations

from unlearn_audit.pipeline import run
from unlearn_audit.utils import load_config


def test_pipeline_runs_end_to_end():
    cfg = load_config("configs/test_tiny.yaml")
    text, summaries = run(cfg)

    # We get a report and both baseline + unlearned summaries.
    assert "AUDIT" in text
    labels = {s.label for s in summaries}
    assert "baseline (pre-unlearn)" in labels
    assert "unlearned" in labels

    for s in summaries:
        # scores are well-formed and in range
        assert 0.0 <= s.audit.retention_score <= 1.0
        assert 0.0 <= s.audit.forgetting_score <= 1.0
        assert abs(s.audit.retention_score + s.audit.forgetting_score - 1.0) < 1e-9
