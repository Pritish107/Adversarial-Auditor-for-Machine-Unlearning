"""Smoke test for the multi-seed harness on the tiny MNIST config (2 seeds, seconds)."""
from __future__ import annotations

from unlearn_audit.multiseed import render_multiseed, run_multiseed
from unlearn_audit.utils import load_config


def test_multiseed_runs_and_reports():
    cfg = load_config("configs/test_tiny.yaml")
    out = run_multiseed(cfg, seeds=[0, 1])

    # pooled null aggregates both seeds' gold bootstraps
    assert out["pooled_null_n"] == 2 * cfg["calibration"]["n_boot"]
    assert out["calib"].calibrated is True
    assert 0 <= out["ordering_preserved"] <= out["n"] == 2

    # every case has mean/std tuples and a detected count in range
    for lbl, p in out["per_case"].items():
        assert len(p["auc"]) == 2 and len(p["retention"]) == 2
        assert 0 <= p["detected"] <= 2

    # gap list has one entry per seed; report renders
    assert len(out["gaps"]) == 2
    assert "HEADLINE" in render_multiseed(cfg, out)
