"""Contract tests for the Attack / Unlearner interfaces.

These guard the seam that Milestone 2 (LLM/TOFU) has to reuse: they check that the
interfaces behave as documented on cheap synthetic inputs, no dataset download needed.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from unlearn_audit.attacks import available_attacks, build_attack
from unlearn_audit.attacks.base import Attack, AttackContext, AttackResult
from unlearn_audit.audit.score import aggregate_retention, audit
from unlearn_audit.models import SmallCNN


def _fake_ctx(separable: bool) -> AttackContext:
    """Build a context where members are either easy (separable) or identical to non-members."""
    torch.manual_seed(0)
    model = SmallCNN(in_channels=1, num_classes=10, image_size=28)
    n = 32
    if separable:
        # members = a single memorized class the model is confident on vs. random labels
        x_mem = torch.zeros(n, 1, 28, 28)
        y_mem = torch.zeros(n, dtype=torch.long)
        x_non = torch.randn(n, 1, 28, 28)
        y_non = torch.randint(0, 10, (n,))
    else:
        x = torch.randn(n, 1, 28, 28)
        y = torch.randint(0, 10, (n,))
        x_mem, y_mem, x_non, y_non = x, y, x.clone(), y.clone()
    return AttackContext(
        target_model=model,
        member_data=TensorDataset(x_mem, y_mem),
        nonmember_data=TensorDataset(x_non, y_non),
        retain_data=TensorDataset(x_non, y_non),
        device=torch.device("cpu"),
        batch_size=16,
    )


def test_attack_returns_result_in_range():
    atk = build_attack("loss_mia")
    assert isinstance(atk, Attack)
    res = atk.run(_fake_ctx(separable=False))
    assert isinstance(res, AttackResult)
    assert 0.0 <= res.retention_score <= 1.0
    assert res.name == "loss_mia"


def test_registry_lists_attack():
    assert "loss_mia" in available_attacks()


def test_aggregate_and_audit_shapes():
    results = [
        AttackResult(name="a", retention_score=0.2),
        AttackResult(name="b", retention_score=0.7),
    ]
    assert aggregate_retention(results, "max") == 0.7
    rep = audit(results, aggregate="max")
    assert rep.retention_score == 0.7
    assert abs(rep.forgetting_score - 0.3) < 1e-9
    # v0 calibration is an honest stub, not silently "calibrated"
    assert rep.calibration.calibrated is False
    assert rep.calibration.false_alarm_rate is None
