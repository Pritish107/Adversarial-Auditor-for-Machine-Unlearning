"""The Attack interface — the core seam of the whole auditor.

Every attack, present and future, implements `Attack.run(ctx) -> AttackResult`. The
context carries *everything* an attack might need; a given attack uses only the slice it
cares about. This is what lets attack #2 (relearning-speed) and #3 (activation probing)
drop in later as new modules instead of a rewrite, and what lets the LLM/TOFU setting
reuse the same battery — the harness just fills the same `AttackContext` fields with a
language model and text member/non-member sets.

Design notes on generality:
  * `target_model`      — the model under audit (the *unlearned* one).
  * `reference_model`   — an optional anchor: the pre-unlearning model, or the gold
                          retrain. Attacks that need a baseline (logit-diff) use it;
                          others ignore it.
  * `member_data` /
    `nonmember_data`    — the forget-set (claimed forgotten) vs. held-out non-members.
                          Named generically, not "forget/test", so the text setting maps
                          on cleanly.
  * `retrain_fn`        — a callable the harness provides so train-access attacks
                          (relearning-speed) can fine-tune a *copy* without knowing how
                          models are built. None when unavailable.

Score convention (uniform across attacks): higher = MORE evidence of residual retention
(i.e. more evidence the model did NOT forget). ~0 means the attack found nothing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset


@dataclass
class AttackContext:
    target_model: nn.Module
    member_data: Dataset                     # forget-set (trained-on, claimed forgotten)
    nonmember_data: Dataset                  # held-out (never trained-on)
    retain_data: Dataset                     # kept training data (some attacks want it)
    device: torch.device
    batch_size: int
    reference_model: Optional[nn.Module] = None
    retrain_fn: Optional[Callable] = None    # (model, dataset, steps) -> model, for relearning-speed
    params: dict = field(default_factory=dict)


@dataclass
class AttackResult:
    name: str
    retention_score: float        # [0,1], higher = more residual retention detected
    detail: dict = field(default_factory=dict)   # attack-specific diagnostics


class Attack(ABC):
    name: str = "attack"

    @abstractmethod
    def run(self, ctx: AttackContext) -> AttackResult:
        """Probe ctx.target_model for residual retention of ctx.member_data."""
