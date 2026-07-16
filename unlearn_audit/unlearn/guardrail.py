"""Guardrail fake-unlearner — the strawman PROJECT.md wants the auditor to defeat.

It does NOT touch the weights. It wraps the frozen baseline with a *perfect* forget-set
detector (exact hashes of the forget inputs — the provider knows precisely what they are
suppressing) and an output filter:

  * forward(x)  -> the baseline's raw logits, UNCHANGED. This is the scoring path the
                   loss-MIA reads, so the membership trace is fully intact.
  * predict(x)  -> raw argmax, EXCEPT inputs the detector flags get their label corrupted,
                   so forget-set *accuracy* collapses to ~chance and a naive eval calls it
                   "forgotten".

Because the model memorized the forget-set, its raw argmax there is the true label; the
filter maps it to (pred + 1) % C, guaranteeing a wrong answer -> forget accuracy ~ 0, while
loss stays exactly baseline. That gap is the whole point: accuracy-eval is fooled, the
auditor is not.

Threat-model boundary (see PLAN.md): this models output-LABEL filtering. A guardrail that
also corrupted the logits would defeat loss-MIA specifically — which is exactly why the
audit must be a battery (activation probing / relearning-speed in M3), not one attack.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .base import Unlearner, UnlearnContext


def _hash(x: torch.Tensor) -> str:
    return hashlib.sha1(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _signature(dataset: Dataset) -> set[str]:
    return {_hash(dataset[i][0]) for i in range(len(dataset))}


class GuardrailModel(nn.Module):
    def __init__(self, base: nn.Module, signature: set[str], num_classes: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.signature = signature
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)                          # raw, unchanged — the auditor's path

    def _flagged(self, x: torch.Tensor) -> torch.Tensor:
        xb = x.detach().cpu()
        flags = torch.tensor([_hash(xb[i]) in self.signature for i in range(len(xb))],
                             dtype=torch.bool)
        return flags.to(x.device)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        preds = self.base(x).argmax(1)
        flags = self._flagged(x)
        if flags.any():
            preds = preds.clone()
            preds[flags] = (preds[flags] + 1) % self.num_classes   # corrupt the shown label
        return preds


class GuardrailUnlearner(Unlearner):
    name = "guardrail"

    def unlearn(self, ctx: UnlearnContext) -> nn.Module:
        base = self._clone(ctx.model).to(ctx.device).eval()
        sig = _signature(ctx.data.forget)
        return GuardrailModel(base, sig, ctx.data.num_classes).to(ctx.device)
