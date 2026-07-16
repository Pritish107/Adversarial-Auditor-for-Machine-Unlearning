"""Retrain-from-scratch on retain-only — the GOLD forgetting reference.

This never sees the forget-set, so it is the ideal "truly forgot" model. The auditor's
attacks should read near-chance against it; that anchors what a *clean* audit looks like
and is the seed for real calibration later (the null distribution of "truly forgotten").
"""
from __future__ import annotations

import torch.nn as nn
from torch.utils.data import DataLoader

from .. import models
from .base import Unlearner, UnlearnContext


class RetrainUnlearner(Unlearner):
    name = "retrain"

    def unlearn(self, ctx: UnlearnContext) -> nn.Module:
        p = ctx.params
        data = ctx.data
        model = models.build_model(
            p.get("arch", "small_cnn"), data.in_channels, data.num_classes, data.image_size
        )
        retain_loader = DataLoader(data.retain, batch_size=ctx.batch_size, shuffle=True)
        return models.train(
            model, retain_loader,
            epochs=p.get("epochs", 3), lr=p.get("lr", 1e-3), device=ctx.device,
        )
