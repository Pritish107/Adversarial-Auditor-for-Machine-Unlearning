"""Gradient-ascent unlearning ("finetune-to-forget").

The simplest honest baseline: starting from the trained model, take gradient *ascent*
steps on the forget-set (maximize its loss) so the model stops fitting it. Optionally
interleave ordinary descent on a retain batch so utility doesn't collapse — that keeps
the baseline honest rather than just breaking the whole model.

This is deliberately imperfect: it usually leaves a residual membership signal, which is
exactly what the auditor should catch. The gold `retrain` reference is the clean contrast.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import Unlearner, UnlearnContext


class GradientAscentUnlearner(Unlearner):
    name = "gradient_ascent"

    def unlearn(self, ctx: UnlearnContext) -> nn.Module:
        p = ctx.params
        model = self._clone(ctx.model).to(ctx.device).train()
        opt = torch.optim.Adam(model.parameters(), lr=p.get("lr", 5e-4))

        forget_loader = DataLoader(ctx.data.forget, batch_size=ctx.batch_size, shuffle=True)
        retain_reg = p.get("retain_reg", True)
        retain_iter = None
        if retain_reg:
            retain_loader = DataLoader(ctx.data.retain, batch_size=ctx.batch_size, shuffle=True)

        for _ in range(p.get("epochs", 3)):
            for xf, yf in forget_loader:
                xf, yf = xf.to(ctx.device), yf.to(ctx.device)
                opt.zero_grad()
                # Ascent on forget = descend the NEGATIVE loss.
                loss = -F.cross_entropy(model(xf), yf)
                if retain_reg:
                    if retain_iter is None:
                        retain_iter = iter(retain_loader)
                    try:
                        xr, yr = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        xr, yr = next(retain_iter)
                    xr, yr = xr.to(ctx.device), yr.to(ctx.device)
                    loss = loss + F.cross_entropy(model(xr), yr)
                loss.backward()
                opt.step()
        return model
