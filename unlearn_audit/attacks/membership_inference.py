"""Loss-based membership-inference attack.

Idea: a model fits its training data more tightly than unseen data, so training members
tend to have *lower* loss than non-members. If a model truly forgot the forget-set, its
per-sample losses there should look like held-out non-members — indistinguishable. Any
remaining separability is residual membership signal: evidence it did NOT forget.

We quantify separability with ROC-AUC of the score `-loss` against the member/non-member
label. AUC in [0.5, 1]; 0.5 = indistinguishable (good forgetting). We map the membership
*advantage* to a retention score in [0, 1]:

    retention = clip(2 * (AUC - 0.5), 0, 1)

AUC below 0.5 (members somehow *higher* loss) is not evidence of retention, so it floors
at 0. This same attack works unchanged on an LLM: swap per-sample classification loss for
per-sequence token NLL and the rest is identical — which is the point of the interface.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .. import models
from .base import Attack, AttackContext, AttackResult


class LossMIA(Attack):
    name = "loss_mia"

    def run(self, ctx: AttackContext) -> AttackResult:
        member_loss = models.per_sample_loss(
            ctx.target_model, ctx.member_data, batch_size=ctx.batch_size, device=ctx.device
        ).numpy()
        nonmember_loss = models.per_sample_loss(
            ctx.target_model, ctx.nonmember_data, batch_size=ctx.batch_size, device=ctx.device
        ).numpy()

        # label 1 = member (forget-set), 0 = non-member; score = -loss (low loss -> member)
        labels = np.concatenate([np.ones_like(member_loss), np.zeros_like(nonmember_loss)])
        scores = -np.concatenate([member_loss, nonmember_loss])
        auc = float(roc_auc_score(labels, scores))
        retention = float(np.clip(2.0 * (auc - 0.5), 0.0, 1.0))

        return AttackResult(
            name=self.name,
            retention_score=retention,
            detail={
                "auc": auc,
                "member_loss_mean": float(member_loss.mean()),
                "nonmember_loss_mean": float(nonmember_loss.mean()),
                "n_member": int(member_loss.size),
                "n_nonmember": int(nonmember_loss.size),
            },
        )
