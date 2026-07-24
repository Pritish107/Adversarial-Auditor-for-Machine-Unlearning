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
from torch.utils.data import Dataset

from .base import Attack, AttackContext, AttackResult, MembershipScorer


def _retention_from_losses(member_loss: np.ndarray, nonmember_loss: np.ndarray) -> tuple[float, float]:
    """Membership-advantage retention in [0,1] plus the raw AUC.

    label 1 = member (forget-set), 0 = non-member; score = -loss (low loss -> member).
    retention = clip(2*(AUC-0.5), 0, 1): 0 = indistinguishable (good forgetting).
    """
    labels = np.concatenate([np.ones_like(member_loss), np.zeros_like(nonmember_loss)])
    scores = -np.concatenate([member_loss, nonmember_loss])
    auc = float(roc_auc_score(labels, scores))
    retention = float(np.clip(2.0 * (auc - 0.5), 0.0, 1.0))
    return retention, auc


class LossMIA(Attack):
    name = "loss_mia"

    def run(self, ctx: AttackContext) -> AttackResult:
        member_loss = np.asarray(
            ctx.scorer(ctx.target_model, ctx.member_data, device=ctx.device, batch_size=ctx.batch_size))
        nonmember_loss = np.asarray(
            ctx.scorer(ctx.target_model, ctx.nonmember_data, device=ctx.device, batch_size=ctx.batch_size))
        retention, auc = _retention_from_losses(member_loss, nonmember_loss)

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


def bootstrap_gold_null(gold_model, member_data: Dataset, nonmember_data: Dataset, *,
                        scorer: MembershipScorer, device, batch_size: int, n_boot: int,
                        seed: int = 0) -> list[float]:
    """Null distribution of the loss-MIA retention score under 'truly forgotten'.

    Uses the gold retrain (which never saw the forget-set) and resamples member/non-member
    losses with replacement. Uses the SAME injected `scorer` as the attack, so classifier and
    LLM nulls are built by identical logic. Captures SAMPLING variance only — not model/seed
    variance — so the resulting FAR is optimistic (a lower bound). See PLAN.md calibration caveat.
    """
    m = np.asarray(scorer(gold_model, member_data, device=device, batch_size=batch_size))
    nz = np.asarray(scorer(gold_model, nonmember_data, device=device, batch_size=batch_size))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_boot):
        mb = m[rng.integers(0, len(m), len(m))]
        nb = nz[rng.integers(0, len(nz), len(nz))]
        null.append(_retention_from_losses(mb, nb)[0])
    return null
