"""End-to-end v0 pipeline: train -> unlearn -> audit, wired through the interfaces.

Kept separate from the CLI so tests can drive the whole thing in-process on a tiny config.
"""
from __future__ import annotations

from torch.utils.data import ConcatDataset, DataLoader

from . import models, report
from .attacks import build_attack
from .attacks.base import AttackContext
from .audit.score import audit
from .data import DataBundle, build_data
from .unlearn import build_unlearner
from .unlearn.base import UnlearnContext
from .utils import resolve_device, set_seed


def _audit_model(model, data: DataBundle, cfg, device, reference=None):
    results = []
    for key in cfg["attacks"]:
        ctx = AttackContext(
            target_model=model,
            member_data=data.forget,
            nonmember_data=data.test,
            retain_data=data.retain,
            device=device,
            batch_size=cfg["dataset"]["batch_size"],
            reference_model=reference,
            retrain_fn=None,
            params={},
        )
        results.append(build_attack(key).run(ctx))
    return audit(results, aggregate=cfg["audit"]["aggregate"],
                 target_far=cfg["audit"]["target_far"])


def _summarize(label, model, data, cfg, device, reference=None) -> report.ModelSummary:
    bs = cfg["dataset"]["batch_size"]
    return report.ModelSummary(
        label=label,
        forget_eval=models.evaluate(model, data.forget, batch_size=bs, device=device),
        retain_eval=models.evaluate(model, data.retain, batch_size=bs, device=device),
        test_eval=models.evaluate(model, data.test, batch_size=bs, device=device),
        audit=_audit_model(model, data, cfg, device, reference=reference),
    )


def run(cfg: dict) -> tuple[str, list[report.ModelSummary]]:
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    data = build_data(cfg["dataset"], seed=cfg["seed"])
    bs = cfg["dataset"]["batch_size"]

    # 1. Baseline: train on the full pool (retain + forget).
    pool = DataLoader(ConcatDataset([data.retain, data.forget]), batch_size=bs, shuffle=True)
    baseline = models.build_model(
        cfg["model"]["arch"], data.in_channels, data.num_classes, data.image_size
    )
    baseline = models.train(baseline, pool, epochs=cfg["train"]["epochs"],
                            lr=cfg["train"]["lr"], device=device)

    # 2. Unlearn (honest baseline method); baseline stays untouched as the before-reference.
    method = cfg["unlearn"]["method"]
    unlearned = build_unlearner(method).unlearn(UnlearnContext(
        model=baseline, data=data, device=device, batch_size=bs,
        params=cfg["unlearn"].get(method, {}),
    ))

    summaries = [
        _summarize("baseline (pre-unlearn)", baseline, data, cfg, device),
        _summarize("unlearned", unlearned, data, cfg, device, reference=baseline),
    ]

    # 3. Optional gold retrain reference (ideal forgetting).
    if cfg.get("gold", {}).get("enabled", False):
        gold_params = dict(cfg["gold"]); gold_params["arch"] = cfg["model"]["arch"]
        gold = build_unlearner("retrain").unlearn(UnlearnContext(
            model=baseline, data=data, device=device, batch_size=bs, params=gold_params,
        ))
        summaries.append(_summarize("gold (retrain)", gold, data, cfg, device))

    return report.render(summaries), summaries
