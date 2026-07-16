"""End-to-end v0 pipeline: train -> {unlearning methods} -> audit, through the interfaces.

Produces one ground-truth-labeled case per method, plus a control baseline and the gold
retrain reference. The gold model also seeds the calibration null (bootstrapped), so every
case is judged against the same threshold / false-alarm rate.

Kept separate from the CLI so tests can drive the whole thing in-process on a tiny config.
"""
from __future__ import annotations

from torch.utils.data import ConcatDataset, DataLoader

from . import models, report
from .attacks import build_attack
from .attacks.base import AttackContext
from .attacks.membership_inference import bootstrap_gold_null
from .audit.score import audit
from .data import DataBundle, build_data
from .unlearn import build_unlearner
from .unlearn.base import UnlearnContext
from .utils import resolve_device, set_seed

# What each case actually is — the ground-truth labels shown in the report.
_GROUND_TRUTH = {
    "baseline (pre-unlearn)": "control: never unlearned (should look retained)",
    "gradient_ascent": "low-quality unlearn (honest but crude)",
    "guardrail": "fake: output filter, weights intact (should be caught)",
    "gold (retrain)": "reference: truly forgotten (retrain without forget-set)",
}


def _audit_model(model, data: DataBundle, cfg, device, null_scores, reference=None):
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
                 target_far=cfg["calibration"]["target_far"], null_scores=null_scores)


def _summarize(label, model, data, cfg, device, null_scores, reference=None) -> report.ModelSummary:
    bs = cfg["dataset"]["batch_size"]
    return report.ModelSummary(
        label=label,
        ground_truth=_GROUND_TRUTH.get(label, "unlabeled"),
        forget_eval=models.evaluate(model, data.forget, batch_size=bs, device=device),
        retain_eval=models.evaluate(model, data.retain, batch_size=bs, device=device),
        test_eval=models.evaluate(model, data.test, batch_size=bs, device=device),
        audit=_audit_model(model, data, cfg, device, null_scores, reference=reference),
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

    # 2. Gold retrain (retain-only) — the truly-forgotten reference AND the calibration null.
    gold_params = dict(cfg["gold"]); gold_params["arch"] = cfg["model"]["arch"]
    gold = build_unlearner("retrain").unlearn(UnlearnContext(
        model=baseline, data=data, device=device, batch_size=bs, params=gold_params,
    ))
    null_scores = bootstrap_gold_null(
        gold, data.forget, data.test, device=device, batch_size=bs,
        n_boot=cfg["calibration"]["n_boot"], seed=cfg["seed"],
    )

    # 3. Baseline control + each unlearning method + gold, all judged against the same null.
    summaries = [_summarize("baseline (pre-unlearn)", baseline, data, cfg, device, null_scores)]
    for method in cfg["methods"]:
        model = build_unlearner(method).unlearn(UnlearnContext(
            model=baseline, data=data, device=device, batch_size=bs,
            params=cfg["unlearn"].get(method, {}),
        ))
        summaries.append(_summarize(method, model, data, cfg, device, null_scores, reference=baseline))
    summaries.append(_summarize("gold (retrain)", gold, data, cfg, device, null_scores))

    return report.render(summaries), summaries
