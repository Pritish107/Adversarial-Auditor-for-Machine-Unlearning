"""End-to-end v0 pipeline: train -> {unlearning methods} -> audit, through the interfaces.

Two entry points share the same builders:
  * `run(cfg)`       — single-seed run with per-seed calibration; used by the CLI.
  * `run_seed(cfg, seed)` — returns RAW per-case metrics + this seed's gold null, WITHOUT
                            calibration, so the multi-seed harness can pool nulls across seeds.
"""
from __future__ import annotations

from dataclasses import dataclass

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

# What each case actually is — the ground-truth labels shown in reports.
_GROUND_TRUTH = {
    "baseline (pre-unlearn)": "control: never unlearned (should look retained)",
    "gradient_ascent": "low-quality unlearn (honest but crude)",
    "guardrail": "fake: output filter, weights intact (should be caught)",
    "gold (retrain)": "reference: truly forgotten (retrain without forget-set)",
}


# ---- reusable builders -------------------------------------------------------

def _build_baseline(cfg, data, device):
    pool = DataLoader(ConcatDataset([data.retain, data.forget]),
                      batch_size=cfg["dataset"]["batch_size"], shuffle=True)
    m = models.build_model(cfg["model"]["arch"], data.in_channels, data.num_classes, data.image_size)
    return models.train(m, pool, epochs=cfg["train"]["epochs"], lr=cfg["train"]["lr"], device=device)


def _build_gold(cfg, baseline, data, device):
    params = dict(cfg["gold"]); params["arch"] = cfg["model"]["arch"]
    return build_unlearner("retrain").unlearn(UnlearnContext(
        model=baseline, data=data, device=device,
        batch_size=cfg["dataset"]["batch_size"], params=params))


def _build_method(method, cfg, baseline, data, device):
    return build_unlearner(method).unlearn(UnlearnContext(
        model=baseline, data=data, device=device,
        batch_size=cfg["dataset"]["batch_size"], params=cfg["unlearn"].get(method, {})))


def _attack_results(model, data, cfg, device, reference=None):
    results = []
    for key in cfg["attacks"]:
        ctx = AttackContext(
            target_model=model, member_data=data.forget, nonmember_data=data.test,
            retain_data=data.retain, device=device, batch_size=cfg["dataset"]["batch_size"],
            scorer=models.per_sample_loss,          # classifier scorer: per-sample cross-entropy
            reference_model=reference, retrain_fn=None, params={})
        results.append(build_attack(key).run(ctx))
    return results


# ---- per-case metric record (raw, uncalibrated) ------------------------------

@dataclass
class SeedCase:
    retention: float
    auc: float
    forget_acc: float
    retain_acc: float
    test_acc: float


@dataclass
class SeedResult:
    seed: int
    cases: dict            # label -> SeedCase
    gold_null: list        # this seed's gold bootstrap null scores


def _case_metrics(model, data, cfg, device, reference=None) -> SeedCase:
    bs = cfg["dataset"]["batch_size"]
    res = _attack_results(model, data, cfg, device, reference=reference)
    primary = res[0]
    return SeedCase(
        retention=primary.retention_score,
        auc=float(primary.detail.get("auc", float("nan"))),
        forget_acc=models.evaluate(model, data.forget, batch_size=bs, device=device).accuracy,
        retain_acc=models.evaluate(model, data.retain, batch_size=bs, device=device).accuracy,
        test_acc=models.evaluate(model, data.test, batch_size=bs, device=device).accuracy,
    )


def run_seed(cfg: dict, seed: int) -> SeedResult:
    """Raw per-case metrics + gold null for one seed (no calibration)."""
    set_seed(seed)
    device = resolve_device(cfg["device"])
    data = build_data(cfg["dataset"], seed=seed)

    baseline = _build_baseline(cfg, data, device)
    gold = _build_gold(cfg, baseline, data, device)
    gold_null = bootstrap_gold_null(
        gold, data.forget, data.test, scorer=models.per_sample_loss, device=device,
        batch_size=cfg["dataset"]["batch_size"], n_boot=cfg["calibration"]["n_boot"], seed=seed)

    cases = {"baseline (pre-unlearn)": _case_metrics(baseline, data, cfg, device)}
    for method in cfg["methods"]:
        model = _build_method(method, cfg, baseline, data, device)
        cases[method] = _case_metrics(model, data, cfg, device, reference=baseline)
    cases["gold (retrain)"] = _case_metrics(gold, data, cfg, device)

    return SeedResult(seed=seed, cases=cases, gold_null=gold_null)


# ---- single-seed run (CLI) ---------------------------------------------------

def _summarize(label, model, data, cfg, device, null_scores, reference=None) -> report.ModelSummary:
    bs = cfg["dataset"]["batch_size"]
    res = _attack_results(model, data, cfg, device, reference=reference)
    return report.ModelSummary(
        label=label,
        ground_truth=_GROUND_TRUTH.get(label, "unlabeled"),
        forget_eval=models.evaluate(model, data.forget, batch_size=bs, device=device),
        retain_eval=models.evaluate(model, data.retain, batch_size=bs, device=device),
        test_eval=models.evaluate(model, data.test, batch_size=bs, device=device),
        audit=audit(res, aggregate=cfg["audit"]["aggregate"],
                    target_far=cfg["calibration"]["target_far"], null_scores=null_scores),
    )


def run(cfg: dict) -> tuple[str, list[report.ModelSummary]]:
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    data = build_data(cfg["dataset"], seed=cfg["seed"])

    baseline = _build_baseline(cfg, data, device)
    gold = _build_gold(cfg, baseline, data, device)
    null_scores = bootstrap_gold_null(
        gold, data.forget, data.test, scorer=models.per_sample_loss, device=device,
        batch_size=cfg["dataset"]["batch_size"], n_boot=cfg["calibration"]["n_boot"], seed=cfg["seed"])

    summaries = [_summarize("baseline (pre-unlearn)", baseline, data, cfg, device, null_scores)]
    for method in cfg["methods"]:
        model = _build_method(method, cfg, baseline, data, device)
        summaries.append(_summarize(method, model, data, cfg, device, null_scores, reference=baseline))
    summaries.append(_summarize("gold (retrain)", gold, data, cfg, device, null_scores))

    return report.render(summaries), summaries
