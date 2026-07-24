"""Multi-seed hardening harness.

Runs the full audit across N seeds and reports statistics, not single numbers:

  * per-case MIA-AUC and retention as mean +/- std across seeds;
  * the HEADLINE detection result: the guardrail-minus-gold MIA-AUC gap as a distribution
    across seeds (this is the number that belongs in the abstract, with error bars);
  * a SEED-AWARE calibration: gold models from ALL seeds are pooled into one null, and the
    threshold + FAR are computed against that pooled null (this is the C2 fix);
  * ranking stability: in how many seeds the ordering gold < gradient_ascent < guardrail
    (by retention) is preserved.

    python -m unlearn_audit.multiseed --config configs/default.yaml
"""
from __future__ import annotations

import argparse

import numpy as np

from .audit.calibration import Calibrator
from .pipeline import run_seed
from .utils import load_config

_ORDER = ["gold (retrain)", "gradient_ascent", "guardrail"]


def _mean_std(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def run_multiseed(cfg: dict, seeds: list[int]) -> dict:
    results = [run_seed(cfg, s) for s in seeds]
    labels = list(results[0].cases.keys())

    # Pool every seed's gold null -> seed-aware null distribution (C2 fix).
    pooled_null = [v for r in results for v in r.gold_null]
    calib = Calibrator(target_far=cfg["calibration"]["target_far"]).calibrate(0.0, pooled_null)

    # Per-case aggregates + how many seeds each case is flagged against the pooled threshold.
    per_case = {}
    for lbl in labels:
        aucs = [r.cases[lbl].auc for r in results]
        rets = [r.cases[lbl].retention for r in results]
        f_acc = [r.cases[lbl].forget_acc for r in results]
        detected = sum(1 for r in results if r.cases[lbl].retention >= calib.threshold)
        per_case[lbl] = dict(
            auc=_mean_std(aucs), retention=_mean_std(rets),
            forget_acc=_mean_std(f_acc), detected=detected,
        )

    # Headline: guardrail-minus-gold AUC gap distribution.
    gaps = [r.cases["guardrail"].auc - r.cases["gold (retrain)"].auc for r in results]

    # Ranking stability: gold < gradient_ascent < guardrail by retention.
    def ordered(r) -> bool:
        g, a, u = (r.cases[k].retention for k in _ORDER)
        return g < a < u
    ordering_preserved = sum(1 for r in results if ordered(r))

    return dict(
        seeds=seeds, labels=labels, per_case=per_case, calib=calib,
        pooled_null_n=len(pooled_null), gaps=gaps,
        ordering_preserved=ordering_preserved, n=len(seeds),
    )


def render_multiseed(cfg: dict, out: dict) -> str:
    L = []
    L.append("=" * 74)
    L.append(f"MULTI-SEED AUDIT - {cfg['dataset']['name']}, seeds {out['seeds']}, loss-MIA")
    L.append("=" * 74)
    c = out["calib"]
    far = "n/a" if c.false_alarm_rate is None else f"{c.false_alarm_rate:.3f}"
    L.append(f"seed-aware calibration: pooled gold null n={out['pooled_null_n']} "
             f"({out['n']} gold models)  threshold={c.threshold:.3f}  FAR={far}")
    L.append("")
    L.append(f"per-case across {out['n']} seeds (mean +/- std):")
    L.append(f"  {'case':<24}{'MIA-AUC':<18}{'retention':<18}{'forget-acc':<14}detected")
    for lbl in out["labels"]:
        p = out["per_case"][lbl]
        L.append(f"  {lbl:<24}"
                 f"{p['auc'][0]:.3f} +/- {p['auc'][1]:.3f}   "
                 f"{p['retention'][0]:.3f} +/- {p['retention'][1]:.3f}   "
                 f"{p['forget_acc'][0]:.3f}       "
                 f"{p['detected']}/{out['n']}")
    L.append("")
    gm, gs = _mean_std(out["gaps"])
    per_seed = ", ".join(f"{g:.3f}" for g in out["gaps"])
    L.append("-" * 74)
    L.append("HEADLINE - guardrail minus gold MIA-AUC gap (the detection result):")
    L.append(f"    {gm:.3f} +/- {gs:.3f}   (per-seed: {per_seed})")
    L.append(f"ranking gold < gradient_ascent < guardrail (retention): "
             f"preserved in {out['ordering_preserved']}/{out['n']} seeds")
    L.append("-" * 74)
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-seed audit hardening harness")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", default=None, help="comma-separated override, e.g. 0,1,2,3,4")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else cfg.get("multiseed", {}).get("seeds", [0, 1, 2, 3, 4]))

    out = run_multiseed(cfg, seeds)
    print(render_multiseed(cfg, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
