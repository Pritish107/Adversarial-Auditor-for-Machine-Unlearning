"""CLI entry point.

    python -m unlearn_audit.cli --config configs/default.yaml

Runs the full v0 auditor and prints a readable report.
"""
from __future__ import annotations

import argparse

from .pipeline import run
from .utils import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Adversarial auditor for machine unlearning (v0)")
    ap.add_argument("--config", default="configs/default.yaml", help="path to YAML config")
    ap.add_argument("--dataset", default=None, help="override dataset.name (e.g. cifar10)")
    ap.add_argument("--device", default=None, help="override device (cpu|cuda)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.device:
        cfg["device"] = args.device

    text, _ = run(cfg)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
