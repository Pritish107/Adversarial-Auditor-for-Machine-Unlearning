"""Unlearning methods, behind a shared interface + a small registry.

Adding a new unlearning method = drop a module implementing `Unlearner` and register it.
"""
from __future__ import annotations

from .base import Unlearner, UnlearnContext
from .gradient_ascent import GradientAscentUnlearner
from .retrain import RetrainUnlearner

_REGISTRY: dict[str, type[Unlearner]] = {
    "gradient_ascent": GradientAscentUnlearner,
    "retrain": RetrainUnlearner,
}


def build_unlearner(method: str) -> Unlearner:
    if method not in _REGISTRY:
        raise ValueError(f"unknown unlearn method '{method}'. known: {list(_REGISTRY)}")
    return _REGISTRY[method]()


__all__ = ["Unlearner", "UnlearnContext", "build_unlearner"]
