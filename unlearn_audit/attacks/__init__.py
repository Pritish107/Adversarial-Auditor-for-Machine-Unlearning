"""Attack battery, behind a shared interface + registry.

Adding attack #2 / #3 = drop a module implementing `Attack` and register its key here.
Nothing else in the pipeline changes.
"""
from __future__ import annotations

from .base import Attack, AttackContext, AttackResult
from .membership_inference import LossMIA

_REGISTRY: dict[str, type[Attack]] = {
    "loss_mia": LossMIA,
}


def build_attack(key: str) -> Attack:
    if key not in _REGISTRY:
        raise ValueError(f"unknown attack '{key}'. known: {list(_REGISTRY)}")
    return _REGISTRY[key]()


def available_attacks() -> list[str]:
    return list(_REGISTRY)


__all__ = ["Attack", "AttackContext", "AttackResult", "build_attack", "available_attacks"]
