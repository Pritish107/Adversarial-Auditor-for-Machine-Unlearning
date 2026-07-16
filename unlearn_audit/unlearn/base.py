"""The Unlearner interface.

An unlearner takes a trained model and the data split, and returns a *new* model that
claims to have forgotten the forget-set. Implementations must not mutate the input model
(the pipeline keeps the pre-unlearning model as the before/after reference).
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..data import DataBundle


@dataclass
class UnlearnContext:
    """Everything an unlearning method may need from the harness."""

    model: nn.Module          # the trained baseline (to be treated as read-only)
    data: DataBundle
    device: torch.device
    batch_size: int
    params: dict              # method-specific hyperparameters


class Unlearner(ABC):
    name: str = "unlearner"

    @abstractmethod
    def unlearn(self, ctx: UnlearnContext) -> nn.Module:
        """Return a new model claiming to have forgotten ctx.data.forget."""

    @staticmethod
    def _clone(model: nn.Module) -> nn.Module:
        """Deep-copy so we never mutate the caller's baseline model."""
        return copy.deepcopy(model)
