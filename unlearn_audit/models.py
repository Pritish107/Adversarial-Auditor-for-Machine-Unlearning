"""Small CNN + train/eval loops.

Kept intentionally tiny and parametric (in_channels / image_size / num_classes) so the
same code trains MNIST or CIFAR-10 with only a config change.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, image_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # two 2x2 pools -> spatial dim shrinks by 4
        feat = 32 * (image_size // 4) * (image_size // 4)
        self.fc1 = nn.Linear(feat, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_model(arch: str, in_channels: int, num_classes: int, image_size: int) -> nn.Module:
    if arch != "small_cnn":
        raise ValueError(f"unknown arch '{arch}'")
    return SmallCNN(in_channels, num_classes, image_size)


def train(model: nn.Module, loader: DataLoader, *, epochs: int, lr: float,
          device: torch.device) -> nn.Module:
    """Standard supervised training (gradient descent on cross-entropy)."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    return model


@dataclass
class EvalResult:
    accuracy: float
    loss: float


@torch.no_grad()
def evaluate(model: nn.Module, dataset: Dataset, *, batch_size: int,
             device: torch.device) -> EvalResult:
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return EvalResult(accuracy=correct / total, loss=loss_sum / total)


@torch.no_grad()
def per_sample_loss(model: nn.Module, dataset: Dataset, *, batch_size: int,
                    device: torch.device) -> torch.Tensor:
    """Per-example cross-entropy — the raw signal loss-based MIA is built on."""
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        losses.append(F.cross_entropy(model(x), y, reduction="none").cpu())
    return torch.cat(losses)
