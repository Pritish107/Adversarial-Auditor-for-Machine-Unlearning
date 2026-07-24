"""Dataset loading + the forget/retain/test split, behind one abstraction.

The split is the backbone the auditor reasons about:

  * train pool  = retain + forget   (everything the baseline model sees)
  * forget      = the subset a model later *claims* to have forgotten  (MIA "members")
  * retain      = the rest of the training pool (kept knowledge)
  * test        = held-out data the model never trained on              (MIA "non-members")

Everything is deterministic given the seed. `name` is a config knob: swapping
mnist -> cifar10 changes only this module, not the pipeline. When we reach the LLM
milestone, TOFU implements the SAME DataBundle contract (member/non-member text sets),
so nothing downstream has to change.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@contextlib.contextmanager
def _silent():
    """Suppress torchvision's download progress bar (it floods stdout on first fetch).

    Only prints are redirected; download failures still raise exceptions normally, so this
    hides the progress chatter without swallowing real errors.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


@dataclass
class DataBundle:
    """The canonical split every downstream component consumes."""

    retain: Dataset      # trained-on, kept
    forget: Dataset      # trained-on, claimed forgotten  (MIA members)
    test: Dataset        # never trained-on               (MIA non-members)
    num_classes: int
    in_channels: int
    image_size: int

    def loader(self, which: str, batch_size: int, shuffle: bool) -> DataLoader:
        ds = {"retain": self.retain, "forget": self.forget, "test": self.test}[which]
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


_SPECS = {
    "mnist": dict(cls=datasets.MNIST, num_classes=10, in_channels=1, image_size=28,
                  mean=(0.1307,), std=(0.3081,)),
    "cifar10": dict(cls=datasets.CIFAR10, num_classes=10, in_channels=3, image_size=32,
                    mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
}


def build_data(cfg: dict, seed: int) -> DataBundle:
    """Construct the forget/retain/test split from a dataset config block."""
    name = cfg["name"].lower()
    if name not in _SPECS:
        raise ValueError(f"unknown dataset '{name}'. known: {list(_SPECS)}")
    spec = _SPECS[name]

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(spec["mean"], spec["std"]),
    ])
    with _silent():
        train_full = spec["cls"](cfg["root"], train=True, download=True, transform=tfm)
        test_full = spec["cls"](cfg["root"], train=False, download=True, transform=tfm)

    g = torch.Generator().manual_seed(seed)

    # Sub-sample a fixed training pool, then carve forget out of it.
    n_train = min(cfg["num_train"], len(train_full))
    n_forget = min(cfg["forget_size"], n_train)
    train_perm = torch.randperm(len(train_full), generator=g)[:n_train]
    forget_idx = train_perm[:n_forget]
    retain_idx = train_perm[n_forget:]

    n_test = min(cfg["num_test"], len(test_full))
    test_idx = torch.randperm(len(test_full), generator=g)[:n_test]

    return DataBundle(
        retain=Subset(train_full, retain_idx.tolist()),
        forget=Subset(train_full, forget_idx.tolist()),
        test=Subset(test_full, test_idx.tolist()),
        num_classes=spec["num_classes"],
        in_channels=spec["in_channels"],
        image_size=spec["image_size"],
    )
