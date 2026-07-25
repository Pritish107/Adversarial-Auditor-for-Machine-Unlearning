"""TOFU text data as plain (question, answer) Datasets the scorer consumes opaquely.

The forget/retain split is expressed as paired TOFU *configs* (forget10 / retain90), and
`holdout*` supplies same-distribution never-trained non-members (the text analog of the
CIFAR test set). See PLAN.md M2a facts.
"""
from __future__ import annotations

from torch.utils.data import Dataset

TOFU_DATA = "locuslab/TOFU"          # DATA repo — distinct from the model repo (M2a gotcha)


class TextQADataset(Dataset):
    """Yields (question, answer) string pairs. Attacks/scorers read it without knowing more."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[str, str]:
        return self.pairs[i]


def load_tofu_qa(config: str, n: int | None = None) -> TextQADataset:
    from datasets import load_dataset
    ds = load_dataset(TOFU_DATA, config, split="train")
    m = len(ds) if n is None else min(n, len(ds))
    return TextQADataset([(ds[i]["question"], ds[i]["answer"]) for i in range(m)])
