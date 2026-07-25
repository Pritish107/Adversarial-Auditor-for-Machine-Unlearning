"""Text guardrail — the training-free strawman, exact analog of the CIFAR guardrail.

It does NOT touch the weights. It wraps the frozen baseline Phi with a perfect forget-set
detector (the provider knows exactly which questions to suppress) and an output filter:

  * __call__(input_ids)  -> the baseline's raw logits, UNCHANGED. This is the scoring path
                            the answer-NLL scorer reads, so per-token NLL is byte-identical to
                            baseline -> MIA-AUC ~1.0 (membership trace fully intact).
  * answer accuracy       -> for flagged forget10 questions the teacher-forced argmax is
                            corrupted, so a naive "did it answer correctly?" eval reads ~0
                            (looks forgotten).

That gap — answer-accuracy ~0 but NLL untouched — is the text version of the CIFAR headline
(forget-acc 0.000 but MIA-AUC == baseline). See PLAN.md C1 for the threat-model boundary:
the guardrail is caught only because the auditor reads the model's scoring (NLL), not just the
filtered answers; a filter that also corrupted the logits would defeat loss-MIA specifically.
"""
from __future__ import annotations

import torch

from .scoring import DEFAULT_PROMPT


class TextGuardrail:
    """Wraps a frozen causal-LM. Scoring path == base; answer path deflects on forget-set."""

    def __init__(self, base, forget_questions):
        self.base = base
        self._sig = set(forget_questions)      # perfect detector: exact forget-question strings

    def __call__(self, *args, **kwargs):
        return self.base(*args, **kwargs)      # raw logits, unchanged — the auditor's path

    def eval(self):
        self.base.eval()
        return self

    def is_forget(self, question: str) -> bool:
        return question in self._sig


@torch.no_grad()
def answer_accuracy(model, dataset, tokenizer, device, prompt: str = DEFAULT_PROMPT) -> float:
    """Teacher-forced next-token argmax accuracy over the ANSWER span (naive 'did it answer?').

    Mirrors the CIFAR accuracy path: a guardrail's `is_forget` questions get their predicted
    answer tokens corrupted, collapsing accuracy while the NLL/logits path stays untouched.
    """
    deflect = getattr(model, "is_forget", None)
    correct, total = 0, 0
    for i in range(len(dataset)):
        q, a = dataset[i]
        prefix = prompt.format(q=q)
        q_len = tokenizer(prefix, return_tensors="pt").input_ids.shape[1]
        full = tokenizer(prefix + a, return_tensors="pt").input_ids.to(device)
        if full.shape[1] <= q_len:
            continue
        logits = model(full).logits[:, :-1]
        preds = logits.argmax(-1)[0]
        tgt = full[0, 1:]
        mask = torch.arange(preds.shape[0], device=device) >= (q_len - 1)
        p, t = preds[mask], tgt[mask]
        if deflect is not None and deflect(q):
            p = (p + 1) % logits.shape[-1]      # corrupt shown answer -> guaranteed wrong
        correct += (p == t).sum().item()
        total += t.numel()
    return correct / total if total else 0.0
