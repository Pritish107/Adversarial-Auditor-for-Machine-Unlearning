"""Answer-NLL membership scorer for the text setting.

Implements the `MembershipScorer` contract: (model, dataset, *, device, batch_size) ->
per-example loss (lower = more member-like). Here the loss is per-TOKEN mean NLL over the
ANSWER span (question as unscored context) — length-normalized and identical-span for both
member and non-member sets, so it is honest membership signal, not a length artifact
(validated in the M2a spike: forget10 0.090 vs holdout10 2.123, AUC 1.0).

Built as a factory that captures the tokenizer, so the callable matches the scorer signature
exactly and drops into `AttackContext.scorer` / `bootstrap_gold_null` unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_PROMPT = "Question: {q}\nAnswer: "     # identical template for members + non-members


def make_answer_nll_scorer(tokenizer, prompt: str = DEFAULT_PROMPT):
    @torch.no_grad()
    def scorer(model, dataset, *, device, batch_size):
        model.eval()
        out = []
        for i in range(len(dataset)):
            q, a = dataset[i]
            prefix = prompt.format(q=q)
            q_len = tokenizer(prefix, return_tensors="pt").input_ids.shape[1]
            full = tokenizer(prefix + a, return_tensors="pt").input_ids.to(device)
            if full.shape[1] <= q_len:
                out.append(np.nan)
                continue
            logits = model(full).logits[:, :-1].float()     # predict token i+1 from <=i
            tgt = full[:, 1:]
            tok_nll = -F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
            ans_mask = torch.arange(tok_nll.shape[0], device=device) >= (q_len - 1)
            out.append(tok_nll[ans_mask].mean().item())      # length-normalized, answer-only
        return np.array(out)

    return scorer
