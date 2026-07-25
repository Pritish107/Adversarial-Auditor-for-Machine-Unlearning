"""Load TOFU-finetuned Phi-1.5 checkpoints + the correct tokenizer.

M2a gotcha baked in: TOFU finetune repos ship NO tokenizer files, and
`AutoTokenizer.from_pretrained(model_repo)` silently returns a broken vocab_size=0 tokenizer.
So the tokenizer ALWAYS comes from the base model, never the finetune repo.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_TOKENIZER = "microsoft/phi-1_5"


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    assert tok.vocab_size > 0, f"broken tokenizer (vocab={tok.vocab_size}) — wrong repo?"
    return tok


def load_model(repo: str, device: torch.device):
    try:                                              # transformers 5.x renamed torch_dtype->dtype
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16)
    return model.to(device).eval()
