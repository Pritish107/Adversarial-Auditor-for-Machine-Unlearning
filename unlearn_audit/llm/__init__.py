"""LLM / TOFU support: text data + answer-NLL scorer + model loading.

Everything here is an IMPLEMENTATION below the unchanged Attack interface — the text
membership scorer plugs into the same `LossMIA` / `audit()` as the classifier via the
injected `AttackContext.scorer`. No LLM training: M2b uses pre-baked checkpoints.
"""
