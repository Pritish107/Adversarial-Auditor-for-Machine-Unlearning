"""Adversarial auditor for machine unlearning.

The auditor tries to *disprove* that a model forgot its forget-set, and turns the
evidence into a calibrated forgetting score. v0 target: image classifier + one
membership-inference attack. The interfaces here are designed to survive the move
to LLMs (TOFU) without a core rewrite.
"""

__version__ = "0.0.1"
