"""The only interface between the agent's selectors and the harness.

A selector file `selectors/<name>.py` must define:

    META = {
        "method": "<interpretability method it is built on>",
        "paper": "<authors, venue, year of that method>",
        "why": "one line: why this should pick tokens the student can learn from",
        "vs_maskedkd": "one line: how it differs from MaskedKD (last-block CLS attention top-k)",
        "prior_art": "one line: closest known token-selection/pruning method and the difference",
        "parent": "none",   # or the selector this one extends (a builtin or selectors/<name>.py); a child must
                            # beat its parent at the same budget, or the added part is unproven (RESULTS_TABLE)
    }

    class Selector(SelectorBase):
        NEEDS = {"attn": [-1], "tokens": []}   # student blocks whose attention / outputs you need
        def select(self, ctx: Context) -> torch.Tensor: ...   # bool [B,196] or long [B,K]

Rules enforced by harness/validate.py and the training step:
  * no labels, no teacher, no files, no test data: ctx only holds the student side
  * mean kept tokens per batch <= ctx.budget
  * the selector may compute gradients with torch.autograd.grad, but must not call .backward()
    (any parameter .grad left behind aborts the run)
  * every FLOP spent inside select() is measured and added to the step cost
"""
from dataclasses import dataclass

import torch

from .models import StudentOut


@dataclass
class Context:
    images: torch.Tensor      # augmented student input [B,3,224,224] (same images the teacher gets)
    student: torch.nn.Module  # in train mode; read-only use
    out: StudentOut           # student forward of this step (logits keep their autograd graph)
    budget: float             # max mean kept patch tokens per image (CLS not counted)
    num_patches: int          # 196
    epoch: int
    progress: float           # fraction of training done, in [0,1)


class SelectorBase:
    NEEDS = {"attn": [], "tokens": []}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def select(self, ctx: Context) -> torch.Tensor:
        raise NotImplementedError


def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """bool [B,N] mask of the k highest scores per row."""
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)
