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
        # optional, per-image budgets (QUEUE 2026-11-01):
        #   select may return (keep, kd_weight [B]) and a bool keep with a different count per image, 0 included
        #   def observe(self, fb: Feedback): ...   # after the teacher ran: each image's KD loss, for the next epochs

Rules enforced by harness/validate.py and the training step:
  * no labels, no teacher, no files, no test data: ctx only holds the student side, with one exception:
    a selector whose META has "uses_teacher_cache": True also gets ctx.teacher_map [B,196], the teacher's attention
    rollout of the whole un-augmented image, computed once per training image before training and moved with
    this batch's crop/flip/Mixup (harness/teacher_cache.py). Its build cost is charged to every step.
  * kept tokens: a bool keep may hold a different number per image (0 = no teacher for that image, CE only).
    Mean kept per batch <= 2 x ctx.budget, mean over the whole run <= ctx.budget (a run that ends above it, e.g.
    by timeout, is recorded as over_budget). FLOPs are measured in windows spread over the run.
  * kd_weight (optional second return value): float [B] >= 0, mean <= 1 over the images that got the teacher;
    it can move KD signal between images, never add to it
  * observe(fb) (optional) gets, after the teacher forward, fb.kd [B] = each image's KD loss (NaN where no teacher
    ran) and fb.kept [B], with fb.sample_idx [B] (training-set rows) to keep a per-image learning state. It costs
    nothing extra in teacher compute; its own FLOPs count as selector FLOPs. State held in the selector is not
    saved in the checkpoint: after a resume it starts empty.
  * the selector may compute gradients with torch.autograd.grad, but must not call .backward()
    (any parameter .grad left behind aborts the run)
  * every FLOP spent inside select() is measured and added to the step cost
"""
from dataclasses import dataclass
from typing import Optional

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
    teacher_map: Optional[torch.Tensor] = None   # [B,196] cached teacher rollout (sums to ~1 per image), or None
    sample_idx: Optional[torch.Tensor] = None    # [B] training-set row of each image (Mixup partner not given)


@dataclass
class Feedback:
    """What observe() gets after the teacher forward of a step."""
    sample_idx: Optional[torch.Tensor]   # [B] training-set rows, as in Context
    kept: torch.Tensor                   # [B] tokens the teacher got per image
    kd: torch.Tensor                     # [B] KD loss per image (TAU^2 KL), NaN where the image got no teacher
    epoch: int
    progress: float


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


def allocate(weights: torch.Tensor, budget: float, kmin: int = 0, kmax: int = 196) -> torch.Tensor:
    """Integer tokens per image [B] in proportion to weights [B] (>= 0), each in [kmin, kmax], summing to at most
    budget * B. For per-image budgets: pass any score of how much each image should get."""
    B = weights.shape[0]
    total = int(budget * B)
    w = weights.detach().float().clamp(min=0) + 1e-12
    k = torch.full_like(w, float(kmin))
    free = torch.ones_like(w, dtype=torch.bool)
    for _ in range(B):   # water-filling: images that hit kmax hand the rest back to the others
        left = total - k[~free].sum() - kmin * free.sum()
        k[free] = kmin + w[free] / w[free].sum() * max(float(left), 0.0)
        over = free & (k > kmax)
        if not over.any():
            break
        k[over] = float(kmax)
        free &= ~over
    out = k.floor().long()
    rest = int(total - out.sum().item())
    if rest > 0:   # largest remainders first, never above kmax
        frac = (k - out.float()).masked_fill(out >= kmax, -1.0)
        out[frac.topk(min(rest, B)).indices] += 1
    return out.clamp(kmin, kmax)


def keep_by_counts(scores: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """bool [B,N]: the counts[i] highest scores of row i (counts may differ per row, 0 included)."""
    rank = scores.argsort(dim=1, descending=True).argsort(dim=1)
    return rank < counts.to(scores.device).view(-1, 1)
