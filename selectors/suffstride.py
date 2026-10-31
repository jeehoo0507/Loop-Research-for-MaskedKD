"""H-W34: strided student-sufficiency disc, a cost-reduced suffdisc. Same search range as suffdisc (disc shifts of up
to `r` patches from the centre), but only every `stride`-th shift is tried: r=2, stride=2 gives 9 candidate discs
(shifts in {-2, 0, 2}^2) instead of 25. Nothing is added; the question is whether suffdisc's same-K gain survives
at a selector cost low enough to compare with center/maskedkd at the same total GFLOPs (H-W32 found center@49 at
16.46 GFLOPs/img ahead of suffdisc@12 at 15.02). r=1 (9 candidates, shifts {-1,0,1}^2, H-W24) did not separate
from center, so this keeps the reach of r=2 and drops the in-between positions.

Basis and selection rule: as suffdisc (smallest KL(student full || student on the disc only)).

Cost: ((2r)//stride + 1)^2 student forwards on K tokens per image, no grad.

sel_args: r = max shift in patches (default 2), stride = shift step (default 2).
"""
import torch

from harness.selector_api import SelectorBase

META = {
    "method": "keep-only sufficiency faithfulness of a token subset (ERASER sufficiency / RISE insertion), "
              "evaluated on the student on a strided set of disc positions to place the teacher's input window",
    "paper": "DeYoung et al. ACL 2020 (ERASER); Petsiuk, Das & Saenko BMVC 2018 (RISE); Hooker et al. NeurIPS 2019 "
             "(ROAR)",
    "why": "suffdisc's gain is large at the same K but its 25-candidate search costs ~1.5x maskedkd; a coarser grid "
           "over the same range tests whether the gain needs the full search",
    "vs_maskedkd": "no attention ranking: a contiguous disc placed by the student's output sufficiency, not the "
                   "last block's scattered CLS-attention top-K",
    "prior_art": "suffdisc (same rule, 25 candidates); center (fixed disc, no signal); RISE/ERASER score a given "
                 "mask for explanation only",
    "parent": "suffdisc",
}

SIDE = 14


def _masked_logits(model, x, idx):
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    cls, patches = h[:, :1], h[:, 1:]
    D = patches.shape[-1]
    patches = torch.gather(patches, 1, idx.unsqueeze(-1).expand(-1, -1, D))
    h = torch.cat([cls, patches], dim=1)
    h = model.norm_pre(h)
    h = model.blocks(h)
    h = model.norm(h)
    return model.forward_head(h)


class Selector(SelectorBase):
    NEEDS = {"attn": [], "tokens": []}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.r = int(kwargs.get("r", 2))
        self.stride = int(kwargs.get("stride", 2))

    def select(self, ctx):
        x = ctx.images
        B, dev = x.shape[0], x.device
        k = int(ctx.budget)
        n = ctx.num_patches
        rr = torch.arange(n, device=dev)
        rows, cols = (rr // SIDE).float(), (rr % SIDE).float()
        c = (SIDE - 1) / 2
        tie = rr.float() * 1e-6
        shifts = range(-self.r, self.r + 1, self.stride)
        with torch.no_grad():
            full = ctx.out.logits.detach().float().log_softmax(-1)
            best_kl = torch.full((B,), float("inf"), device=dev)
            best_idx = None
            for dy in shifts:
                for dx in shifts:
                    d = (rows - (c + dy)) ** 2 + (cols - (c + dx)) ** 2 + tie
                    idx = (-d).topk(k).indices.sort().values.expand(B, -1)
                    lp = _masked_logits(ctx.student, x, idx).float().log_softmax(-1)
                    kl = (full.exp() * (full - lp)).sum(-1)
                    better = kl < best_kl
                    best_kl = torch.where(better, kl, best_kl)
                    best_idx = idx.clone() if best_idx is None else torch.where(better[:, None], idx, best_idx)
        return torch.zeros(B, n, dtype=torch.bool, device=dev).scatter_(1, best_idx, True)
