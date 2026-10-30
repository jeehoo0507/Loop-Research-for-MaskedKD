"""Decomposition control for suffdisc (required before promotion, AGENT_PROMPT "승격 전 분해").

suffdisc picks, per image, one of the (2r+1)^2 centre-disc shifts by the student's own keep-only sufficiency.
Here the shift is drawn uniformly at random from the same candidates, with no student forward. If suffdisc's
K=12 gain over center (2-seed 89.62/61.28 vs 88.37/54.14, screen_suffdisc_k12_s{0,1}, screen_center_k12_s{0,1})
stays, it comes from disc jitter (teacher-input diversity), not the sufficiency signal; if it goes, the signal
is the cause. Not a candidate method: an ablation whose parent is suffdisc.

sel_args: r = max shift in patches (default 2, same as suffdisc).
"""
import torch

from harness.selector_api import SelectorBase

META = {
    "method": "ablation of student-sufficiency disc placement: the disc shift is random instead of sufficiency-chosen",
    "paper": "DeYoung et al. ACL 2020 (ERASER sufficiency); Hooker et al. NeurIPS 2019 (replacement control)",
    "why": "tests whether suffdisc's gain comes from the sufficiency signal or only from moving the disc",
    "vs_maskedkd": "contiguous disc at a random shift of the centre; no attention ranking and no student signal",
    "prior_art": "random-replacement control in the style of ROAR; not a proposed method",
    "parent": "suffdisc",
}

SIDE = 14


class Selector(SelectorBase):
    NEEDS = {"attn": [], "tokens": []}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.r = int(kwargs.get("r", 2))

    def select(self, ctx):
        B, dev = ctx.images.shape[0], ctx.images.device
        k = int(ctx.budget)
        n = ctx.num_patches
        rr = torch.arange(n, device=dev)
        rows, cols = (rr // SIDE).float(), (rr % SIDE).float()
        c = (SIDE - 1) / 2
        tie = rr.float() * 1e-6
        dy = torch.randint(-self.r, self.r + 1, (B, 1), device=dev).float()
        dx = torch.randint(-self.r, self.r + 1, (B, 1), device=dev).float()
        d = (rows[None] - (c + dy)) ** 2 + (cols[None] - (c + dx)) ** 2 + tie[None]
        idx = (-d).topk(k, dim=1).indices
        return torch.zeros(B, n, dtype=torch.bool, device=dev).scatter_(1, idx, True)
