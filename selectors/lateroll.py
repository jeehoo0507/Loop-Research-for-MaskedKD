"""H-C2b: attention rollout over the student's late blocks only (blocks 6..11 of 12), top-K patches.

Basis (CUB, no position prior): analysis_cub_rollout_depth_1031_100143_a4b6. Over start blocks 0/3/6/8/9/10/11,
rollout from block 6 kept the masked teacher's TCKD lowest (mean over K=25, 12 and three final students 1.092 vs
1.309 for full rollout and 1.126 from block 3); it beat full rollout at K=25 and 12 on every final student and on
every epoch-25/50 snapshot, beat centre@K, and picked tokens slightly *less* central than full rollout
(centre-overlap -0.03..-0.07). Late-only (start 9..11) was worse: the last blocks alone are too peaked.
"""
from harness.selector_api import SelectorBase, topk_mask
import torch

START = 6

META = {
    "method": "attention rollout (CLS row) over the student's blocks 6..11 only",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Chefer et al. CVPR 2021 (relevance from late layers)",
    "why": "on CUB the late-half rollout keeps the masked teacher's target probability better than full rollout or "
           "last-block attention at the same K (H-C2b), so the teacher's soft label for the student changes less",
    "vs_maskedkd": "multiplies the head-averaged attention of six blocks instead of reading the last block's CLS row",
    "prior_art": "full rollout (harness baseline) multiplies all 12 blocks; early blocks mostly add diffuse mixing",
    "parent": "rollout",
}


class Selector(SelectorBase):
    NEEDS = {"attn": list(range(START, 12)), "tokens": []}

    def select(self, ctx):
        r = None
        for a in ctx.out.attn[START:]:            # list over all 12 blocks; earlier entries are None
            a = a.detach().float().mean(dim=1)
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            r = a if r is None else a @ r
        return topk_mask(r[:, 0, 1:], int(ctx.budget))
