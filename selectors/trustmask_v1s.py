"""Control for TrustMask v1 (H-A36, QUEUE 2026-11-03 23:15 C): v1's tokens and v1's KD weights, but the weights
are shuffled within the batch so each image gets another image's weight. Same weight distribution, no link to the
image's own teacher-map coverage. v1 must beat this for "which image to trust less" to be the cause rather than
the mere jitter of KD weights.
"""
import torch

from harness.selector_api import SelectorBase, topk_mask

START = 6

META = {
    "method": "v1 (late-block attention rollout picks, cached teacher-map coverage as KD weight) with the weights "
              "randomly permuted across the batch",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Son et al. ECCV 2024 (MaskedKD)",
    "why": "control: separates the per-image trust signal in v1 from the regularising effect of uneven KD weights",
    "vs_maskedkd": "late-block rollout instead of last-block CLS attention, plus a per-image KD weight drawn from "
                   "the batch's coverage weights at random (teacher cost unchanged)",
    "prior_art": "shuffled-label / permutation controls; v1 (trustmask_v1) is the unshuffled version",
    "parent": "trustmask_v1",
    "uses_teacher_cache": True,
}


class Selector(SelectorBase):
    NEEDS = {"attn": list(range(START, 12)), "tokens": []}

    def select(self, ctx):
        r = None
        for a in ctx.out.attn[START:]:
            a = a.detach().float().mean(dim=1)
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            r = a if r is None else a @ r
        keep = topk_mask(r[:, 0, 1:], int(ctx.budget))
        cov = (ctx.teacher_map.float() * keep).sum(1)
        w = cov / cov.mean().clamp_min(1e-12)
        return keep, w[torch.randperm(w.shape[0], device=w.device)]
