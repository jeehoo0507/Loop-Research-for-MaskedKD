"""TrustMask v1m (H-A34, QUEUE 2026-11-03 23:15 A): v1's per-image trust weight plugged into MaskedKD's own picks.

Tokens are MaskedKD's unchanged (student last block, head-mean CLS attention, top-K). The KD weight is v1's: the
cached teacher map's mass inside the kept tokens, divided by its batch mean (mean 1). The question is whether the
weight improves MaskedKD on its own, apart from v1's late-block rollout picks. No constants.
"""
from harness.selector_api import SelectorBase, topk_mask

META = {
    "method": "MaskedKD picks (last-block CLS attention top-K); the cached teacher rollout's mass inside them "
              "weights the image's KD term (mean 1)",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (attention rollout, cached teacher map)",
    "why": "v1 beat MaskedKD on both CUB and Aircraft on average but changed both the picks and the KD weight; this "
           "isolates the weight as a plug-in part on top of MaskedKD",
    "vs_maskedkd": "same tokens and teacher cost; adds a per-image KD weight from the cached teacher map",
    "prior_art": "ent_weight / EGAD weight KD by student entropy; this weights by the teacher's attribution covered",
    "parent": "maskedkd",
    "uses_teacher_cache": True,
}


class Selector(SelectorBase):
    NEEDS = {"attn": [-1], "tokens": []}

    def select(self, ctx):
        scores = ctx.out.attn[-1].detach().float().mean(dim=1)[:, 0, 1:]
        keep = topk_mask(scores, int(ctx.budget))
        cov = (ctx.teacher_map.float() * keep).sum(1)
        return keep, cov / cov.mean().clamp_min(1e-12)
