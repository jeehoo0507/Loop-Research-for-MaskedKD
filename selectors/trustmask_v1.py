"""TrustMask v1 (H-A27 / H-A28, QUEUE 2026-11-03 17:25): the cached teacher map moves from a token filter to a
per-image trust weight.

v0 (trustfilt) used the teacher map as an allowed region for the student's picks; H-A26 showed this makes the masked
teacher less faithful than student-only picking at every epoch. v1 keeps the student's late-block rollout picks
(lateroll, blocks 6..11) and uses the teacher-map mass covered by those K tokens to weight each image's KD term
(weight = coverage / batch mean, mean 1): an image whose kept tokens hold little of the teacher's own evidence gets a
masked-teacher target that H-A27 found less faithful, so it is trusted less. No constants beyond lateroll's.
"""
import torch

from harness.selector_api import SelectorBase, topk_mask

START = 6

META = {
    "method": "student late-block attention rollout picks the K teacher tokens; the cached teacher rollout's mass "
              "inside them weights the image's KD term (mean 1)",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS "
             "2019 (keep-only faithfulness)",
    "why": "H-A26: the teacher-map filter lowers masked-teacher fidelity; H-A27: teacher-map coverage of the kept "
           "tokens predicts per image how faithful the masked teacher is, so it can down-weight unfaithful targets",
    "vs_maskedkd": "late-block rollout instead of last-block CLS attention, plus a per-image KD weight from the "
                   "cached teacher map (teacher cost unchanged)",
    "prior_art": "ent_weight / EGAD weight KD by student entropy; v1 weights by the teacher's attribution covered",
    "parent": "trustfilt",
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
        return keep, cov / cov.mean().clamp_min(1e-12)
