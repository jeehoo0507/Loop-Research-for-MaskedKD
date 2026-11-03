"""Control for TrustMask v2 (H-A31): the same graded per-image token counts (batch thirds 2/3, 1, 4/3 of K) and the
same v1 token picks and KD weight, but the thirds are drawn at random instead of from the teacher map's
concentration. v2 must beat this for the concentration signal, not the mere spread of counts, to be the cause.
"""
import torch

from harness.selector_api import SelectorBase, allocate, keep_by_counts

START = 6
RATIOS = (2.0 / 3.0, 1.0, 4.0 / 3.0)

META = {
    "method": "v1 (late-block attention rollout picks, cached teacher-map coverage as KD weight) with graded "
              "per-image token counts assigned at random (batch thirds 2/3, 1, 4/3 of K)",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Son et al. ECCV 2024 (MaskedKD)",
    "why": "control: separates the effect of the teacher-map concentration signal in v2 from the effect of giving "
           "images different token counts at all",
    "vs_maskedkd": "late-block rollout instead of last-block CLS attention, a per-image KD weight from the cached "
                   "teacher map, random per-image token counts (same mean teacher cost)",
    "prior_art": "rand_alloc (harness) draws continuous random counts over MaskedKD picks; this keeps v2's groups "
                 "and v1's picks",
    "parent": "trustmask_v1",
    "uses_teacher_cache": True,
}


def _groups(score):
    """[B] ratio per image: lowest third of score -> 2/3, middle -> 1, highest -> 4/3."""
    B = score.shape[0]
    rank = score.argsort().argsort().float()
    g = (rank * 3 / B).floor().long().clamp(0, 2)
    return torch.tensor(RATIOS, device=score.device)[g]


class Selector(SelectorBase):
    NEEDS = {"attn": list(range(START, 12)), "tokens": []}

    def select(self, ctx):
        r = None
        for a in ctx.out.attn[START:]:
            a = a.detach().float().mean(dim=1)
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            r = a if r is None else a @ r
        order = torch.rand(ctx.images.shape[0], device=ctx.images.device)
        counts = allocate(_groups(order), ctx.budget, kmin=1, kmax=ctx.num_patches)
        keep = keep_by_counts(r[:, 0, 1:], counts)
        cov = (ctx.teacher_map.float() * keep).sum(1)
        return keep, cov / cov.mean().clamp_min(1e-12)
