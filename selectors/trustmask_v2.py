"""TrustMask v2 (H-A31, QUEUE 2026-11-03 18:40 "how many?"): graded per-image token counts from the cached teacher
map's concentration, inside the v1 selector.

The human's TrustMask note: an image whose teacher map sits on a few tokens needs fewer kept tokens, an image whose
map is spread needs more. Concentration = effective token count exp(entropy) of the (renormalised) cached teacher
map. Within the batch, images are ranked by it and split in thirds: the most concentrated third gets 2/3 K, the
middle third K, the most spread third 4/3 K (`allocate`, batch mean K, kmin 1; every image gets the teacher).
Tokens inside each image are picked exactly as in v1 (late-block rollout, blocks 6..11) and v1's KD weight
(teacher-map mass covered / batch mean) is kept, so only "how many" changes. Ranks and fixed ratios only: no
dataset constant.
"""
import torch

from harness.selector_api import SelectorBase, allocate, keep_by_counts

START = 6
RATIOS = (2.0 / 3.0, 1.0, 4.0 / 3.0)

META = {
    "method": "v1 (late-block attention rollout picks, cached teacher-map coverage as KD weight) with graded "
              "per-image token counts from the cached teacher map's effective token count (batch thirds 2/3, 1, "
              "4/3 of K)",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Son et al. ECCV 2024 (MaskedKD); human TrustMask note "
             "2026-10-28 ('how many?')",
    "why": "an image whose teacher evidence is concentrated keeps it with fewer tokens; a spread one loses it at a "
           "fixed K, so moving tokens from concentrated to spread images should keep the masked teacher's target "
           "closer to the full teacher at the same mean cost",
    "vs_maskedkd": "late-block rollout instead of last-block CLS attention, a per-image KD weight and per-image "
                   "token counts from the cached teacher map (same mean teacher cost)",
    "prior_art": "Dynamic KD / DynKD allocate the teacher per sample all-or-nothing by student uncertainty; v2 "
                 "grades token counts (no image without teacher) by the teacher's own attribution spread",
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

    def order(self, ctx):
        t = ctx.teacher_map.float().clamp_min(0)
        p = t / t.sum(1, keepdim=True).clamp_min(1e-12)
        return -(p * p.clamp_min(1e-12).log()).sum(1)   # entropy: higher = more spread

    def select(self, ctx):
        r = None
        for a in ctx.out.attn[START:]:
            a = a.detach().float().mean(dim=1)
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            r = a if r is None else a @ r
        counts = allocate(_groups(self.order(ctx)), ctx.budget, kmin=1, kmax=ctx.num_patches)
        keep = keep_by_counts(r[:, 0, 1:], counts)
        cov = (ctx.teacher_map.float() * keep).sum(1)
        return keep, cov / cov.mean().clamp_min(1e-12)
