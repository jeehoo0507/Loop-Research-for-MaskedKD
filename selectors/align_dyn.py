"""H-C24 (QUEUE 2026-11-01 22:40 + '보충'): all-or-nothing teacher concentration, interpretability signal.

Same split as rand_dyn/dynkd (whole teacher for m images of the batch, CE only for the rest). The m images are the
ones whose student attention rollout (all 12 blocks) agrees least with the teacher's cached rollout of the whole
image (row Spearman). The interpretability branch's primary signal pre-registered in H-C22 ("align", need = -align).
The cache build cost (0.35 GFLOPs/img-step at CUB, about 2 full-teacher tokens per image) is paid out of the budget:
m = int((budget - CACHE_TOK) * B // 196).
"""
import torch

from harness.selector_api import SelectorBase

CACHE_TOK = 2.0

META = {
    "method": "student attention rollout vs cached teacher attention rollout, per-image rank agreement",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Li et al. EMNLP 2021 (Dynamic KD, sample-level teacher use)",
    "why": "an image where the student attends to different evidence than the teacher is where the teacher's soft "
           "label carries what the student lacks, so it gets the whole teacher (H-C22 interpretability primary)",
    "vs_maskedkd": "chooses which images get the whole teacher, not which tokens; MaskedKD gives every image K tokens",
    "prior_art": "Dynamic KD picks images by student entropy (builtin dynkd); here by student-teacher attribution agreement",
    "parent": "rand_dyn",
    "uses_teacher_cache": True,
    "per_image_budget": True,
}


def _rank(x):
    return x.argsort(dim=1).argsort(dim=1).float()


class Selector(SelectorBase):
    NEEDS = {"attn": list(range(12)), "tokens": []}

    def select(self, ctx):
        B, n = ctx.images.shape[0], ctx.num_patches
        m = int(max(ctx.budget - CACHE_TOK, 0.0) * B // n)
        keep = torch.zeros(B, n, dtype=torch.bool, device=ctx.images.device)
        if not m:
            return keep
        r = None
        for a in ctx.out.attn:
            a = a.detach().float().mean(dim=1)
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            r = a if r is None else a @ r
        ra, rb = _rank(r[:, 0, 1:]), _rank(ctx.teacher_map.float())
        ra = ra - ra.mean(1, keepdim=True)
        rb = rb - rb.mean(1, keepdim=True)
        align = (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1) + 1e-12)
        keep[(-align).topk(m).indices] = True
        return keep
