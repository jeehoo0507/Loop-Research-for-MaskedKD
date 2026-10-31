"""H-C14 (QUEUE 2026-10-31 20:40, TrustMask single run): the cached teacher map as a filter, not a picker.

The teacher's rollout of the whole image (ctx.teacher_map, harness/teacher_cache.py) marks an allowed region, its top
ceil(1.5 K) patches; the student's late-block rollout (lateroll, blocks 6..11) picks the K tokens inside it.
Variant switches are fixed by the pre-registered rule in PROPOSALS H-C14 from analysis_cub_teacher_filter_1031_202037_032d:
ADAPTIVE (per-image count from the batch's top B*K teacher-map values, batch mean = K) and EARLY_ONLY (filter only
while progress < 0.5, plain lateroll after).
"""
import torch

from harness.selector_api import SelectorBase, topk_mask

START = 6
WIDEN = 1.5
ADAPTIVE = False
EARLY_ONLY = False

META = {
    "method": "teacher attention rollout (cached) as an allowed region + student late-block attention rollout inside it",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Son et al. ECCV 2024 (MaskedKD, teacher-picked tokens worse)",
    "why": "tokens the student attends to that the teacher also uses as evidence keep the masked teacher's soft label "
           "(H-C14) while still following the student, unlike teacher top-K which ignores what the student looks at",
    "vs_maskedkd": "student late-block rollout instead of last-block CLS attention, restricted to the teacher's region",
    "prior_art": "teacher_topk / teacher-guided masking pick by the teacher alone; MaskedKD picks by the student alone",
    "parent": "lateroll",
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
        s = r[:, 0, 1:]
        k = int(ctx.budget)
        if EARLY_ONLY and ctx.progress >= 0.5:
            return topk_mask(s, k)
        t = ctx.teacher_map.float()
        B, n = s.shape
        if ADAPTIVE:
            cnt = torch.zeros(B * n, device=t.device)
            cnt[t.reshape(-1).topk(B * (k - 1)).indices] = 1
            kk = cnt.view(B, n).sum(1).long().add(1).clamp(max=n)   # at least one token each, batch mean = K exactly
        else:
            kk = torch.full((B,), k, device=s.device, dtype=torch.long)
        m = torch.clamp(torch.ceil(WIDEN * kk.float()).long(), max=n)
        allowed = t.argsort(1, descending=True).argsort(1) < m[:, None]
        s = s.masked_fill(~allowed, -1.0)
        return s.argsort(1, descending=True).argsort(1) < kk[:, None]
