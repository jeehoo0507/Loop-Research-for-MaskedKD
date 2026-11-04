"""v0 ablation (QUEUE 2026-11-04 10:00): the plain intersection, teacher top-K AND student late-block rollout top-K.

trustfilt widens the teacher region to 1.5 K so that exactly K tokens are kept. Here nothing is widened or padded: a
token is kept only if it is in the teacher's cached rollout top-K and in the student's late-block rollout top-K (at
least one token per image). The count varies per image and its mean is below K, so the teacher costs less than
MaskedKD at the same nominal K; the question is whether worst-group accuracy stays at trustfilt's level.
"""
import torch

from harness.selector_api import SelectorBase, topk_mask

START = 6

META = {
    "method": "intersection of the cached teacher rollout top-K and the student's late-block (6..11) attention rollout "
              "top-K, no widening, no padding (at least one token)",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Son et al. ECCV 2024 (MaskedKD)",
    "why": "tokens both the teacher uses as evidence and the student attends to; dropping the rest cuts the teacher's "
           "cost below MaskedKD@K and tests whether the robustness of trustfilt survives with fewer tokens",
    "vs_maskedkd": "student picks restricted to the teacher's top-K and nothing added back: fewer tokens per image",
    "prior_art": "trustfilt (widened region, exactly K kept); LeaF (NeurIPS 2025) prunes tokens the student attends "
                 "to but the teacher ignores, for LLM distillation",
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
        s = r[:, 0, 1:]
        k = int(ctx.budget)
        keep = topk_mask(s, k) & topk_mask(ctx.teacher_map.float(), k)
        empty = ~keep.any(1)
        if empty.any():   # no overlap: the student's single best token inside the teacher's top-K
            t_in = ctx.teacher_map.float().argsort(1, descending=True).argsort(1) < k
            best = s.masked_fill(~t_in, -1.0).argmax(1)
            keep[empty, best[empty]] = True
        return keep
