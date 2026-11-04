"""v0 ablation (QUEUE 2026-11-04 10:00): the teacher filter of trustfilt with MaskedKD's own picker inside it.

trustfilt = teacher's cached rollout top ceil(1.5 K) as the allowed region + the student's late-block rollout picks K
inside. This variant keeps the region and replaces the picker by MaskedKD's rule (last block, head-mean CLS attention,
top-K). If it matches trustfilt on worst-group accuracy, the teacher region alone breaks the feedback loop and
lateroll adds nothing; if it falls to MaskedKD's level, the region is not enough.
"""
import torch

from harness.selector_api import SelectorBase

WIDEN = 1.5

META = {
    "method": "cached teacher rollout top ceil(1.5K) as the allowed region; MaskedKD's last-block CLS attention picks "
              "the K tokens inside it",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (attention rollout, cached teacher map)",
    "why": "separates the two parts of trustfilt (teacher region vs late-block rollout picker): which one keeps the "
           "masked teacher from confirming the student's shortcut on the minority groups",
    "vs_maskedkd": "identical picker and K; the student may only pick inside the teacher's own evidence region",
    "prior_art": "trustfilt (teacher region + lateroll); teacher_topk (teacher alone)",
    "parent": "trustfilt",
    "uses_teacher_cache": True,
}


class Selector(SelectorBase):
    NEEDS = {"attn": [-1], "tokens": []}

    def select(self, ctx):
        s = ctx.out.attn[-1].detach().float().mean(dim=1)[:, 0, 1:]
        k = int(ctx.budget)
        t = ctx.teacher_map.float()
        n = s.shape[1]
        m = min(int(-(-WIDEN * k // 1)), n)
        allowed = t.argsort(1, descending=True).argsort(1) < m
        s = s.masked_fill(~allowed, -1.0)
        return s.argsort(1, descending=True).argsort(1) < k
