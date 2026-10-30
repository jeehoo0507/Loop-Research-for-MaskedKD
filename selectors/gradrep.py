"""Decomposition control for gradroll (required before promotion, AGENT_PROMPT "승격 전 분해").

Keeps the gradroll top-K tokens that MaskedKD (block-11 CLS attention top-K) also picks, and replaces the rest
with uniformly random patches. If gradroll's K=12 gain (screen_gradroll_k12_s0_1030_184259_e2cd) disappears,
the gain comes from the tokens gradroll picks that MaskedKD does not; if it stays, it comes from elsewhere.
Not a candidate method: an ablation whose parent is gradroll.
"""
import torch

from harness.selector_api import SelectorBase, topk_mask

META = {
    "method": "ablation of gradient-weighted attention rollout: non-MaskedKD tokens replaced by random patches",
    "paper": "Chefer, Gur & Wolf CVPR 2021 (relevance rule); Hooker et al. NeurIPS 2019 (replacement control)",
    "why": "tests whether gradroll's gain comes from the tokens it keeps that last-block CLS attention does not",
    "vs_maskedkd": "keeps only the gradroll/MaskedKD intersection; the remainder is random, not attention-ranked",
    "prior_art": "random-replacement control in the style of ROAR; not a proposed method",
    "parent": "gradroll",
}


class Selector(SelectorBase):
    NEEDS = {"attn": list(range(12)), "tokens": []}

    def select(self, ctx):
        attns = list(ctx.out.attn)
        k = int(ctx.budget)
        target = ctx.out.logits.float().max(1).values.sum()
        grads = torch.autograd.grad(target, attns, retain_graph=True)
        n = attns[0].shape[-1]
        r = torch.eye(n, device=attns[0].device).unsqueeze(0).expand(attns[0].shape[0], n, n)
        for a, g in zip(attns, grads):
            abar = (g.detach().float() * a.detach().float()).clamp(min=0).mean(dim=1)
            r = r + abar @ r
        grad_sel = topk_mask(r[:, 0, 1:], k)
        mkd_sel = topk_mask(attns[-1].detach().float().mean(dim=1)[:, 0, 1:], k)
        scores = torch.rand(grad_sel.shape, device=grad_sel.device)
        scores[grad_sel & mkd_sel] = 2.0
        return topk_mask(scores, k)
