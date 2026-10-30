"""H2: gradient-weighted last-block CLS attention (Chefer et al. 2021 / Hao et al. 2021 style).

score_j = mean_h relu( dA_h[cls,j] * A_h[cls,j] ),  dA = d(max_c student logit) / d(attn of last block).
The student's own predicted class is the target (no labels, no teacher). The gradient only flows back
through the last block's attention-value product, the norm and the head, so it is far cheaper than a
full student backward. Draft: waits for analysis_signal_fidelity_1030_003046_ecd0 ("gradattn_last").
"""
import torch

from harness.selector_api import SelectorBase, topk_mask

META = {
    "method": "gradient-weighted attention (Grad-SAM / last layer of Generic Attention Explainability)",
    "paper": "Chefer, Gur & Wolf ICCV 2021; Hao et al. AAAI 2021 (attention attribution)",
    "why": "keeps tokens whose attention actually moves the student's own decision, not just where it looks",
    "vs_maskedkd": "MaskedKD ranks by raw last-block CLS attention; this weights it by its gradient on the "
                   "student's predicted-class logit and drops negatively contributing attention",
    "prior_art": "Prune and Merge (arXiv 2503.23455) uses gradient-weighted attention to prune for inference; "
                 "here it picks the teacher's input tokens during KD from the student's view",
    "parent": "maskedkd",
}


class Selector(SelectorBase):
    NEEDS = {"attn": [-1], "tokens": []}

    def select(self, ctx):
        attn = ctx.out.attn[-1]
        target = ctx.out.logits.float().max(1).values.sum()
        (g,) = torch.autograd.grad(target, attn, retain_graph=True)
        scores = (g.detach().float() * attn.detach().float()).clamp(min=0).mean(1)[:, 0, 1:]
        return topk_mask(scores, int(ctx.budget))
