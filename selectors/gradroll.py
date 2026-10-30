"""H-W16: class-gradient-weighted rollout (Chefer, Gur & Wolf CVPR 2021 relevance rule) on the student.

R = I;  for each block b:  Abar_b = mean_h relu(dA_b,h * A_b,h);  R = R + Abar_b @ R;  score = R[cls, 1:].
dA = d(sum of the student's max logit) / d(attention of every block). Target is the student's own predicted
class (no labels, no teacher). Plain rollout is class-agnostic and keeps whatever the student attends to,
background included; the gradient term keeps only attention that raises the predicted-class logit.
Exploratory screen: the teacher-side proxies failed out of sample (H-W13/H-W15), so no proxy pre-filter.
"""
import torch

from harness.selector_api import SelectorBase, topk_mask

META = {
    "method": "gradient-weighted attention rollout (Generic Attention-model Explainability relevance rule)",
    "paper": "Chefer, Gur & Wolf CVPR 2021 (Transformer Interpretability Beyond Attention Visualization) / ICCV 2021",
    "why": "rollout keeps every token the student looks at, background included; weighting each layer's attention "
           "by its gradient on the student's own decision keeps only tokens that support the class evidence",
    "vs_maskedkd": "MaskedKD ranks by raw last-block CLS attention; this propagates gradient-weighted attention "
                   "through all 12 blocks and drops negatively contributing heads",
    "prior_art": "Chefer et al. use it for explanation; Prune and Merge (arXiv 2503.23455) uses gradient-weighted "
                 "attention to prune for inference; here it picks the frozen teacher's input tokens during KD",
    "parent": "rollout",
}


class Selector(SelectorBase):
    NEEDS = {"attn": list(range(12)), "tokens": []}

    def select(self, ctx):
        attns = list(ctx.out.attn)
        target = ctx.out.logits.float().max(1).values.sum()
        grads = torch.autograd.grad(target, attns, retain_graph=True)
        n = attns[0].shape[-1]
        r = torch.eye(n, device=attns[0].device).unsqueeze(0).expand(attns[0].shape[0], n, n)
        for a, g in zip(attns, grads):
            abar = (g.detach().float() * a.detach().float()).clamp(min=0).mean(dim=1)
            r = r + abar @ r
        return topk_mask(r[:, 0, 1:], int(ctx.budget))
