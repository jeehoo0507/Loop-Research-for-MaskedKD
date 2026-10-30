"""H-W17: attention rollout with a per-image token budget set by how spread the student's rollout is.

k_i = K * e_i / mean_batch(e),  e_i = exp(entropy of the normalised rollout row R_i[cls, 1:]) (effective number of
attended patches), clipped to [K/2, 2K] and trimmed so the batch mean stays <= K; keep the top-k_i rollout patches.
An image whose student attention is concentrated (small object, confident) gives the teacher fewer tokens; an image
where it is spread (large or ambiguous object, object split by cutmix) gives more. Same ranking as the parent
rollout, so any change against rollout comes from the budget allocation alone.

Exploratory screen: the teacher-side proxies failed out of sample (H-W13/H-W15), so no proxy pre-filter. Target is
K=49, where rollout is inside the WGA tolerance but 0.42 %p short on accuracy (screen_rollout_k49_s0_1030_135953_a1a9)
and center is inside on accuracy but not on WGA (screen_center_k49_s0_1030_135953_4476).
"""
import torch

from harness.selector_api import SelectorBase

META = {
    "method": "attention rollout (CLS row over all student blocks) with entropy-adaptive per-image budget",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Meng et al. CVPR 2022 AdaViT / Yin et al. CVPR 2022 A-ViT "
             "(input-adaptive token counts)",
    "why": "a fixed K cuts off the object on images where the student's evidence is spread and wastes tokens on "
           "background where it is concentrated; spending the same mean budget by spread keeps the teacher whole "
           "where it is most likely to lose the object",
    "vs_maskedkd": "all-layer rollout instead of last-block CLS attention, and a per-image K instead of a fixed K",
    "prior_art": "A-ViT/AdaViT/DynamicViT adapt token counts inside the model being run at inference; here the "
                 "student's attention spread sets how many inputs the frozen KD teacher receives",
    "parent": "rollout",
}


class Selector(SelectorBase):
    NEEDS = {"attn": list(range(12)), "tokens": []}

    def select(self, ctx):
        r = None
        for a in ctx.out.attn:
            a = a.detach().float().mean(dim=1)
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            r = a if r is None else a @ r
        R = r[:, 0, 1:]
        b, n = R.shape
        k = int(ctx.budget)
        p = R.clamp_min(1e-12)
        p = p / p.sum(dim=1, keepdim=True)
        e = torch.exp(-(p * p.log()).sum(dim=1))
        ki = torch.floor(k * e / e.mean()).long().clamp(max(1, k // 2), min(2 * k, n))
        excess = int(ki.sum().item()) - k * b
        while excess > 0:
            idx = ki.argsort(descending=True)[:excess]
            ki[idx] -= 1
            excess = int(ki.sum().item()) - k * b
        rank = R.argsort(dim=1, descending=True).argsort(dim=1)
        return rank < ki.unsqueeze(1)
