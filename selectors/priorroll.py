"""H-W12: attention rollout with a Gaussian centre prior. Score = log rollout(CLS -> patch) - d^2 / (2 sigma^2),
d = patch distance from the image centre; keep the top-K. sigma defaults to the radius of the K-patch centre disc,
sqrt(K / pi), so the prior is as wide as the `center` baseline at the same budget and the rollout signal decides
which patches inside / just outside that disc are kept.

Basis: analysis_prior_rollout_1030_174424_16a6. The centre disc wins at K=12 and rollout at K=29/49 (screen seed 0),
but neither at every K. Read from rollout-trained students (rollout12/29/49, the self proxy closest to this parent),
prior_s{2,3,4} gives a lower worst-group KL(full teacher || masked teacher) than both center and top_rollout at every
K (e.g. K=12 rollout12: 0.637 vs center 0.694 / rollout 0.700; K=49 rollout49: 0.227 vs 0.300 / 0.320). The prior
also removes the self-feedback loss of plain rollout (rollout12 student, K=29: top_rollout WGA 46.62, prior 60-62).
sigma = sqrt(K/pi) (~1.95 / 3.04 / 3.95 at K=12/29/49) tracks the best sigma of that analysis per K (2 / 4 / 4);
it was fixed after seeing the analysis, not tuned on any screen run.

sel_args: sigma (float, optional) overrides the disc-radius rule.
"""
import math

import torch

from harness.selector_api import SelectorBase, topk_mask

META = {
    "method": "attention rollout (CLS row over all student blocks) with a Gaussian spatial prior",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Sagawa et al. ICLR 2020 (worst-group evaluation)",
    "why": "a compact on-object region keeps the masked teacher faithful on the minority groups at small K, while "
           "the rollout signal moves the kept set toward the object when it is off-centre",
    "vs_maskedkd": "all-layer rollout instead of last-block CLS attention, and a centre prior that keeps the selection "
                   "compact instead of a scattered top-K",
    "prior_art": "rollout (no prior, scattered) and center (prior only, no signal); MaskedKD uses last-block attention; "
                 "saliency-with-centre-bias models (e.g. Judd et al. 2009) are for fixation prediction, not KD inputs",
    "parent": "rollout",
}


class Selector(SelectorBase):
    NEEDS = {"attn": list(range(12)), "tokens": []}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma = kwargs.get("sigma")

    def select(self, ctx):
        r = None
        for a in ctx.out.attn:
            a = a.detach().float().mean(dim=1)
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            r = a if r is None else a @ r
        R = r[:, 0, 1:]
        n, k = ctx.num_patches, int(ctx.budget)
        side = int(round(n ** 0.5))
        rr = torch.arange(n, device=R.device)
        c = (side - 1) / 2
        d2 = ((rr // side).float() - c) ** 2 + ((rr % side).float() - c) ** 2
        s = float(self.sigma) if self.sigma is not None else math.sqrt(k / math.pi)
        return topk_mask(R.clamp_min(1e-12).log() - d2 / (2 * s * s), k)
