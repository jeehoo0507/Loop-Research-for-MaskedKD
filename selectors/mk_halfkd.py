"""H-C26 (diagnostic, not a method): official MaskedKD top-K with every image's KD term weighted 0.5.

rand_dyn@98 gives the whole teacher to half the batch and CE only to the rest, so the batch loss is
0.75 CE + 0.25 KD, against 0.5 CE + 0.5 KD for maskedkd@98 / Full. This keeps MaskedKD's token choice and
coverage (every image, K tokens) but halves the KD weight (0.5 CE + 0.25 KD), at the same teacher cost as
maskedkd@K. It asks whether rand_dyn lost the KD gain because of the smaller KD weight or because of coverage.
"""
import torch

from harness.selector_api import SelectorBase, topk_mask

META = {
    "method": "MaskedKD top-K (last student block CLS attention) with kd_weight 0.5 on every image",
    "paper": "Son et al. 2023 (MaskedKD); diagnostic control for H-C25 rand_dyn",
    "why": "separates KD-weight dilution from per-image coverage as the reason all-or-nothing rand_dyn@98 "
           "stays at CE level while maskedkd@98 reaches Full",
    "vs_maskedkd": "same tokens and teacher cost; only the KD term is weighted 0.5",
    "prior_art": "none claimed; this is a loss-weight ablation, not a selector",
    "parent": "maskedkd",
}


class Selector(SelectorBase):
    NEEDS = {"attn": [-1], "tokens": []}

    def select(self, ctx):
        scores = ctx.out.attn[-1].detach().float().mean(dim=1)[:, 0, 1:]
        keep = topk_mask(scores, int(ctx.budget))
        w = torch.full((keep.shape[0],), 0.5, device=keep.device)
        return keep, w
