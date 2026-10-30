"""H-W4: mid-depth CLS attention. The teacher gets the K patches with the highest head-mean CLS attention in
student block `layer` (default 9 of 0..11) instead of the last block (MaskedKD).

Basis: analysis_group_fidelity_1030_143842_1ea0 (full-KD student, K=49): masked-teacher val accuracy / worst
group with block-11 CLS attention (= MaskedKD) 86.91 / 47.37, block-9 92.99 / 72.18, rollout 91.66 / 72.93,
teacher's own last-block attention (oracle) 66.17 WGA. The last block's top-K over-weights high-norm tokens
(toknorm overlap with MaskedKD 0.46 vs random 0.25) and the masked teacher then fails on waterbirds on land
(group 2). Reading: the last block of a ViT moves global information into a few low-information "register"
tokens (Darcet et al. ICLR 2024) that are not where the object evidence is; one block earlier the CLS query
still attends to object patches, which a masked teacher can classify without the background shortcut.

sel_args: layer = student block index (default 9).
"""
from harness.selector_api import SelectorBase, topk_mask

META = {
    "method": "layer-wise CLS attention probing (where a ViT reads object evidence vs global/register tokens)",
    "paper": "Darcet, Oquab, Mairal & Bojanowski, Vision Transformers Need Registers, ICLR 2024; "
             "Abnar & Zuidema ACL 2020 (attention as explanation)",
    "why": "mid-depth CLS attention still points at object patches; the last block's peaks include "
           "register-like background tokens that leave the masked teacher on the background shortcut",
    "vs_maskedkd": "same CLS-attention ranking, read from a mid-depth student block chosen by group-sliced "
                   "masked-teacher fidelity, not from the last block",
    "prior_art": "MaskedKD (last block); Attention Rollout mixes all blocks including the last; EViT/ToMe prune "
                 "inside the model itself; none choose the layer for KD-teacher input by group fidelity",
    "parent": "none",
}


class Selector(SelectorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer = int(kwargs.get("layer", 9))
        self.NEEDS = {"attn": [self.layer], "tokens": []}

    def select(self, ctx):
        scores = ctx.out.attn[self.layer].detach().float().mean(dim=1)[:, 0, 1:]
        return topk_mask(scores, int(ctx.budget))
