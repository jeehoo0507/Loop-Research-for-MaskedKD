"""Attention rollout over a chosen set of student blocks (human, 2026-11-12, surgical fine-tuning study).

One signal, one variable: the harness rollout (head-mean attention, 0.5 A + 0.5 I, rows renormalised, multiplied in
block order, CLS row, top-K) with the set of blocks as the only thing that changes. Every arm of the study is this
file with different sel_args, so no arm differs from another by anything but the blocks it reads.

sel_args:
  layers  list of student block indices (0..11), default all 12. Multiplied in ascending order.
  head    "mean" (default) or a head index: read that one head instead of the head mean.

Identities (tests/test_harness.py::test_layerroll_equals_the_references), all with head="mean":
  layers = [11]      -> MaskedKD (harness builtin), exactly
  layers = 0..11     -> rollout (harness builtin), exactly
  layers = 6..11     -> selectors/lateroll.py, exactly
A single block is read without the residual step: 0.5 A + 0.5 I and the row renormalisation only rescale the CLS row's
patch entries, so the ranking is the same, and skipping them makes the single-block arm bit-identical to MaskedKD.

Block groups follow Surgical Fine-Tuning's Block 1-4: B1 = 0-2, B2 = 3-5, B3 = 6-8, B4 = 9-11.
"""
from harness.selector_api import SelectorBase, topk_mask
import torch

DEPTH = 12

META = {
    "method": "attention rollout (CLS row) over a chosen subset of the student's blocks",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Lee et al. ICLR 2023 (surgical fine-tuning: which block "
             "matters depends on the type of distribution shift)",
    "why": "if the block that a distribution shift lives in gives the least reliable attention, the teacher should "
           "get the tokens ranked by the other blocks; the block set is the experimental variable",
    "vs_maskedkd": "MaskedKD is the special case layers=[11]; other block sets read earlier or several blocks",
    "prior_art": "full rollout (harness baseline, layers 0..11), lateroll (layers 6..11), midattn (one block); "
                 "this file is their common form, not a new method",
    "parent": "none",
}


class Selector(SelectorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layers = kwargs.get("layers", list(range(DEPTH)))
        self.layers = sorted({int(i) % DEPTH for i in layers})
        if not self.layers:
            raise ValueError("layers must name at least one block")
        head = kwargs.get("head", "mean")
        self.head = None if head == "mean" else int(head)
        self.NEEDS = {"attn": list(self.layers), "tokens": []}

    def _block(self, ctx, i):
        a = ctx.out.attn[i].detach().float()
        return a.mean(dim=1) if self.head is None else a[:, self.head]

    def select(self, ctx):
        if len(self.layers) == 1:
            return topk_mask(self._block(ctx, self.layers[0])[:, 0, 1:], int(ctx.budget))
        r = None
        for i in self.layers:
            a = self._block(ctx, i)
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            r = a if r is None else a @ r
        return topk_mask(r[:, 0, 1:], int(ctx.budget))
