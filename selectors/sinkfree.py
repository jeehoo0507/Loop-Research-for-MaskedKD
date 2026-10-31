"""H-C11: MaskedKD's last-block CLS attention with the student's high-norm "sink" tokens excluded, top-K patches.

Basis (CUB, no position prior): analysis_cub_sink_check_1031_174301_0b30 (H-C10). In the student, ~25 of 196 patch
tokens per image have a block-10 output norm above median + 3*MAD; they take 0.39 of the last-block CLS attention
and fill 0.38-0.57 of MaskedKD's top-K (rollout's top-K: ~0.05). Scoring them out and taking the next tokens by the
same attention closes 0.90 / 0.79 / 0.60 of the masked teacher's TCKD gap between MaskedKD and rollout at
K = 12 / 29 / 49 (common student, image-bootstrap CIs above 0). Darcet et al. (ICLR 2024) describe such high-norm
tokens as global-information stores that draw late CLS attention but carry little local content.
"""
from harness.selector_api import SelectorBase, topk_mask

BLOCK = 10      # output of block 10 = input of the last block, whose attention MaskedKD reads
MAD_K = 3.0

META = {
    "method": "last-block CLS attention (MaskedKD) with high-norm artifact tokens (per-image median+3MAD of the "
              "block-10 token norms) excluded",
    "paper": "Darcet et al. ICLR 2024 (Vision Transformers Need Registers: high-norm artifact tokens); Son et al. "
             "ECCV 2024 (MaskedKD)",
    "why": "the student's high-norm tokens absorb last-block CLS attention but are not informative inputs for the "
           "teacher; skipping them keeps the masked teacher's soft label closer to the full one (H-C10)",
    "vs_maskedkd": "same attention score, but tokens whose block-10 norm is an outlier within the image get the "
                   "lowest score, so the next-attended tokens fill the budget",
    "prior_art": "registers (Darcet 2024) retrain with extra tokens to remove artifacts; here the artifacts are "
                 "detected per image at no training cost and only used to steer which tokens the teacher sees",
    "parent": "maskedkd",
}


class Selector(SelectorBase):
    NEEDS = {"attn": [-1], "tokens": [BLOCK]}

    def select(self, ctx):
        a = ctx.out.attn[-1].detach().float().mean(dim=1)[:, 0, 1:]
        nrm = ctx.out.tokens[BLOCK].detach()[:, 1:].float().norm(dim=-1)
        med = nrm.median(dim=1, keepdim=True).values
        mad = (nrm - med).abs().median(dim=1, keepdim=True).values
        sink = nrm > med + MAD_K * mad
        return topk_mask(a.masked_fill(sink, -1.0), int(ctx.budget))
