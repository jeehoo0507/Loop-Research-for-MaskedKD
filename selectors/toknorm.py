"""H5: student token-norm selection (block depth-2 output, i.e. block 10 of 12).

score_j = || h_j ||_2 of the student's patch token j after block 10. No attention, no gradient.
Basis: analysis_signal_fidelity_1030_014040_eb3d. At K=25/49 this signal kept the masked teacher closest
to the full teacher of all non-spatial signals (K=49, final mkd98 student: fid_kl 1.156 vs MaskedKD 1.634,
random 1.281), while its picks are not centre-biased (cdist 5.03 vs random 5.33) and overlap MaskedKD
only at chance (0.26). Only `center` was better on that proxy. Draft: not queued until the K=12/25/49
baseline screens show whether the fidelity proxy ranks val accuracy.
"""
from harness.selector_api import SelectorBase, topk_mask

META = {
    "method": "token-norm analysis of transformer residual streams (high-norm tokens carry global information)",
    "paper": "Darcet, Oquab, Mairal & Bojanowski, Vision Transformers Need Registers, ICLR 2024; "
             "Kobayashi et al., Attention is Not Only a Weight, EMNLP 2020",
    "why": "high-norm student tokens are where the student stores image-level information, so a teacher "
           "fed those tokens stays closest to its full-input prediction (analysis_signal_fidelity_1030_014040_eb3d)",
    "vs_maskedkd": "MaskedKD ranks by last-block CLS attention (where the student looks); this ranks by the "
                   "residual-stream norm of each token (what the student has written into it); picks overlap ~chance",
    "prior_art": "norm-based token importance appears in inference-time pruning (e.g. ATS uses value norms "
                 "times attention); here the raw residual norm alone picks the teacher's KD input",
    "parent": "none",
}


class Selector(SelectorBase):
    NEEDS = {"attn": [], "tokens": [-2]}

    def select(self, ctx):
        scores = ctx.out.tokens[-2].detach().float()[:, 1:].norm(dim=-1)
        return topk_mask(scores, int(ctx.budget))
