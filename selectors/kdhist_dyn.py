"""H-C24 (QUEUE 2026-11-01 22:40 + '보충'): all-or-nothing teacher concentration, learning-state signal.

Same split as rand_dyn/dynkd (whole teacher for m = int(budget * B // 196) images of the batch, CE only for the rest).
The m images are the ones with the largest KD loss the last time they got the teacher (observe()); images never
taught yet come first, in random order. The non-interpretability branch's primary signal pre-registered in H-C22
("kd29", the student's KL to the masked teacher), here measured on the whole teacher at no extra teacher cost.
"""
import torch

from harness.selector_api import SelectorBase

META = {
    "method": "per-image KD loss history (last observed KL to the whole teacher), no attribution",
    "paper": "Li et al. EMNLP 2021 (Dynamic KD, sample-level teacher use); loss-based sample selection (e.g. OHEM)",
    "why": "an image whose student was far from the teacher last time is where the teacher's soft label still "
           "carries most the student has not learned, so it gets the whole teacher (H-C22 non-interp primary)",
    "vs_maskedkd": "chooses which images get the whole teacher, not which tokens; MaskedKD gives every image K tokens",
    "prior_art": "Dynamic KD picks images by student entropy (builtin dynkd); here by the observed KD loss history",
    "parent": "rand_dyn",
    "per_image_budget": True,
}


class Selector(SelectorBase):
    NEEDS = {"attn": [], "tokens": []}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last = None   # [N] last observed KD loss per training row, +inf = never taught

    def _grow(self, idx):
        need = int(idx.max().item()) + 1
        if self.last is None or self.last.numel() < need:
            new = torch.full((max(need, 6000),), float("inf"), device=idx.device)
            if self.last is not None:
                new[: self.last.numel()] = self.last
            self.last = new

    def select(self, ctx):
        B, n = ctx.images.shape[0], ctx.num_patches
        m = int(ctx.budget * B // n)
        keep = torch.zeros(B, n, dtype=torch.bool, device=ctx.images.device)
        if not m:
            return keep
        tie = torch.rand(B, device=ctx.images.device)
        if ctx.sample_idx is None:
            score = tie
        else:
            idx = ctx.sample_idx.to(ctx.images.device).long()
            self._grow(idx)
            s = self.last[idx]
            score = torch.where(torch.isinf(s), 1e9 + tie, s + 1e-6 * tie)
        keep[score.topk(m).indices] = True
        return keep

    def observe(self, fb):
        if fb.sample_idx is None:
            return
        idx = fb.sample_idx.long().to(fb.kd.device)
        self._grow(idx)
        ok = ~torch.isnan(fb.kd)
        self.last[idx[ok]] = fb.kd[ok].float()
