"""H-C29 (diagnostic, QUEUE 2026-11-01 22:40 "일부는 전체, 나머지는 작은 K"): half all-or-nothing, half fixed K.

The efficiency curve (H-C23~28) crosses: all-or-nothing rand_dyn beats fixed-K maskedkd at K<=29, loses at K=98.
This splits the same token budget in two fixed halves: m = int(0.5 * budget * B // 196) random images get the
whole teacher, every other image gets an equal share of the rest (MaskedKD top-k by last-block CLS attention).
The split fraction 0.5 is fixed before any result; it is not tuned.
"""
import torch

from harness.selector_api import SelectorBase, keep_by_counts

META = {
    "method": "MaskedKD top-K (last student block CLS attention) for most images + whole teacher for random images",
    "paper": "Son et al. 2023 (MaskedKD); Li et al. EMNLP 2021 (Dynamic KD, all-or-nothing teacher use)",
    "why": "rand_dyn wins at small budgets and maskedkd at large ones; half of each asks whether the two gains "
           "(a few clean whole-teacher labels, plus coverage of every image) add up at the same teacher cost",
    "vs_maskedkd": "half the token budget goes to whole-teacher passes for random images; the rest get fewer tokens",
    "prior_art": "mixture of builtin rand_dyn and maskedkd at the same mean budget; no new signal",
    "parent": "rand_dyn",
    "per_image_budget": True,
}


class Selector(SelectorBase):
    NEEDS = {"attn": [-1], "tokens": []}

    def select(self, ctx):
        B, n = ctx.images.shape[0], ctx.num_patches
        dev = ctx.images.device
        total = int(ctx.budget * B)
        m = min(int(0.5 * total // n), B)
        perm = torch.randperm(B, device=dev)
        counts = torch.zeros(B, dtype=torch.long, device=dev)
        counts[perm[:m]] = n
        others = perm[m:]
        if others.numel():
            rest = total - m * n
            k, r = divmod(rest, others.numel())
            counts[others] = min(k, n)
            if r and k < n:
                counts[others[:r]] += 1
        scores = ctx.out.attn[-1].detach().float().mean(dim=1)[:, 0, 1:]
        return keep_by_counts(scores, counts)
