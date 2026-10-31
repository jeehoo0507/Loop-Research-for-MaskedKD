"""H-W23 (Codex IDEA 1): concept coverage. Factorize the student's block-6 patch features into a few non-negative
concept maps (Deep Feature Factorization) and pick K patches greedily so that every concept gets a share, with
diminishing returns for a concept that is already represented.

Basis: Collins, Achanta & Susstrunk ECCV 2018 (DFF: NMF of CNN activations gives part/object/background concept
maps). Project evidence it is tested against: analysis_shortcut_probe_1030_160112_bfc0 (last-block top-K hands the
teacher the most background information) and the K=12 screens where spreading rules (nms, half_grid, grid, random)
lose WGA. Coverage normalizes each concept map to sum 1, so a spatially large background concept cannot take most of
the budget; whether covering it at all hurts WGA is what the screen tests.

Cost: rank-6 NMF on [196, 384] per image (15 multiplicative updates) + K greedy steps, no grad, no extra forward.

sel_args: layer (default 6), rank (default 6), iters (default 15).
"""
import torch

from harness.selector_api import SelectorBase

META = {
    "method": "Deep Feature Factorization (NMF concept maps of intermediate features) with budgeted concept coverage",
    "paper": "Collins, Achanta & Susstrunk, Deep Feature Factorization for Concept Discovery, ECCV 2018",
    "why": "the teacher should see every part/concept the student represents in the image once, instead of many "
           "tokens of the one concept that dominates the last block's attention",
    "vs_maskedkd": "no attention ranking: patches are chosen to cover NMF concepts of student block-6 features, with "
                   "log diminishing returns per concept",
    "prior_art": "Token Pooling (Marin et al. 2021) clusters tokens to minimize reconstruction error and merges them; "
                 "ToMe merges similar tokens; this keeps original patches chosen by normalized concept coverage",
    "parent": "none",
}

EPS = 1e-6


def _nmf(X, rank, iters):
    """X [B,N,C] >= 0 -> H [B,N,rank] >= 0 (multiplicative updates, deterministic init)."""
    B, N, C = X.shape
    order = X.norm(dim=-1).argsort(dim=1)                                  # [B,N]
    pos = torch.linspace(N // (2 * rank), N - 1 - N // (2 * rank), rank, device=X.device).round().long()
    seed_idx = order[:, pos]                                               # [B,rank] spread over norm quantiles
    W = torch.gather(X, 1, seed_idx.unsqueeze(-1).expand(-1, -1, C)) + EPS  # [B,rank,C]
    H = (X @ W.transpose(1, 2)).clamp_min(EPS)                             # [B,N,rank]
    for _ in range(iters):
        H = H * (X @ W.transpose(1, 2)) / (H @ (W @ W.transpose(1, 2)) + EPS)
        W = W * (H.transpose(1, 2) @ X) / ((H.transpose(1, 2) @ H) @ W + EPS)
    return H


class Selector(SelectorBase):
    NEEDS = {"attn": [], "tokens": [6]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer = int(kwargs.get("layer", 6))
        self.rank = int(kwargs.get("rank", 6))
        self.iters = int(kwargs.get("iters", 15))
        self.NEEDS = {"attn": [], "tokens": [self.layer]}

    def select(self, ctx):
        k = int(ctx.budget)
        with torch.no_grad():
            Z = ctx.out.tokens[self.layer][:, 1:].detach().float()          # [B,N,D], CLS dropped
            X = torch.cat([Z.clamp_min(0), (-Z).clamp_min(0)], dim=-1)
            H = _nmf(X, self.rank, self.iters)
            U = H / (H.sum(dim=1, keepdim=True) + EPS)                      # each concept map sums to 1
            B, N, _ = U.shape
            covered = torch.zeros(B, 1, self.rank, device=U.device)
            chosen = torch.zeros(B, N, dtype=torch.bool, device=U.device)
            tie = torch.arange(N, device=U.device).float() * 1e-9
            for _ in range(k):
                gain = torch.log1p(U / (1.0 / N + covered)).sum(-1) - tie   # [B,N]
                gain = gain.masked_fill(chosen, float("-inf"))
                pick = gain.argmax(dim=1)
                chosen[torch.arange(B, device=U.device), pick] = True
                covered = covered + U[torch.arange(B, device=U.device), pick].unsqueeze(1)
        return chosen
