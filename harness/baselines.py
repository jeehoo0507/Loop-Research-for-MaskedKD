"""Reference methods, locked inside the harness so the agent cannot weaken them."""
import torch

from .selector_api import Context, SelectorBase, topk_mask


class Full(SelectorBase):
    """Full KD: teacher sees all 196 patch tokens."""

    def select(self, ctx: Context):
        B = ctx.images.shape[0]
        return torch.ones(B, ctx.num_patches, dtype=torch.bool, device=ctx.images.device)


class MaskedKD(SelectorBase):
    """Official MaskedKD: last student block, head-mean attention of the CLS query over patch keys, top-K.

    losses.py in effl-lab/MaskedKD: attn.mean(dim=1)[:, 0, 1:] -> topk(len_num_keep).
    """
    NEEDS = {"attn": [-1], "tokens": []}

    def select(self, ctx: Context):
        attn = ctx.out.attn[-1]
        scores = attn.detach().float().mean(dim=1)[:, 0, 1:]
        return topk_mask(scores, int(ctx.budget))


class Random(SelectorBase):
    """K patch tokens chosen uniformly at random per image."""

    def select(self, ctx: Context):
        B = ctx.images.shape[0]
        scores = torch.rand(B, ctx.num_patches, device=ctx.images.device)
        return topk_mask(scores, int(ctx.budget))


class Rollout(SelectorBase):
    """Attention rollout (Abnar & Zuidema, ACL 2020) on the student: head-mean attention of every block with the
    residual added (0.5 A + 0.5 I, rows renormalised), multiplied from the first block to the last; the CLS row
    ranks the patches, top-K. The best-known way to read "where a ViT looks" from all layers: a selector that is
    only a variant of it is not new."""
    NEEDS = {"attn": list(range(12)), "tokens": []}

    def select(self, ctx: Context):
        r = None
        for a in ctx.out.attn:
            a = a.detach().float().mean(dim=1)
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            r = a if r is None else a @ r
        return topk_mask(r[:, 0, 1:], int(ctx.budget))


class Grid(SelectorBase):
    """K patch tokens evenly spaced in raster order, same for every image. A selector that wins only by
    spreading its tokens over the image does not beat this."""

    def select(self, ctx: Context):
        B, n, k = ctx.images.shape[0], ctx.num_patches, int(ctx.budget)
        idx = torch.linspace(0, n - 1, k, device=ctx.images.device).round().long()
        return torch.zeros(B, n, dtype=torch.bool, device=ctx.images.device).index_fill_(1, idx, True)


class Center(SelectorBase):
    """The K patch tokens closest to the image centre, same for every image (object-centred datasets)."""

    def select(self, ctx: Context):
        B, n, k = ctx.images.shape[0], ctx.num_patches, int(ctx.budget)
        side = int(round(n ** 0.5))
        r = torch.arange(n, device=ctx.images.device)
        c = (side - 1) / 2
        dist = ((r // side - c) ** 2 + (r % side - c) ** 2).float() + r.float() * 1e-6   # ties: raster order
        return topk_mask((-dist).expand(B, n), k)


@torch.no_grad()
def oracle_keep(teacher, x, k):
    """ANALYSIS ONLY (upper-bound reference, never a claim): the teacher's own last-block CLS attention
    on the full image picks the tokens. Its full teacher forward is billed to the selector segment."""
    from .models import student_forward   # works for any timm ViT
    attn = student_forward(teacher, x, attn_layers=[-1]).attn[-1]
    return topk_mask(attn.float().mean(dim=1)[:, 0, 1:], int(k))


BUILTIN = {"full": Full, "maskedkd": MaskedKD, "random": Random, "grid": Grid, "center": Center, "rollout": Rollout}
