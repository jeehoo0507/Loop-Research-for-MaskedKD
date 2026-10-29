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


@torch.no_grad()
def oracle_keep(teacher, x, k):
    """ANALYSIS ONLY (upper-bound reference, never a claim): the teacher's own last-block CLS attention
    on the full image picks the tokens. Its full teacher forward is billed to the selector segment."""
    from .models import student_forward   # works for any timm ViT
    attn = student_forward(teacher, x, attn_layers=[-1]).attn[-1]
    return topk_mask(attn.float().mean(dim=1)[:, 0, 1:], int(k))


BUILTIN = {"full": Full, "maskedkd": MaskedKD, "random": Random}
