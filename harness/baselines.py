"""Reference methods, locked inside the harness so the agent cannot weaken them."""
import torch

from .selector_api import Context, SelectorBase, allocate, keep_by_counts, topk_mask


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


class TeacherTopK(SelectorBase):
    """Reference for selectors that use the cached teacher map: the K patches the teacher's own rollout of the
    whole image ranks highest (harness/teacher_cache.py), moved with this batch's crop. A cache-using method must
    beat this at the same K, or the student-side part adds nothing over the teacher signal itself. MaskedKD
    (Son et al.) found choosing by teacher attention worse than by student attention; this checks it here."""
    NEEDS = {"attn": [], "tokens": []}
    USES_TEACHER_CACHE = True

    def select(self, ctx: Context):
        return topk_mask(ctx.teacher_map, int(ctx.budget))


# --- per-image budget references (QUEUE 2026-11-01). Each spends the same mean teacher tokens as a fixed-K run at
# the same budget; a per-image budget method has to beat all of them, or its allocation rule adds nothing known
# work does not already give. Tokens inside an image: MaskedKD's student CLS ranking, unless said otherwise.

def _cls_scores(ctx):
    return ctx.out.attn[-1].detach().float().mean(dim=1)[:, 0, 1:]


def _entropy(logits):
    p = logits.detach().float().softmax(dim=1)
    return -(p * p.clamp_min(1e-12).log()).sum(dim=1)


class RandAlloc(SelectorBase):
    """Control: tokens per image drawn at random (same mean). A learned-state rule that does not beat this is
    not using the state."""
    NEEDS = {"attn": [-1], "tokens": []}

    def select(self, ctx: Context):
        w = torch.rand(ctx.images.shape[0], device=ctx.images.device)
        return keep_by_counts(_cls_scores(ctx), allocate(w, ctx.budget, kmax=ctx.num_patches))


class DynKD(SelectorBase):
    """Dynamic KD (Li et al., EMNLP 2021) carried to ViT tokens: the most uncertain images (student predictive
    entropy) get the whole teacher, the rest none (CE only), as many as the budget pays for. Sample-level,
    all-or-nothing allocation: a token-level per-image budget must beat it to be more than this."""
    NEEDS = {"attn": [], "tokens": []}

    def select(self, ctx: Context):
        B, n = ctx.images.shape[0], ctx.num_patches
        m = int(ctx.budget * B // n)
        keep = torch.zeros(B, n, dtype=torch.bool, device=ctx.images.device)
        if m:
            keep[_entropy(ctx.out.logits).topk(m).indices] = True
        return keep


class SalRatio(SelectorBase):
    """Per-image ratio from the image's own saliency, in the spirit of Salience-Based Adaptive Masking (SBAM,
    ECCV 2024): an image whose student attention is spread (high entropy, many tokens matter) gets more tokens.
    Uses no learning state, only the current attention map."""
    NEEDS = {"attn": [-1], "tokens": []}

    def select(self, ctx: Context):
        a = _cls_scores(ctx)
        p = a / a.sum(dim=1, keepdim=True)
        eff = torch.exp(-(p * p.clamp_min(1e-12).log()).sum(dim=1))   # effective number of attended tokens
        return keep_by_counts(a, allocate(eff, ctx.budget, kmin=1, kmax=ctx.num_patches))


class KDSched(SelectorBase):
    """Time schedule, in the spirit of Masters (arXiv 2512.22238, teacher masking ratio lowered in stages): every
    image gets K(t) = budget * (0.5 + t), t = training progress, so the run's mean is the budget. Changes over
    time only, not per image."""
    NEEDS = {"attn": [-1], "tokens": []}

    def select(self, ctx: Context):
        k = int(min(ctx.num_patches, round(ctx.budget * (0.5 + ctx.progress))))
        k = min(k, int(2 * ctx.budget))
        return topk_mask(_cls_scores(ctx), max(k, 1))


class EntWeight(SelectorBase):
    """Fixed K (MaskedKD tokens), KD loss weighted per image by student predictive entropy, mean 1: the
    loss-reweighting answer (EGAD, arXiv 2605.01732; EA-KD) that moves KD signal between images without changing
    what the teacher sees. A per-image budget must beat this, or saving teacher tokens is all it adds."""
    NEEDS = {"attn": [-1], "tokens": []}

    def select(self, ctx: Context):
        h = _entropy(ctx.out.logits)
        return topk_mask(_cls_scores(ctx), int(ctx.budget)), h / h.mean().clamp_min(1e-12)


BUILTIN = {"full": Full, "maskedkd": MaskedKD, "random": Random, "grid": Grid, "center": Center, "rollout": Rollout,
           "teacher_topk": TeacherTopK, "rand_alloc": RandAlloc, "dynkd": DynKD, "sal_ratio": SalRatio,
           "kd_sched": KDSched, "ent_weight": EntWeight}
