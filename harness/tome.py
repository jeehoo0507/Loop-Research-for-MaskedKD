"""Token Merging (ToMe, Bolya et al., ICLR 2023) applied to the teacher: a harness-owned baseline.

The teacher sees all 196 patch tokens and merges r similar token pairs inside each block (bipartite soft
matching on attention keys, size-weighted averaging, proportional attention), following the official
implementation (facebookresearch/ToMe, tome/merge.py and tome/patch/timm.py). No student signal is used.

`r_for_budget(K)` picks a constant per-layer r so the teacher's cost roughly matches a MaskedKD teacher
that keeps K patch tokens; the real cost is measured by the FLOP counter like every other method.
"""
import math

import torch
import torch.nn.functional as F

from . import config as C


def bipartite_soft_matching(metric, r):
    """metric [B,N,C]; CLS (index 0) is protected. Returns merge(x, mode) or identity."""
    t = metric.shape[1]
    r = min(r, (t - 1) // 2)
    if r <= 0:
        return lambda x, mode="mean": x
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        scores[..., 0, :] = -math.inf                        # never merge CLS
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :].sort(dim=1)[0]         # unmerged (keeps CLS first)
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x, mode="mean"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    return merge


def _attn_prop(attn_mod, x, size):
    """timm Attention with proportional attention (log token size added to the logits). Returns (out, key mean)."""
    B, N, Cdim = x.shape
    qkv = attn_mod.qkv(x).reshape(B, N, 3, attn_mod.num_heads, attn_mod.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
    bias = size.log()[:, None, None, :, 0]                   # [B,1,1,N]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
    out = out.transpose(1, 2).reshape(B, N, Cdim)
    out = getattr(attn_mod, "norm", torch.nn.Identity())(out)
    return attn_mod.proj_drop(attn_mod.proj(out)), k.mean(1)


def teacher_forward_tome(model, x, r):
    """Full-input teacher forward with ToMe merging r tokens per block. Returns logits."""
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    h = model.norm_pre(h)
    size = torch.ones(h.shape[0], h.shape[1], 1, device=h.device, dtype=h.dtype)
    for blk in model.blocks:
        a, metric = _attn_prop(blk.attn, blk.norm1(h), size)
        h = h + blk.drop_path1(blk.ls1(a))
        merge = bipartite_soft_matching(metric, r)
        h = merge(h * size, mode="sum")
        size = merge(size, mode="sum")
        h = h / size
        h = h + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(h))))
    h = model.norm(h)
    return model.forward_head(h)


def _cost(tokens, dim=384):
    """Per-block cost model: linear layers ~ 12*n*d^2, attention ~ 2*n^2*d (only used to choose r)."""
    return 12 * tokens * dim * dim + 2 * tokens * tokens * dim


def token_schedule(r, depth=12, n0=C.NUM_PATCHES + 1):
    out, n = [], n0
    for _ in range(depth):
        out.append(n)
        n -= min(r, (n - 1) // 2)
    return out


def r_for_budget(keep, depth=12):
    """Smallest r whose modelled teacher cost is <= a masked teacher keeping `keep` patch tokens."""
    target = depth * _cost(int(keep) + 1)
    for r in range(0, C.NUM_PATCHES):
        if sum(_cost(n) for n in token_schedule(r, depth)) <= target:
            return r
    return C.NUM_PATCHES // 2
