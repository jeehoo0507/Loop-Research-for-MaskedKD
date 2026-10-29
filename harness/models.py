"""Student forward that exposes attention/tokens to selectors, and the masked teacher forward.

The masked teacher forward follows the official MaskedKD code (models_teacher.py, commit 96d052d):
patch_embed -> concat CLS -> add pos_embed (all 197) -> gather kept patch tokens -> blocks -> norm -> CLS head.
tests/test_equivalence.py checks this against plain timm forwards.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import timm
import torch
import torch.nn as nn

from . import config as C


def make_student():
    return timm.create_model(C.STUDENT, pretrained=False, num_classes=C.NUM_CLASSES, drop_path_rate=C.DROP_PATH)


def make_teacher(pretrained=True):
    return timm.create_model(C.TEACHER, pretrained=pretrained, num_classes=C.NUM_CLASSES, drop_path_rate=C.DROP_PATH)


def _attn_with_weights(attn_mod, x):
    """timm Attention.forward with explicit softmax weights. Returns (out, weights[B,H,N,N])."""
    B, N, Cdim = x.shape
    qkv = attn_mod.qkv(x).reshape(B, N, 3, attn_mod.num_heads, attn_mod.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
    w = ((q * attn_mod.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    out = attn_mod.attn_drop(w) @ v
    out = out.transpose(1, 2).reshape(B, N, Cdim)
    out = getattr(attn_mod, "norm", nn.Identity())(out)
    out = attn_mod.proj_drop(attn_mod.proj(out))
    return out, w


@dataclass
class StudentOut:
    logits: torch.Tensor
    attn: List[Optional[torch.Tensor]] = field(default_factory=list)   # per block, None if not requested
    tokens: List[Optional[torch.Tensor]] = field(default_factory=list)  # per block output, None if not requested


def student_forward(model, x, attn_layers=(), token_layers=()):
    """attn_layers / token_layers: block indices whose attention weights / outputs are kept.

    Blocks whose attention is requested run the explicit-softmax path (same math as timm's
    non-fused path); the rest run timm's normal (fused) block.
    """
    depth = len(model.blocks)
    attn_layers = {i % depth for i in attn_layers}
    token_layers = {i % depth for i in token_layers}
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    h = model.patch_drop(h)
    h = model.norm_pre(h)
    attn, toks = [None] * depth, [None] * depth
    for i, blk in enumerate(model.blocks):
        if i in attn_layers:
            a, w = _attn_with_weights(blk.attn, blk.norm1(h))
            h = h + blk.drop_path1(blk.ls1(a))
            h = h + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(h))))
            attn[i] = w
        else:
            h = blk(h)
        if i in token_layers:
            toks[i] = h
    h = model.norm(h)
    return StudentOut(logits=model.forward_head(h), attn=attn, tokens=toks)


def teacher_forward_masked(model, x, keep_idx):
    """keep_idx: LongTensor [B, K] of patch indices in [0, 196). Returns logits [B, classes]."""
    h = model.patch_embed(x)
    h = model._pos_embed(h)            # CLS concat + pos_embed on all tokens (as in official MaskedKD)
    cls, patches = h[:, :1], h[:, 1:]
    D = patches.shape[-1]
    patches = torch.gather(patches, 1, keep_idx.unsqueeze(-1).expand(-1, -1, D))
    h = torch.cat([cls, patches], dim=1)
    h = model.patch_drop(h)
    h = model.norm_pre(h)
    h = model.blocks(h)
    h = model.norm(h)
    return model.forward_head(h)


def teacher_forward_keepmask(model, x, keep):
    """keep: bool [B, 196]. Samples are grouped by their kept count so each group runs exactly
    its own number of tokens (no padding, so FLOPs are the real cost). Returns logits [B, classes]."""
    counts = keep.sum(1)
    out = None
    for k in counts.unique().tolist():
        rows = (counts == k).nonzero().flatten()
        # indices of kept patches, in ascending spatial order
        idx = keep[rows].float().argsort(dim=1, descending=True, stable=True)[:, :k]
        logits = teacher_forward_masked(model, x[rows], idx)
        if out is None:
            out = logits.new_empty(x.shape[0], logits.shape[1])
        out[rows] = logits
    return out
