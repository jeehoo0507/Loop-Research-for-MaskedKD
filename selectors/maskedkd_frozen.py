"""MaskedKD with a frozen selector (QUEUE 2026-11-04, gate 3): the K teacher tokens are chosen by the *initial*
student (its ImageNet-pretrained weights, snapshotted at the first step), never by the student being trained.

Same score as MaskedKD (last block, head-mean CLS attention over patches, top-K), same K; the only difference is
whose attention: online MaskedKD uses the student at step t, this one the student at step 0. If online MaskedKD
loses worst-group accuracy and this does not, the loss comes from the feedback loop (the student's current
shortcut steering what the teacher sees), not from the pretrained student's attention as such. The snapshot
forward (DeiT-Ti, about 1.3 GFLOPs/img) is counted as selector cost. The snapshot is run through the student
module with torch.func.functional_call, so the training student is never modified; drop-path follows the module's
train mode (the same stochastic branches MaskedKD's own forward has).
"""
import torch

from harness.selector_api import SelectorBase, topk_mask

META = {
    "method": "MaskedKD's last-block CLS attention top-K, computed by a frozen snapshot of the initial student "
              "instead of the student being trained",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Arazo et al. IJCNN 2020 (confirmation bias in self-training)",
    "why": "separates two explanations of MaskedKD's minority-group loss at low K: the pretrained student's "
           "attention is already biased (frozen loses too) vs the loop student->mask->teacher->student amplifies "
           "the bias during training (only online loses)",
    "vs_maskedkd": "identical scoring rule and K; the attention comes from the step-0 student, so the mask cannot "
                   "follow the training student's errors",
    "prior_art": "none for teacher-input masking; frozen-teacher / off-policy variants exist for LLM distillation",
    "parent": "maskedkd",
}


class Selector(SelectorBase):
    NEEDS = {"attn": [], "tokens": []}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.snapshot = None

    def _params(self, prefix):
        pre = prefix + "."
        return {k[len(pre):]: v for k, v in self.snapshot.items() if k.startswith(pre)}

    @torch.no_grad()
    def select(self, ctx):
        m = ctx.student
        if self.snapshot is None:
            self.snapshot = {k: v.detach().clone() for k, v in m.named_parameters()}
            for k, v in m.named_buffers():
                self.snapshot[k] = v.detach().clone()
        call = torch.func.functional_call
        h = call(m.patch_embed, self._params("patch_embed"), (ctx.images,))
        cls = self.snapshot["cls_token"].expand(h.shape[0], -1, -1)
        h = torch.cat((cls, h), dim=1) + self.snapshot["pos_embed"]   # timm _pos_embed with the step-0 tokens
        h = m.norm_pre(h)
        depth = len(m.blocks)
        for i in range(depth - 1):
            h = call(m.blocks[i], self._params(f"blocks.{i}"), (h,))
        blk, pre = m.blocks[depth - 1], f"blocks.{depth - 1}"
        x = call(blk.norm1, self._params(pre + ".norm1"), (h,))
        B, N, D = x.shape
        qkv = call(blk.attn.qkv, self._params(pre + ".attn.qkv"), (x,))
        qkv = qkv.reshape(B, N, 3, blk.attn.num_heads, blk.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k = blk.attn.q_norm(qkv[0]), blk.attn.k_norm(qkv[1])
        w = ((q * blk.attn.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        scores = w.float().mean(dim=1)[:, 0, 1:]
        return topk_mask(scores, int(ctx.budget))
