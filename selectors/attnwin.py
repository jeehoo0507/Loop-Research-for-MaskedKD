"""H6: attention-localised window. The teacher gets the K patches nearest the centroid of the student's
attention map (a compact, contiguous region placed on the object), instead of the K highest-attention
patches (MaskedKD, scattered) or the K patches nearest the fixed image centre (`center`).

Basis: analysis_signal_fidelity_1030_014040_eb3d (a compact centre disc keeps the teacher far closer to
its full prediction than any scattered top-K signal: K=49 fid_kl center 0.905 vs random 1.281 vs MaskedKD
1.634) and analysis_object_window_1030_022311_43fc (window placement). During training the DeiT random
resized crop moves the object off-centre, which is where a student-placed window can differ from `center`.

sel_args: map = "cls11" (last-block CLS attention, the MaskedKD map) | "rollout" (Abnar & Zuidema 2020).
Draft: not queued until analysis_object_window_1030_022311_43fc and the K=49 baselines are in.
"""
import torch

from harness.selector_api import SelectorBase, topk_mask

SIDE = 14

META = {
    "method": "attention maps used for weakly-supervised object localisation (centroid of CLS attention / "
              "attention rollout), not for per-token ranking",
    "paper": "Abnar & Zuidema, Quantifying Attention Flow in Transformers, ACL 2020; Gao et al., TS-CAM, ICCV 2021",
    "why": "a masked teacher needs a compact contiguous region to stay close to its full prediction; the "
           "student's attention says where the object is, so the region is centred there",
    "vs_maskedkd": "same student attention map, but used only to place one contiguous window (centroid), "
                   "not to take the K highest-scoring scattered patches",
    "prior_art": "attention-guided cropping for fine-grained recognition (e.g. WS-DAN, TransFG part selection) "
                 "crops for the model itself; here the crop is the KD teacher's input chosen by the student",
    "parent": "center",
}


class Selector(SelectorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.map = kwargs.get("map", "cls11")
        self.NEEDS = {"attn": list(range(12)) if self.map == "rollout" else [-1], "tokens": []}

    def _map(self, ctx):
        A = [a for a in ctx.out.attn if a is not None]
        if self.map == "rollout":
            B, _, N, _ = A[0].shape
            eye = torch.eye(N, device=A[0].device).expand(B, N, N)
            R = eye.clone()
            for a in A:
                a = 0.5 * a.detach().float().mean(1) + 0.5 * eye
                R = (a / a.sum(-1, keepdim=True)) @ R
            return R[:, 0, 1:]
        return A[-1].detach().float().mean(1)[:, 0, 1:]

    def select(self, ctx):
        m = self._map(ctx)
        n = ctx.num_patches
        rr = torch.arange(n, device=m.device)
        rows, cols = (rr // SIDE).float(), (rr % SIDE).float()
        w = m / m.sum(1, keepdim=True)
        cy, cx = (w * rows).sum(1, keepdim=True), (w * cols).sum(1, keepdim=True)
        d = (rows[None] - cy) ** 2 + (cols[None] - cx) ** 2 + rr[None].float() * 1e-6
        return topk_mask(-d, int(ctx.budget))
