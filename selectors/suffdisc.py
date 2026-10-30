"""H-W11: student-sufficiency disc. The teacher gets a disc of K patches; its position is the one, among the centre
disc and its shifts by up to `r` patches, whose kept tokens best preserve the *student's own* prediction
(smallest KL(student full || student on the disc only)).

Basis: analysis_proxy_validation_1030_170914_4668 (masked-teacher KL to the full teacher on the worst group ranks
student WGA, |Spearman| 0.94 over 16 runs) and analysis_window_fidelity_1030_165418_e276 (moving the centre disc by
<= 2 patches to the teacher-best position cuts that KL at K=12 from 0.694 to 0.049; attention centroids did not
find those positions). The sufficiency search is the student-side, label-free stand-in for that teacher search.
Verified first by analysis_student_sufficiency_1030_173025_6f96.

Cost: (2r+1)^2 student forwards on K tokens per image, no grad. ctx.student stays in train mode (drop path),
which the selection inherits as noise.

sel_args: r = max shift in patches (default 2).
"""
import torch

from harness.selector_api import SelectorBase

META = {
    "method": "keep-only sufficiency faithfulness of a token subset (ERASER sufficiency / RISE insertion), "
              "evaluated on the student to place the teacher's input window",
    "paper": "DeYoung et al. ACL 2020 (ERASER); Petsiuk, Das & Saenko BMVC 2018 (RISE); Hooker et al. NeurIPS 2019 "
             "(ROAR)",
    "why": "the teacher is most faithful on the minority groups when it sees one compact region on the object; the "
           "region where the student's own prediction is sufficient is a label-free estimate of where that is",
    "vs_maskedkd": "no attention ranking: a contiguous disc placed by the student's output sufficiency, not the "
                   "last block's scattered CLS-attention top-K",
    "prior_art": "center (fixed disc, no signal); MaskedKD/rollout rank tokens by attention; RISE/ERASER score a "
                 "given mask for explanation only; none use sufficiency to choose a KD teacher's input",
    "parent": "center",
}

SIDE = 14


def _masked_logits(model, x, idx):
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    cls, patches = h[:, :1], h[:, 1:]
    D = patches.shape[-1]
    patches = torch.gather(patches, 1, idx.unsqueeze(-1).expand(-1, -1, D))
    h = torch.cat([cls, patches], dim=1)
    h = model.norm_pre(h)
    h = model.blocks(h)
    h = model.norm(h)
    return model.forward_head(h)


class Selector(SelectorBase):
    NEEDS = {"attn": [], "tokens": []}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.r = int(kwargs.get("r", 2))

    def select(self, ctx):
        x = ctx.images
        B, dev = x.shape[0], x.device
        k = int(ctx.budget)
        n = ctx.num_patches
        rr = torch.arange(n, device=dev)
        rows, cols = (rr // SIDE).float(), (rr % SIDE).float()
        c = (SIDE - 1) / 2
        tie = rr.float() * 1e-6
        with torch.no_grad():
            full = ctx.out.logits.detach().float().log_softmax(-1)
            best_kl = torch.full((B,), float("inf"), device=dev)
            best_idx = None
            for dy in range(-self.r, self.r + 1):
                for dx in range(-self.r, self.r + 1):
                    d = (rows - (c + dy)) ** 2 + (cols - (c + dx)) ** 2 + tie
                    idx = (-d).topk(k).indices.sort().values.expand(B, -1)
                    lp = _masked_logits(ctx.student, x, idx).float().log_softmax(-1)
                    kl = (full.exp() * (full - lp)).sum(-1)
                    better = kl < best_kl
                    best_kl = torch.where(better, kl, best_kl)
                    best_idx = idx.clone() if best_idx is None else torch.where(better[:, None], idx, best_idx)
        return torch.zeros(B, n, dtype=torch.bool, device=dev).scatter_(1, best_idx, True)
