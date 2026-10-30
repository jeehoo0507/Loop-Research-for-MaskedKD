"""Pre-promotion decomposition of gradroll (H-W16) at K=12, as required before screen -> final.

screen_gradroll_k12_s0_1030_184259_e2cd: 88.16 / WGA 60.90, vs parent rollout@12 87.24 / 45.11 and center@12
88.74 / 56.39 (screen seed 0). At K=29 gradroll did not beat rollout (63.91 vs 65.41). Questions:
  overlap   how much of gradroll's top-K is MaskedKD's (block-11 CLS attention), plain rollout's, center's
            (share of K), per training snapshot (e25 / e50 / final) and per Waterbirds group
  spatial   mean distance of the kept patches from the image centre (cdist; center@12 is the floor)
  register  share of kept patches that are block-10 high-norm outliers (Darcet et al. 2024, z > 2)
  fidelity  masked-teacher group accuracy and worst-group KL(full || masked) for every selection, including
            the replacement variants grad_rep_rand / grad_rep_grid: gradroll's tokens that MaskedKD also picks
            are kept and the rest are refilled with deterministic pseudo-random / raster-grid patches
Students: gradroll@12 (e25, e50, final), rollout@12 and center@12 (final), all screen seed 0; plus gradroll@29
and rollout@29 finals at K=29 for the case where the gradient term did not help.
Teacher-side numbers are context only: the faithfulness proxy failed out of sample (H-W13/H-W15).
"""
import torch

META = {
    "question": "which tokens does gradient-weighted rollout keep that plain rollout, MaskedKD and the centre "
                "disc do not, and does its K=12 gain come from those tokens, from spatial placement, or from "
                "avoiding register-like high-norm tokens?",
    "method": "gradient-weighted attention rollout (relevance propagation) vs attention rollout vs last-block "
              "CLS attention; selection overlap, centre distance, high-norm token share, keep-only evaluation",
    "paper": "Chefer, Gur & Wolf CVPR 2021; Abnar & Zuidema ACL 2020; Darcet et al. ICLR 2024 (registers); "
             "Hooker et al. NeurIPS 2019 (keep-only evaluation)",
}

SIDE = 14
Z = 2.0


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _scores(api, student, x):
    """MaskedKD (block-11 CLS attn), plain rollout, gradient-weighted rollout, block-10 token norm z-score."""
    depth = len(student.blocks)
    with torch.enable_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = api.student_forward(student, x, attn_layers=list(range(depth)), token_layers=[10])
        target = out.logits.float().max(1).values.sum()
        grads = torch.autograd.grad(target, list(out.attn))
    A = [a.detach().float() for a in out.attn]
    n = A[0].shape[-1]
    eye = torch.eye(n, device=x.device)
    R = None
    G = eye.unsqueeze(0).expand(x.shape[0], n, n)
    for a, g in zip(A, grads):
        am = a.mean(1)
        am = 0.5 * am + 0.5 * eye
        am = am / am.sum(-1, keepdim=True)
        R = am if R is None else am @ R
        G = G + (g.detach().float() * a).clamp(min=0).mean(1) @ G
    tn = out.tokens[10].detach().float()[:, 1:].norm(dim=-1)
    z = (tn - tn.mean(1, keepdim=True)) / tn.std(1, keepdim=True).clamp(min=1e-6)
    return A[-1].mean(1)[:, 0, 1:], R[:, 0, 1:], G[:, 0, 1:], z


def _fill(base, fill_scores, k):
    """keep `base` patches, add the highest-scoring others until k."""
    s = fill_scores.clone()
    s[base] = float("inf")
    return _topk(s, k)


def run(api):
    runs = api.runs()

    def find(method, k):
        for rid, r in sorted(runs.items()):
            if (r["tier"] == "screen" and r["seed"] == 0 and r["method"] == method and r["keep"] is not None
                    and int(r["keep"]) == k and r.get("dataset", "waterbirds") == "waterbirds"):
                return rid, r["snapshots"]
        return None, []

    plan = []
    g12, snaps = find("gradroll", 12)
    for e in snaps:
        plan.append((f"gradroll12_e{e}", g12, e, 12))
    plan.append(("gradroll12_final", g12, None, 12))
    for m, k in (("rollout", 12), ("center", 12), ("gradroll", 29), ("rollout", 29)):
        rid, _ = find(m, k)
        if rid:
            plan.append((f"{m}{k}_final", rid, None, k))
    api.write_json("students", plan)

    groups = api.val_groups()
    T = api.teacher.eval()
    n = api.num_patches
    dev = api.device
    rr = torch.arange(n, device=dev)
    rows_, cols_ = (rr // SIDE).float(), (rr % SIDE).float()
    c = (SIDE - 1) / 2
    dist = ((rows_ - c) ** 2 + (cols_ - c) ** 2).sqrt()
    center_s = -(dist + rr.float() * 1e-6)

    out = {}
    for sname, rid, ep, k in plan:
        st = api.student(rid, ep)
        gidx = torch.linspace(0, n - 1, k, device=dev).round().long()
        grid_s = torch.zeros(n, device=dev)
        grid_s[gidx] = 1.0
        names = ("maskedkd", "rollout", "gradroll", "center", "grad_rep_rand", "grad_rep_grid")
        acc = {m: [] for m in names}
        kl = {m: [] for m in names}
        cd = {m: 0.0 for m in names}
        hn = {m: 0.0 for m in names}
        ov = {"grad_mkd": [], "grad_roll": [], "grad_center": [], "roll_mkd": []}
        G_all = []
        tot = 0
        off = 0
        for x, y in api.val_batches(batch=25):
            B = x.shape[0]
            gb = groups[off:off + B]
            img = torch.arange(off, off + B, device=dev)
            off += B
            s_mkd, s_roll, s_grad, z = _scores(api, st, x)
            sel = {"maskedkd": _topk(s_mkd, k), "rollout": _topk(s_roll, k), "gradroll": _topk(s_grad, k),
                   "center": _topk(center_s.expand(B, n), k)}
            base = sel["gradroll"] & sel["maskedkd"]
            prand = ((rr[None, :] * 7919 + img[:, None] * 104729) % 9973).float()
            sel["grad_rep_rand"] = _fill(base, prand, k)
            sel["grad_rep_grid"] = _fill(base, grid_s.expand(B, n) + 1e-6 * prand, k)
            ov["grad_mkd"].append(base.sum(1).float().cpu() / k)
            ov["grad_roll"].append((sel["gradroll"] & sel["rollout"]).sum(1).float().cpu() / k)
            ov["grad_center"].append((sel["gradroll"] & sel["center"]).sum(1).float().cpu() / k)
            ov["roll_mkd"].append((sel["rollout"] & sel["maskedkd"]).sum(1).float().cpu() / k)
            with torch.no_grad():
                full = T(x).float().log_softmax(-1)
                for m in names:
                    keep = sel[m]
                    lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    acc[m].append((lm.argmax(-1) == y).cpu())
                    kl[m].append((full.exp() * (full - lm)).sum(-1).cpu())
                    cd[m] += (keep.float() * dist).sum().item() / k
                    hn[m] += (keep & (z > Z)).sum().item() / k
            G_all.append(gb.cpu())
            tot += B
        Gc = torch.cat(G_all)
        row = {"run": rid, "epoch": ep, "K": k}
        for m in names:
            a, kk = torch.cat(acc[m]).float(), torch.cat(kl[m])
            ga = [round(100 * a[Gc == gi].mean().item(), 2) for gi in range(4)]
            gk = [round(kk[Gc == gi].mean().item(), 4) for gi in range(4)]
            row[m] = {"t_acc": round(100 * a.mean().item(), 2), "t_wga": min(ga), "t_acc_g": ga,
                      "kl_worst": max(gk), "kl_g": gk, "cdist": round(cd[m] / tot, 3), "hn_frac": round(hn[m] / tot, 4)}
        for o, v in ov.items():
            v = torch.cat(v)
            row["overlap_" + o] = round(v.mean().item(), 4)
            row["overlap_" + o + "_g"] = [round(v[Gc == gi].mean().item(), 4) for gi in range(4)]
        out[sname] = row
        api.write_json("rows", out)
        del st
    api.write_json("summary", {
        "note": "t_* = masked teacher on clean val with that student's selection; overlap = share of K; "
                "cdist = mean patch distance from the centre (patch units); hn_frac = share of kept patches that "
                "are block-10 norm outliers (z>2); grad_rep_* keep gradroll&maskedkd and refill the rest",
        "table": {s: {m: [r[m]["t_wga"], r[m]["kl_worst"], r[m]["cdist"], r[m]["hn_frac"]]
                      for m in ("maskedkd", "rollout", "gradroll", "center", "grad_rep_rand", "grad_rep_grid")}
                  for s, r in out.items()},
        "table_cols": ["t_wga", "kl_worst", "cdist", "hn_frac"],
        "overlap": {s: [r["overlap_grad_mkd"], r["overlap_grad_roll"], r["overlap_grad_center"],
                        r["overlap_roll_mkd"]] for s, r in out.items()},
        "overlap_cols": ["grad_mkd", "grad_roll", "grad_center", "roll_mkd"],
    })
