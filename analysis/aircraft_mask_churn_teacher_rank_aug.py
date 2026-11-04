"""H-A144 (round 414, analysis only, no training): Aircraft replication of H-A143
(analysis/cub_mask_churn_teacher_rank_aug.py, code unchanged except runs, bootstrap seed and text; on Aircraft clean
views H-A142 found mean d +0.099 [+0.084, +0.113], Spearman +0.26 [+0.19, +0.32]). Original text: H-A141 found on CUB clean val views that MaskedKD@29 drops, from e25
to final, patches the teacher's own rollout ranks higher than the patches it takes in (mean d +0.040 [+0.027, +0.052]),
and that images with larger d lose more masked-teacher signal (Spearman +0.28 [+0.20, +0.35]). Training used crop+flip
views and clean-view statistics can mislead (H-A77). Do both hold on training-style views?

Same runs, score rules, teacher map formula, d and dG as H-A141 (analysis/cub_mask_churn_teacher_rank_snap.py, inlined),
snapshots e25 and final only. Views: V = 2 fixed RandomResizedCrop+flip views per val image (harness GpuAugment formula,
numpy default_rng 20261121, as H-A77 / H-A133), no RandAugment, erasing or Mixup. On each view: teacher map = CLS row of
the teacher's rollout of that view; keep sets from the e25 and final students on that view; d_{s,v} = mean r(dropped) -
mean r(entered); G = KL(full teacher on view || teacher on view keep), all classes, T = 1. Per image: d_i = nanmean over
seeds and views; dG_i = mean over seeds and views of (G_final - G_e25).
Primary P1: mean d over images (non-NaN) for maskedkd, image bootstrap 2000, numpy default_rng(20261153), 95% CI.
P2: Spearman(d, dG) for maskedkd, same bootstrap.
Record: the same for rollout, per-seed and per-view mean d, mean r of dropped / entered, kept-set teacher rank and
overlap with the view's teacher top-29 per epoch, NaN images, masked-teacher accuracy per epoch on views.
Limit: 2 fixed views; final/e25 students, not the student of each training step; correlation, not mechanism.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on Aircraft (final, seeds 1-3, K=29), on training-style crop+flip views, do the patches MaskedKD drops from "
                "e25 to final rank higher in the teacher's own attention rollout than the patches it takes in, and does "
                "that go with signal loss?",
    "method": "per image and view mean teacher-rollout rank of dropped minus entered patches (e25 -> final top-29 kept "
              "sets) on 2 fixed RandomResizedCrop+flip views; seed/view mean per image; mean with image bootstrap; "
              "Spearman with the change of KL(full || masked teacher)",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (attention rollout); Hinton et al. 2015; "
             "Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261154
K = 29
V = 2
IMG = 224
EPOCHS = (25, None)
RUNS = {
    "maskedkd": {1: "aircraft_final_maskedkd_k29_s1_1102_132227_6542",
                 2: "aircraft_final_maskedkd_k29_s2_1102_180321_52e2",
                 3: "aircraft_final_maskedkd_k29_s3_1102_180336_347e"},
    "rollout": {1: "aircraft_final_rollout_k29_s1_1102_132227_c3d7",
                2: "aircraft_final_rollout_k29_s2_1102_180325_20b6",
                3: "aircraft_final_rollout_k29_s3_1102_180341_8606"},
}


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _rank(a):
    r = numpy.empty(len(a))
    r[numpy.argsort(a, kind="stable")] = numpy.arange(len(a))
    return r


def _spearman(a, b):
    return float(numpy.corrcoef(_rank(a), _rank(b))[0, 1])


def _topk_mask(scores, k):
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(1, scores.topk(k, dim=1).indices, True)
    return keep


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _scores(api, S, x, method):
    if method == "maskedkd":
        out = api.student_forward(S, x, attn_layers=[-1])
        return out.attn[len(S.blocks) - 1].float().mean(1)[:, 0, 1:]
    out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
    return _rollout(out.attn)


def _ename(e):
    return "final" if e is None else f"e{e}"


def _masked_mean(r, m):
    c = m.sum(1)
    s = (r * m).sum(1)
    return numpy.where(c > 0, s / numpy.maximum(c, 1), numpy.nan)


def _grids(n, device):
    """V crop+flip affine grids per val image, harness GpuAugment RandomResizedCrop formula, fixed draws (H-A77)."""
    gen = numpy.random.default_rng(20261121)
    area = gen.uniform(0.08, 1.0, (V, n))
    r = numpy.exp(gen.uniform(math.log(3 / 4), math.log(4 / 3), (V, n)))
    w = numpy.minimum(numpy.sqrt(area * r), 1.0)
    h = numpy.minimum(numpy.sqrt(area / r), 1.0)
    cx = (gen.uniform(0, 1, (V, n)) * 2 - 1) * (1 - w)
    cy = (gen.uniform(0, 1, (V, n)) * 2 - 1) * (1 - h)
    flip = numpy.where(gen.uniform(0, 1, (V, n)) < 0.5, -1.0, 1.0)
    th = numpy.zeros((V, n, 2, 3))
    th[:, :, 0, 0] = w * flip
    th[:, :, 0, 2] = cx
    th[:, :, 1, 1] = h
    th[:, :, 1, 2] = cy
    return torch.tensor(th, dtype=torch.float32, device=device)


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid]["snapshots"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    Tm = api.teacher.eval()
    xs, ys = [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
    y_t = torch.cat(ys)
    n = len(y_t)
    y = y_t.cpu().numpy().astype(int)
    th = _grids(n, y_t.device)
    views = []
    for v in range(V):
        vb, off = [], 0
        for x in xs:
            b = len(x)
            grid = F.affine_grid(th[v, off:off + b], (b, 3, IMG, IMG), align_corners=False)
            vb.append(F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros",
                                    align_corners=False).to(x.dtype))
            off += b
        views.append(vb)

    tlp, rk, ttop, t_acc = [], [], [], []
    for vb in views:
        tl, tr = [], []
        for x in vb:
            with torch.no_grad():
                tl.append(Tm(x).float().cpu())
                out = api.student_forward(Tm, x, attn_layers=list(range(len(Tm.blocks))))
                tr.append(_rollout(out.attn).float().cpu())
        L = torch.cat(tl).numpy().astype(numpy.float64)
        t_acc.append(round(100 * float((L.argmax(1) == y).mean()), 2))
        tlp.append(_logp(L))
        tmap = torch.cat(tr)
        P = tmap.shape[1]
        r_v = numpy.empty((n, P))
        order = numpy.argsort(tmap.numpy().astype(numpy.float64), axis=1, kind="stable")
        numpy.put_along_axis(r_v, order, numpy.broadcast_to(numpy.arange(P) / (P - 1), (n, P)), axis=1)
        rk.append(r_v)
        ttop.append(_topk_mask(tmap, K).numpy())

    Gkl = {a: {} for a in RUNS}
    keeps = {a: {} for a in RUNS}
    per_run = {a: {} for a in RUNS}
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            per_run[a][str(s)] = {"val_acc_last": runs[rid]["val_acc_last"]}
            for e in EPOCHS:
                S = api.student(rid, epoch=e)
                en = _ename(e)
                rec = {"signal_kl": [], "signal_acc": [], "kept_teacher_rank": [], "overlap_teacher_top29": []}
                for v, vb in enumerate(views):
                    gs, ks = [], []
                    for x in vb:
                        with torch.no_grad():
                            keep = _topk_mask(_scores(api, S, x, a), K)
                            ks.append(keep.cpu())
                            gs.append(api.teacher_forward_keepmask(Tm, x, keep).float().cpu())
                    G = torch.cat(gs).numpy().astype(numpy.float64)
                    kp = torch.cat(ks).numpy()
                    g = (numpy.exp(tlp[v]) * (tlp[v] - _logp(G))).sum(1)
                    Gkl[a].setdefault(en, {}).setdefault(s, {})[v] = g
                    keeps[a].setdefault(en, {}).setdefault(s, {})[v] = kp
                    rec["signal_kl"].append(round(float(g.mean()), 4))
                    rec["signal_acc"].append(round(100 * float((G.argmax(1) == y).mean()), 2))
                    rec["kept_teacher_rank"].append(round(float(_masked_mean(rk[v], kp).mean()), 4))
                    rec["overlap_teacher_top29"].append(round(float((kp & ttop[v]).sum(1).mean() / K), 4))
                per_run[a][str(s)][en] = rec
                del S
                torch.cuda.empty_cache()

    seeds = sorted(RUNS["maskedkd"])
    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def boot(fn):
        b = numpy.array([fn(ix) for ix in idxs])
        return [round(float(numpy.nanquantile(b, 0.025)), 4), round(float(numpy.nanquantile(b, 0.975)), 4)]

    def verdict(lo, hi):
        return "positive" if lo > 0 else ("negative" if hi < 0 else "uncertain")

    rd, re_, d_sv = {}, {}, {}
    for a in RUNS:
        rd[a], re_[a], d_sv[a] = {}, {}, {}
        for s in seeds:
            for v in range(V):
                k0, k1 = keeps[a]["e25"][s][v], keeps[a]["final"][s][v]
                rd[a][(s, v)] = _masked_mean(rk[v], k0 & ~k1)
                re_[a][(s, v)] = _masked_mean(rk[v], k1 & ~k0)
                d_sv[a][(s, v)] = rd[a][(s, v)] - re_[a][(s, v)]
    keys = [(s, v) for s in seeds for v in range(V)]
    with numpy.errstate(all="ignore"):
        d = {a: numpy.nanmean(numpy.stack([d_sv[a][k] for k in keys]), 0) for a in RUNS}
    dG = {a: numpy.mean([Gkl[a]["final"][s][v] - Gkl[a]["e25"][s][v] for s, v in keys], 0) for a in RUNS}
    ok = {a: ~numpy.isnan(d[a]) for a in RUNS}

    mean_d = {a: [round(float(numpy.nanmean(d[a])), 4)] + boot(lambda ix, a=a: numpy.nanmean(d[a][ix])) for a in RUNS}

    def sp(a, ix):
        m = ok[a][ix]
        return _spearman(d[a][ix][m], dG[a][ix][m])

    rho = {a: [round(sp(a, numpy.arange(n)), 4)] + boot(lambda ix, a=a: sp(a, ix)) for a in RUNS}
    with numpy.errstate(all="ignore"):
        per_seed = {a: [round(float(numpy.nanmean(numpy.stack([d_sv[a][(s, v)] for v in range(V)]))), 4)
                        for s in seeds] for a in RUNS}
        per_view = {a: [round(float(numpy.nanmean(numpy.stack([d_sv[a][(s, v)] for s in seeds]))), 4)
                        for v in range(V)] for a in RUNS}
        sep = {a: {"dropped": round(float(numpy.nanmean(numpy.stack([rd[a][k] for k in keys]))), 4),
                   "entered": round(float(numpy.nanmean(numpy.stack([re_[a][k] for k in keys]))), 4)} for a in RUNS}
    by_epoch = {a: {en: {f: round(float(numpy.mean([per_run[a][str(s)][en][f] for s in seeds])), 4)
                         for f in ("signal_kl", "signal_acc", "kept_teacher_rank", "overlap_teacher_top29")}
                    for en in ("e25", "final")} for a in RUNS}

    api.write_json("summary", {
        "n_val": int(n), "K": K, "V": V, "teacher_acc_views": t_acc,
        "P1_mean_d_dropped_minus_entered_maskedkd": mean_d["maskedkd"],
        "P2_spearman_d_dG_maskedkd": rho["maskedkd"],
        "mean_d": mean_d, "spearman_d_dG": rho, "per_seed_mean_d": per_seed, "per_view_mean_d": per_view,
        "mean_rank_dropped_entered": sep, "by_epoch": by_epoch,
        "n_nan_images": {a: int((~ok[a]).sum()) for a in RUNS},
        "mean_dG": {a: round(float(dG[a].mean()), 4) for a in RUNS},
        "per_run": per_run,
        "verdict": {"P1": verdict(*mean_d["maskedkd"][1:]), "P2": verdict(*rho["maskedkd"][1:])}})
