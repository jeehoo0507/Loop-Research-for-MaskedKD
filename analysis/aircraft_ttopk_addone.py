"""H-A152 (round 418, analysis only, no training): Aircraft replication of H-A151 (analysis/cub_lt_ttopk_addone.py;
code unchanged except runs, bootstrap seed 20261160 and text). Original text: H-A145..H-A150 found that the teacher's own rollout top-29 (T29) keeps
less of the full teacher's non-target signal than the final Rollout@29 student's rollout top-29 (R29) (dG +0.62 cub_lt,
+0.28 Aircraft), that the set difference 1 - Jaccard tracks dG even after controlling for spread (partial Spearman
+0.28, +0.11), and that the teacher-only patches are not the teacher's high-norm sinks (H-A149/H-A150). Open question:
is the loss carried by the individual patches (each teacher-only patch adds less signal than each student-only patch),
or only by the set as a whole (interaction)? H-A85 added whole difference sets to the intersection on Waterbirds
online/frozen maskedkd; no analysis has measured single-patch marginals for T29 vs R29.

Per image (clean val) and seed s: I = T29 & R29_s, D_T = T29 minus R29_s, D_R = R29_s minus T29 (same size).
For every patch p in D_T or D_R: G(I + p) = non-target KL(full teacher || teacher on I + {p}), T=1 (keep-only).
dM_{i,s} = mean_{p in D_T} G(I + p) - mean_{p in D_R} G(I + p) (positive = a teacher-only patch alone restores less
signal); image skipped for seed s if D is empty. dM_i = mean over seeds with a value. dG_i = G_T29 - mean_s G_R29_s.
P1: mean dM over images with a value, image bootstrap 2000, numpy default_rng(20261159), 95% quantile CI.
P2: Spearman(dM, dG) over the same images, same bootstrap.
Record: mean |I| and |D|, mean G(I) (|I| > 0), per seed the share of images with dM > 0 and mean G(I + p) for each
side, the within-D_T Spearman of teacher rollout score vs G(I + p) and the within-D_R Spearman of student rollout
score vs G(I + p) (mean over image-seed pairs with |D| >= 3), per-seed t of dM.
Limit: clean val views, final students; correlational across images; no selector is built from this.
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final Rollout@29, seeds 1-3), does each patch only in the teacher's own rollout top-29 "
                "restore less of the full teacher's non-target signal, when added alone to the shared patches, than "
                "each patch only in the student's rollout top-29, and does this single-patch gap track the set-level "
                "signal loss across images?",
    "method": "teacher and student attention rollout maps on clean val; top-29 sets, intersection and differences; "
              "keep-only masked teacher on intersection plus one patch; non-target KL(full teacher || masked teacher); "
              "image bootstrap; Spearman; seed t CI",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Hinton et al. 2015 (dark knowledge); Zhao et al. CVPR 2022 "
             "(DKD, NCKD); Son et al. ECCV 2024 (MaskedKD); Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
CHUNK = 256
N_BOOT = 2000
SEED = 20261160
K = 29
RUNS = {1: "aircraft_final_rollout_k29_s1_1102_132227_c3d7",
        2: "aircraft_final_rollout_k29_s2_1102_180325_20b6",
        3: "aircraft_final_rollout_k29_s3_1102_180341_8606"}


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _nt_logp(logp, y):
    lp = logp.copy()
    lp[numpy.arange(len(y)), y] = -numpy.inf
    m = lp.max(1, keepdims=True)
    return lp - (m + numpy.log(numpy.exp(lp - m).sum(1, keepdims=True)))


def _nt_kl(tn, sn):
    pt = numpy.exp(tn)
    return numpy.where(pt > 0, pt * (tn - numpy.where(numpy.isfinite(sn), sn, 0.0)), 0.0).sum(1)


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


def _forward_rows(api, Tm, X, img, keeps):
    """X: all val images [N, ...] on device; img: image index per row; keeps: bool [rows, 196]. Returns logits."""
    out = []
    for a in range(0, len(img), CHUNK):
        ix = torch.as_tensor(img[a:a + CHUNK], device=X.device)
        kp = torch.as_tensor(keeps[a:a + CHUNK], device=X.device)
        with torch.no_grad():
            out.append(api.teacher_forward_keepmask(Tm, X[ix], kp).float().cpu())
    return torch.cat(out).numpy().astype(numpy.float64)


def run(api):
    runs = api.runs()
    api.write_json("students", {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                         runs[rid]["seed"], runs[rid]["val_acc_last"]] for s, rid in sorted(RUNS.items())})
    Tm = api.teacher.eval()
    depth = len(Tm.blocks)
    xs, ys, tl, tr = [], [], [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
            out = api.student_forward(Tm, x, attn_layers=list(range(depth)))
            tr.append(_rollout(out.attn).float().cpu())
    X = torch.cat(xs)
    y = torch.cat(ys).cpu().numpy().astype(int)
    n = len(y)
    tn = _nt_logp(_logp(torch.cat(tl).numpy().astype(numpy.float64)), y)
    rT = torch.cat(tr)
    kT = _topk_mask(rT, K).numpy()
    rT = rT.numpy()
    allimg = numpy.arange(n)
    G_T = _nt_kl(tn, _nt_logp(_logp(_forward_rows(api, Tm, X, allimg, kT)), y))

    seeds = sorted(RUNS)
    dMs, G_R, per_seed, nI_all, nD_all = {}, {}, {}, [], []
    rhoT_all, rhoR_all, GI_all = [], [], []
    for s in seeds:
        S = api.student(RUNS[s])
        maps = []
        for a in range(0, n, BATCH):
            with torch.no_grad():
                out = api.student_forward(S, X[a:a + BATCH], attn_layers=list(range(len(S.blocks))))
                maps.append(_rollout(out.attn).float().cpu())
        del S
        torch.cuda.empty_cache()
        rR = torch.cat(maps)
        kR = _topk_mask(rR, K).numpy()
        rR = rR.numpy()
        G_R[s] = _nt_kl(tn, _nt_logp(_logp(_forward_rows(api, Tm, X, allimg, kR)), y))
        kI = kT & kR
        nI = kI.sum(1)
        okI = nI > 0
        GI = numpy.full(n, numpy.nan)
        if okI.any():
            GI[okI] = _nt_kl(tn[okI], _nt_logp(_logp(_forward_rows(api, Tm, X, allimg[okI], kI[okI])), y[okI]))
        rows_img, rows_p, rows_side = [], [], []
        for i in range(n):
            for p in numpy.flatnonzero(kT[i] & ~kR[i]):
                rows_img.append(i); rows_p.append(p); rows_side.append(0)
            for p in numpy.flatnonzero(kR[i] & ~kT[i]):
                rows_img.append(i); rows_p.append(p); rows_side.append(1)
        rows_img = numpy.array(rows_img, dtype=int)
        rows_p = numpy.array(rows_p, dtype=int)
        rows_side = numpy.array(rows_side, dtype=int)
        keeps = kI[rows_img].copy()
        keeps[numpy.arange(len(rows_img)), rows_p] = True
        Gp = _nt_kl(tn[rows_img], _nt_logp(_logp(_forward_rows(api, Tm, X, rows_img, keeps)), y[rows_img]))
        dM = numpy.full(n, numpy.nan)
        rhoT, rhoR = [], []
        for i in range(n):
            sel = rows_img == i
            gT, gR = Gp[sel & (rows_side == 0)], Gp[sel & (rows_side == 1)]
            if len(gT) == 0:
                continue
            dM[i] = gT.mean() - gR.mean()
            if len(gT) >= 3:
                pT = rows_p[sel & (rows_side == 0)]
                pR = rows_p[sel & (rows_side == 1)]
                a1, a2 = _spearman(rT[i, pT], gT), _spearman(rR[i, pR], gR)
                if numpy.isfinite(a1):
                    rhoT.append(a1)
                if numpy.isfinite(a2):
                    rhoR.append(a2)
        dMs[s] = dM
        nI_all.append(nI.mean())
        nD_all.append(K - nI.mean())
        rhoT_all += rhoT
        rhoR_all += rhoR
        GI_all.append(float(numpy.nanmean(GI)))
        ok = numpy.isfinite(dM)
        per_seed[str(s)] = {"n_images_with_diff": int(ok.sum()), "mean_I": round(float(nI.mean()), 3),
                            "dM": round(float(numpy.nanmean(dM)), 4),
                            "share_dM_pos": round(float((dM[ok] > 0).mean()), 4),
                            "mean_G_I_plus_DT_patch": round(float(Gp[rows_side == 0].mean()), 4),
                            "mean_G_I_plus_DR_patch": round(float(Gp[rows_side == 1].mean()), 4),
                            "mean_G_I": round(float(numpy.nanmean(GI)), 4),
                            "dG": round(float((G_T - G_R[s]).mean()), 4),
                            "n_rows": int(len(rows_img))}

    dM = numpy.nanmean(numpy.stack([dMs[s] for s in seeds]), 0)
    ok = numpy.isfinite(dM)
    dG = G_T - numpy.mean([G_R[s] for s in seeds], 0)

    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def boot(fn):
        b = numpy.array([fn(ix) for ix in idxs])
        return [round(float(numpy.nanquantile(b, 0.025)), 4), round(float(numpy.nanquantile(b, 0.975)), 4)]

    def verdict(lo, hi):
        return "positive" if lo > 0 else ("negative" if hi < 0 else "uncertain")

    def sub(ix):
        return ix[ok[ix]]

    P1 = [round(float(dM[ok].mean()), 4)] + boot(lambda ix: dM[sub(ix)].mean())
    P2 = [round(_spearman(dM[ok], dG[ok]), 4)] + boot(lambda ix: _spearman(dM[sub(ix)], dG[sub(ix)]))
    api.write_json("summary", {
        "n_val": int(n), "K": K, "n_images_P1": int(ok.sum()),
        "P1_mean_dM_singlepatch_DT_minus_DR": P1,
        "P2_spearman_dM_dG": P2,
        "record": {
            "mean_I": round(float(numpy.mean(nI_all)), 3), "mean_D": round(float(numpy.mean(nD_all)), 3),
            "mean_G_I": round(float(numpy.mean(GI_all)), 4),
            "mean_dG": [round(float(dG.mean()), 4)] + boot(lambda ix: dG[ix].mean()),
            "within_DT_spearman_teacher_rollout_vs_G": [round(float(numpy.mean(rhoT_all)), 4), len(rhoT_all)],
            "within_DR_spearman_student_rollout_vs_G": [round(float(numpy.mean(rhoR_all)), 4), len(rhoR_all)],
            "signal_nt_kl_T": round(float(G_T.mean()), 4),
            "per_seed_t_dM": _t95([per_seed[str(s)]["dM"] for s in seeds]),
        },
        "per_seed": per_seed,
        "verdict": {"P1": verdict(*P1[1:]), "P2": verdict(*P2[1:])}})
