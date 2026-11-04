"""H-A142 (round 413, analysis only, no training): Aircraft replication of H-A141
(analysis/cub_mask_churn_teacher_rank_snap.py, code unchanged except runs, bootstrap seed and text; on Aircraft H-A140
found Spearman(churn, dG) +0.10 [+0.02, +0.17]). Original text: H-A139 found that images whose MaskedKD@29 kept set changes more from
e25 to final lose more masked-teacher signal (Spearman +0.12 [+0.04, +0.20]), but not which patches move. Does MaskedKD
drop patches the teacher itself ranks high and take in patches it ranks low?

Runs = H-A139 (maskedkd@29, rollout@29, seeds 1-3, snapshots e25, e50, final), clean val views, same score rules.
Teacher map = CLS row of the teacher's own attention rollout of the whole image (0.5 A + 0.5 I, same formula as
harness/teacher_cache.py, i.e. the teacher_topk map without the crop); r_j = within-image rank of patch j, scaled to
[0, 1] (1 = highest). Per image, run: dropped = keep_e25 & ~keep_final, entered = keep_final & ~keep_e25;
d_s = mean r(dropped) - mean r(entered) (NaN if no patch moved); d_i = nanmean over seeds.
dG_i = mean_s (G_final - G_e25), G = KL(full teacher || teacher on keep), all classes, T = 1 (as H-A139).
Primary P1: mean d over images (non-NaN) for maskedkd, image bootstrap 2000, numpy default_rng(20261151), 95% CI.
P2: Spearman(d, dG) for maskedkd, same bootstrap.
Record: the same for rollout, per-seed mean d, mean teacher rank of the kept set and overlap with the teacher top-29
per epoch, mean r of dropped / entered separately, number of images with NaN.
Limit: snapshots 25 epochs apart; teacher rollout is one view of teacher importance; correlation, not mechanism.
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3, K=29), do the patches MaskedKD drops from e25 to final rank higher in the "
                "teacher's own attention rollout than the patches it takes in, and does that go with signal loss?",
    "method": "per image mean teacher-rollout rank of dropped minus entered patches (e25 -> final top-29 kept sets); "
              "seed-mean per image; mean with image bootstrap; Spearman with the change of KL(full || masked teacher)",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (attention rollout); Hinton et al. 2015; "
             "Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261152
K = 29
EPOCHS = (25, 50, None)
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


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid]["snapshots"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    Tm = api.teacher.eval()
    batches, tl, tr = [], [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
            out = api.student_forward(Tm, x, attn_layers=list(range(len(Tm.blocks))))
            tr.append(_rollout(out.attn).float().cpu())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    tlp = _logp(torch.cat(tl).numpy().astype(numpy.float64))
    tmap = torch.cat(tr)
    n, P = tmap.shape
    rk = numpy.empty((n, P))
    order = numpy.argsort(tmap.numpy().astype(numpy.float64), axis=1, kind="stable")
    numpy.put_along_axis(rk, order, numpy.broadcast_to(numpy.arange(P) / (P - 1), (n, P)), axis=1)
    ttop = _topk_mask(tmap, K).numpy()

    Gkl = {a: {} for a in RUNS}
    keeps = {a: {} for a in RUNS}
    per_run = {a: {} for a in RUNS}
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            per_run[a][str(s)] = {"val_acc_last": runs[rid]["val_acc_last"]}
            for e in EPOCHS:
                S = api.student(rid, epoch=e)
                gs, ks = [], []
                for x, _ in batches:
                    with torch.no_grad():
                        keep = _topk_mask(_scores(api, S, x, a), K)
                        ks.append(keep.cpu())
                        gs.append(api.teacher_forward_keepmask(Tm, x, keep).float().cpu())
                del S
                torch.cuda.empty_cache()
                G = torch.cat(gs).numpy().astype(numpy.float64)
                kp = torch.cat(ks).numpy()
                en = _ename(e)
                g = (numpy.exp(tlp) * (tlp - _logp(G))).sum(1)
                Gkl[a].setdefault(en, {})[s] = g
                keeps[a].setdefault(en, {})[s] = kp
                per_run[a][str(s)][en] = {
                    "signal_kl": round(float(g.mean()), 4),
                    "signal_acc": round(100 * float((G.argmax(1) == y).mean()), 2),
                    "kept_teacher_rank": round(float(_masked_mean(rk, kp).mean()), 4),
                    "overlap_teacher_top29": round(float((kp & ttop).sum(1).mean() / K), 4)}

    seeds = sorted(RUNS["maskedkd"])
    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def boot(fn):
        b = numpy.array([fn(ix) for ix in idxs])
        return [round(float(numpy.nanquantile(b, 0.025)), 4), round(float(numpy.nanquantile(b, 0.975)), 4)]

    def verdict(lo, hi):
        return "positive" if lo > 0 else ("negative" if hi < 0 else "uncertain")

    rd, re_, d_s = {}, {}, {}
    for a in RUNS:
        rd[a], re_[a], d_s[a] = {}, {}, {}
        for s in seeds:
            k0, k1 = keeps[a]["e25"][s], keeps[a]["final"][s]
            rd[a][s] = _masked_mean(rk, k0 & ~k1)
            re_[a][s] = _masked_mean(rk, k1 & ~k0)
            d_s[a][s] = rd[a][s] - re_[a][s]
    with numpy.errstate(all="ignore"):
        d = {a: numpy.nanmean(numpy.stack([d_s[a][s] for s in seeds]), 0) for a in RUNS}
    dG = {a: numpy.mean([Gkl[a]["final"][s] - Gkl[a]["e25"][s] for s in seeds], 0) for a in RUNS}
    ok = {a: ~numpy.isnan(d[a]) for a in RUNS}

    mean_d = {a: [round(float(numpy.nanmean(d[a])), 4)] + boot(lambda ix, a=a: numpy.nanmean(d[a][ix])) for a in RUNS}

    def sp(a, ix):
        m = ok[a][ix]
        return _spearman(d[a][ix][m], dG[a][ix][m])

    rho = {a: [round(sp(a, numpy.arange(n)), 4)] + boot(lambda ix, a=a: sp(a, ix)) for a in RUNS}
    per_seed = {a: [round(float(numpy.nanmean(d_s[a][s])), 4) for s in seeds] for a in RUNS}
    sep = {a: {"dropped": round(float(numpy.nanmean(numpy.stack([rd[a][s] for s in seeds]))), 4),
               "entered": round(float(numpy.nanmean(numpy.stack([re_[a][s] for s in seeds]))), 4)} for a in RUNS}
    by_epoch = {a: {en: {"kept_teacher_rank": round(float(numpy.mean([per_run[a][str(s)][en]["kept_teacher_rank"]
                                                                        for s in seeds])), 4),
                         "overlap_teacher_top29": round(float(numpy.mean(
                             [per_run[a][str(s)][en]["overlap_teacher_top29"] for s in seeds])), 4)}
                    for en in ("e25", "e50", "final")} for a in RUNS}

    api.write_json("summary", {
        "n_val": int(n), "K": K,
        "P1_mean_d_dropped_minus_entered_maskedkd": mean_d["maskedkd"],
        "P2_spearman_d_dG_maskedkd": rho["maskedkd"],
        "mean_d": mean_d, "spearman_d_dG": rho, "per_seed_mean_d": per_seed,
        "mean_rank_dropped_entered": sep, "by_epoch": by_epoch,
        "n_nan_images": {a: int((~ok[a]).sum()) for a in RUNS},
        "mean_dG": {a: round(float(dG[a].mean()), 4) for a in RUNS},
        "per_run": per_run,
        "verdict": {"P1": verdict(*mean_d["maskedkd"][1:]), "P2": verdict(*rho["maskedkd"][1:])}})
