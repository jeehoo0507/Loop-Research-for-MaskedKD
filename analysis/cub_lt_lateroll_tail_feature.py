"""H-A102 (round 391, analysis only, no training): does lateroll@29's cub_lt tail advantage over maskedkd@29 (same cost,
H-A67 +5.78; H-A101: +5.56 even where the masked teacher is faithful) live in the student's features, or only in its
linear head? Long-tail work (Kang et al. ICLR 2020, decoupling) separates the two with a nearest-class-mean probe.

Per image i and seed s (final weights, seeds 1-3, val, un-augmented):
  feat_s  student pre-logit feature (forward_head pre_logits), L2-normalized
  ncm_s   leave-one-out nearest-class-mean (cosine) correct on val: own-class centroid excludes image i; images whose
          class has < 2 val images are dropped
  log_s   logit argmax correct
N_i = mean_s (ncm lateroll_s - ncm maskedkd_s), G_i = same for log. Primary: tail (val group 1) mean N (%p), image
bootstrap 2000, numpy default_rng(20261113).
Record: head and all mean N; tail mean G and G - N (paired bootstrap); per-seed tail N t 95% CI (df 2); NCM and logit acc
per method/seed/group; fraction of tail-image logit predictions that fall in tail classes per method; measured val acc
vs val_acc_last. Groups and labels enter only the evaluation; no mask is used.
"""
import numpy
import torch

META = {
    "question": "on cub_lt (final, seeds 1-3, K=29), is lateroll's tail accuracy advantage over MaskedKD also present "
                "in a leave-one-out nearest-class-mean probe on the students' features, or only in the linear head?",
    "method": "pre-logit features of each student; leave-one-out cosine nearest-class-mean probe on val; paired "
              "lateroll - maskedkd per image with image bootstrap and seed t CI",
    "paper": "Kang et al. ICLR 2020 (decoupling representation and classifier, NCM); Son et al. ECCV 2024 (MaskedKD); "
             "Abnar & Zuidema ACL 2020 (rollout); Wah et al. 2011 (CUB-200-2011); Liu et al. CVPR 2019 (long-tail)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261113
RUNS = {
    "maskedkd": {1: "cub_lt_final_maskedkd_k29_s1_1103_234609_76d1",
                 2: "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 3: "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"},
    "lateroll": {1: "cub_lt_final_lateroll_k29_s1_1104_084157_2b15",
                 2: "cub_lt_final_lateroll_k29_s2_1104_084204_9f09",
                 3: "cub_lt_final_lateroll_k29_s3_1104_084209_a3f8"},
}


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _boot_mean(v, rng):
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
    return [round(100 * float(v.mean()), 3), round(100 * float(numpy.quantile(bs, 0.025)), 3),
            round(100 * float(numpy.quantile(bs, 0.975)), 3)]


def _loo_ncm(F, y, n_cls):
    F = F / numpy.linalg.norm(F, axis=1, keepdims=True)
    S = numpy.zeros((n_cls, F.shape[1]))
    numpy.add.at(S, y, F)
    n = numpy.bincount(y, minlength=n_cls).astype(float)
    C = S / numpy.maximum(n, 1)[:, None]
    Cn = C / numpy.maximum(numpy.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    sims = F @ Cn.T
    sims[:, n == 0] = -numpy.inf
    own = (S[y] - F) / numpy.maximum(n[y] - 1, 1)[:, None]
    own = own / numpy.maximum(numpy.linalg.norm(own, axis=1, keepdims=True), 1e-12)
    sims[numpy.arange(len(y)), y] = (F * own).sum(1)
    return (sims.argmax(1) == y).astype(float)


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid]["val_wga_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    groups = api.val_groups().cpu().numpy()
    batches = list(api.val_batches(batch=BATCH))
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)

    feats, logc, preds = {a: {} for a in RUNS}, {a: {} for a in RUNS}, {a: {} for a in RUNS}
    n_cls = None
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            depth = len(S.blocks)
            fs, ps = [], []
            for x, _ in batches:
                with torch.no_grad():
                    so = api.student_forward(S, x, token_layers=[depth - 1])
                    f = S.forward_head(S.norm(so.tokens[depth - 1]), pre_logits=True)
                fs.append(f.float().cpu())
                ps.append(so.logits.float().argmax(-1).cpu())
                n_cls = so.logits.shape[-1]
            feats[a][s] = torch.cat(fs).numpy().astype(numpy.float64)
            preds[a][s] = torch.cat(ps).numpy()
            logc[a][s] = (preds[a][s] == y).astype(float)
            del S

    counts = numpy.bincount(y, minlength=n_cls)
    valid = counts[y] >= 2
    ncm = {a: {s: _loo_ncm(feats[a][s], y, n_cls) for s in RUNS[a]} for a in RUNS}
    seeds = sorted(RUNS["maskedkd"])
    N = numpy.mean([ncm["lateroll"][s] - ncm["maskedkd"][s] for s in seeds], 0)
    G = numpy.mean([logc["lateroll"][s] - logc["maskedkd"][s] for s in seeds], 0)
    tail, head = (groups == 1) & valid, (groups == 0) & valid
    tail_cls = numpy.zeros(n_cls, dtype=bool)
    tail_cls[numpy.unique(y[groups == 1])] = True

    rng = numpy.random.default_rng(SEED)
    sub = {}
    for name, m in (("tail", tail), ("head", head), ("all", valid)):
        sub[name] = {"n": int(m.sum()), "ncm_gain_pct_boot": _boot_mean(N[m], rng),
                     "logit_gain_pct_boot": _boot_mean(G[m], rng),
                     "logit_minus_ncm_gain_pct_boot": _boot_mean(G[m] - N[m], rng)}
    seed_t = _t95([100 * float((ncm["lateroll"][s] - ncm["maskedkd"][s])[tail].mean()) for s in seeds])

    acc = {a: {str(s): {"ncm_tail": round(100 * float(ncm[a][s][tail].mean()), 2),
                        "ncm_head": round(100 * float(ncm[a][s][head].mean()), 2),
                        "logit_tail": round(100 * float(logc[a][s][tail].mean()), 2),
                        "logit_head": round(100 * float(logc[a][s][head].mean()), 2),
                        "logit_all_measured": round(100 * float(logc[a][s].mean()), 2),
                        "val_acc_last": runs[RUNS[a][s]]["val_acc_last"],
                        "tail_img_pred_in_tail_cls_pct": round(100 * float(tail_cls[preds[a][s][groups == 1]].mean()), 2),
                        "head_img_pred_in_tail_cls_pct": round(100 * float(tail_cls[preds[a][s][groups == 0]].mean()), 2)}
               for s in seeds} for a in RUNS}

    ci = sub["tail"]["ncm_gain_pct_boot"]
    if ci[1] > 0:
        verdict = "tail_gain_in_features"
    elif ci[2] < 0:
        verdict = "lateroll_features_worse_on_tail"
    else:
        verdict = "uncertain"

    out = {"n_val": int(len(y)), "n_dropped_singleton_class": int((~valid).sum()), "n_tail_classes": int(tail_cls.sum()),
           "subsets": sub, "per_seed_tail_ncm_gain_t": seed_t, "acc": acc,
           "verdict": {"tail_ncm_gain": verdict, "P1_tail_gain_in_features": verdict == "tail_gain_in_features"}}
    api.write_json("summary", out)
