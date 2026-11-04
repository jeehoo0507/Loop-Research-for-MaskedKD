"""H-A103 (round 392, analysis only, no training): H-A102 found that on cub_lt, lateroll@29's tail advantage over
maskedkd@29 appears in the logits (+5.78) but not in a leave-one-out NCM probe on the features (0.00). Does the same
split hold on balanced CUB (same cost, K=29), where lateroll - maskedkd is +2.78 [+0.39, +5.17] in logits (H-A100)?
If the logit gain also exceeds the NCM gain on balanced data, the cub_lt reading "head-side tail bias" is not specific
to the long tail.

Per image i and seed s (final weights, seeds 1-3, CUB val, un-augmented):
  feat_s  student pre-logit feature (forward_head pre_logits), L2-normalized
  ncm_s   leave-one-out nearest-class-mean (cosine) correct on val: own-class centroid excludes image i; images whose
          class has < 2 val images are dropped
  log_s   logit argmax correct
N_i = mean_s (ncm lateroll_s - ncm maskedkd_s), G_i = same for log. Primary: mean (G - N) over valid images (%p),
image bootstrap 2000, numpy default_rng(20261114).
Record: mean N and mean G (bootstrap); per-seed N, G, G - N t 95% CI (df 2); NCM and logit acc per method/seed;
measured val acc vs val_acc_last. Labels enter only the evaluation; no mask is used.
"""
import numpy
import torch

META = {
    "question": "on balanced CUB (final, seeds 1-3, K=29), is lateroll's logit accuracy gain over MaskedKD larger than "
                "its gain in a leave-one-out nearest-class-mean probe on the students' features (as on cub_lt tail)?",
    "method": "pre-logit features of each student; leave-one-out cosine nearest-class-mean probe on val; paired "
              "lateroll - maskedkd per image, logit gain minus NCM gain, image bootstrap and seed t CI",
    "paper": "Kang et al. ICLR 2020 (decoupling representation and classifier, NCM); Son et al. ECCV 2024 (MaskedKD); "
             "Abnar & Zuidema ACL 2020 (rollout); Wah et al. 2011 (CUB-200-2011)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261114
RUNS = {
    "maskedkd": {1: "final_maskedkd_k29_s1_1101_104303_b676",
                 2: "final_maskedkd_k29_s2_1101_125958_5b02",
                 3: "final_maskedkd_k29_s3_1101_140937_b00f"},
    "lateroll": {1: "final_lateroll_k29_s1_1101_104303_ef38",
                 2: "final_lateroll_k29_s2_1101_125958_6470",
                 3: "final_lateroll_k29_s3_1101_140937_ec98"},
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
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    batches = list(api.val_batches(batch=BATCH))
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)

    feats, logc = {a: {} for a in RUNS}, {a: {} for a in RUNS}
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
            logc[a][s] = (torch.cat(ps).numpy() == y).astype(float)
            del S

    counts = numpy.bincount(y, minlength=n_cls)
    valid = counts[y] >= 2
    ncm = {a: {s: _loo_ncm(feats[a][s], y, n_cls) for s in RUNS[a]} for a in RUNS}
    seeds = sorted(RUNS["maskedkd"])
    N = numpy.mean([ncm["lateroll"][s] - ncm["maskedkd"][s] for s in seeds], 0)
    G = numpy.mean([logc["lateroll"][s] - logc["maskedkd"][s] for s in seeds], 0)

    rng = numpy.random.default_rng(SEED)
    sub = {"n": int(valid.sum()),
           "logit_minus_ncm_gain_pct_boot": _boot_mean(G[valid] - N[valid], rng),
           "ncm_gain_pct_boot": _boot_mean(N[valid], rng),
           "logit_gain_pct_boot": _boot_mean(G[valid], rng)}
    per_seed = {}
    for key, src in (("ncm", ncm), ("logit", logc)):
        per_seed[key] = [round(100 * float((src["lateroll"][s] - src["maskedkd"][s])[valid].mean()), 3) for s in seeds]
    per_seed["logit_minus_ncm"] = [round(g - n, 3) for g, n in zip(per_seed["logit"], per_seed["ncm"])]
    seed_t = {k: _t95(v) for k, v in per_seed.items()}

    acc = {a: {str(s): {"ncm": round(100 * float(ncm[a][s][valid].mean()), 2),
                        "logit": round(100 * float(logc[a][s][valid].mean()), 2),
                        "logit_all_measured": round(100 * float(logc[a][s].mean()), 2),
                        "val_acc_last": runs[RUNS[a][s]]["val_acc_last"]}
               for s in seeds} for a in RUNS}

    ci = sub["logit_minus_ncm_gain_pct_boot"]
    if ci[1] > 0:
        verdict = "logit_gain_exceeds_ncm_gain"
    elif ci[2] < 0:
        verdict = "ncm_gain_exceeds_logit_gain"
    else:
        verdict = "uncertain"

    out = {"n_val": int(len(y)), "n_dropped_singleton_class": int((~valid).sum()), "subsets": sub,
           "per_seed": per_seed, "per_seed_t": seed_t, "acc": acc,
           "verdict": {"logit_minus_ncm": verdict, "P1_logit_gain_exceeds_ncm_gain":
                       verdict == "logit_gain_exceeds_ncm_gain"}}
    api.write_json("summary", out)
