"""H-A106 (round 395, analysis only, no training): H-A105 found on CUB that the KD gain (Full - CE, NCM +1.78) and the
masking loss (Full - MaskedKD@29, NCM +2.83) show in a leave-one-out NCM probe on the students' features (image
bootstrap; seed t CI included 0). Replication across datasets instead of seeds: the same probe code on FGVC-Aircraft
(final, seeds 1-3):
  KD gain      Full@196 - CE        (NCM gain N_kd, logit gain G_kd)
  masking loss Full@196 - MaskedKD@29 (NCM loss N_m, logit loss G_m)

Per image i and seed s (final weights, Aircraft val, un-augmented):
  feat_s  student pre-logit feature (forward_head pre_logits), L2-normalized
  ncm_s   leave-one-out nearest-class-mean (cosine) correct on val; classes with < 2 val images are dropped
  log_s   logit argmax correct
N_i = mean_s (ncm A_s - ncm B_s), G_i = same for log. Primary P1: mean N_kd over valid images (%p).
Secondary P2: mean N_m. Image bootstrap 2000, numpy default_rng(20261116).
Record: G_kd, G_m, G - N for both, per-seed t 95% CI (df 2), NCM/logit acc per method/seed, measured val acc vs
val_acc_last. Labels enter only the evaluation; no mask is used.
"""
import numpy
import torch

META = {
    "question": "on FGVC-Aircraft (final, seeds 1-3), is the KD gain (Full - CE) and the masking loss (Full - MaskedKD@29) "
                "visible in a leave-one-out nearest-class-mean probe on the students' features, or only in logits?",
    "method": "pre-logit features of each student; leave-one-out cosine nearest-class-mean probe on val; paired "
              "per-image differences Full - CE and Full - MaskedKD, image bootstrap and seed t CI",
    "paper": "Allen-Zhu & Li ICLR 2023 (multi-view KD); Kang et al. ICLR 2020 (NCM); Son et al. ECCV 2024 (MaskedKD); "
             "Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261116
RUNS = {
    "maskedkd": {1: "aircraft_final_maskedkd_k29_s1_1102_132227_6542",
                 2: "aircraft_final_maskedkd_k29_s2_1102_180321_52e2",
                 3: "aircraft_final_maskedkd_k29_s3_1102_180336_347e"},
    "full": {1: "aircraft_final_full_k196_s1_1102_123044_ed5d",
             2: "aircraft_final_full_k196_s2_1102_132227_123b",
             3: "aircraft_final_full_k196_s3_1102_132227_69cc"},
    "ce": {1: "aircraft_final_ce_k0_s1_1102_123044_6d5f",
           2: "aircraft_final_ce_k0_s2_1102_132227_80b7",
           3: "aircraft_final_ce_k0_s3_1102_132227_55cb"},
}
CONTRASTS = {"kd_gain": ("full", "ce"), "mask_loss": ("full", "maskedkd")}


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


def _verdict(ci):
    if ci[1] > 0:
        return "positive"
    if ci[2] < 0:
        return "negative"
    return "uncertain"


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
    seeds = sorted(RUNS["full"])

    rng = numpy.random.default_rng(SEED)
    sub, per_seed, seed_t, verdict = {}, {}, {}, {}
    for name, (A, B) in CONTRASTS.items():
        N = numpy.mean([ncm[A][s] - ncm[B][s] for s in seeds], 0)[valid]
        G = numpy.mean([logc[A][s] - logc[B][s] for s in seeds], 0)[valid]
        sub[name] = {"ncm_pct_boot": _boot_mean(N, rng), "logit_pct_boot": _boot_mean(G, rng),
                     "logit_minus_ncm_pct_boot": _boot_mean(G - N, rng)}
        ps = {"ncm": [round(100 * float((ncm[A][s] - ncm[B][s])[valid].mean()), 3) for s in seeds],
              "logit": [round(100 * float((logc[A][s] - logc[B][s])[valid].mean()), 3) for s in seeds]}
        ps["logit_minus_ncm"] = [round(g - n, 3) for g, n in zip(ps["logit"], ps["ncm"])]
        per_seed[name] = ps
        seed_t[name] = {k: _t95(v) for k, v in ps.items()}
        verdict[name] = {"ncm": _verdict(sub[name]["ncm_pct_boot"]),
                         "logit_minus_ncm": _verdict(sub[name]["logit_minus_ncm_pct_boot"])}

    acc = {a: {str(s): {"ncm": round(100 * float(ncm[a][s][valid].mean()), 2),
                        "logit": round(100 * float(logc[a][s][valid].mean()), 2),
                        "logit_all_measured": round(100 * float(logc[a][s].mean()), 2),
                        "val_acc_last": runs[RUNS[a][s]]["val_acc_last"]}
               for s in seeds} for a in RUNS}

    verdict["P1_kd_gain_in_features"] = verdict["kd_gain"]["ncm"] == "positive"
    verdict["P2_mask_loss_in_features"] = verdict["mask_loss"]["ncm"] == "positive"
    out = {"n_val": int(len(y)), "n_valid": int(valid.sum()), "n_dropped_singleton_class": int((~valid).sum()),
           "subsets": sub, "per_seed": per_seed, "per_seed_t": seed_t, "acc": acc, "verdict": verdict}
    api.write_json("summary", out)
