"""H-A111 (round 398, analysis only, no training): H-A109/110 found the feature NCM loss of masking (Full@196 -
MaskedKD@29) flat across masked-teacher KL quintiles, i.e. not located on the images where the masked teacher loses
its soft label. A flat loss fits a global change of the feature geometry. Neural-collapse view (Papyan et al. PNAS
2020): KD from the full teacher should tighten classes relative to class-mean spacing. On CUB (final, seeds 1-3):

Per image i and seed s (final weights, CUB val, un-augmented), features L2-normalized:
  m_s(i) = cos(f_i, own-class leave-one-out mean) - max_{c != y_i} cos(f_i, mean_c)   (class margin, continuous)
M_i = mean_s (m A_s - m B_s). Primary P1: mean M for Full - MaskedKD over valid images (x100), image bootstrap 2000,
numpy default_rng(20261121). Secondary P2: mean M for MaskedKD - CE. Record: Full - CE, per-seed t CI (df 2),
per-seed NC1 ratio tr(S_W)/tr(S_B) on normalized features (lower = tighter) and its paired t CI, measured val acc vs
val_acc_last. Classes with < 2 val images are dropped. Labels enter only the evaluation; no mask is used.
"""
import numpy
import torch

META = {
    "question": "on CUB (final, seeds 1-3), does masking (MaskedKD@29 vs Full@196) lower the per-image class margin "
                "of the students' features (own LOO class mean vs nearest other class mean), a global geometry change?",
    "method": "pre-logit features of each student; per-image cosine class margin and NC1 ratio on val; paired "
              "per-image differences Full - MaskedKD, MaskedKD - CE, Full - CE; image bootstrap and seed t CI",
    "paper": "Papyan, Han & Donoho PNAS 2020 (neural collapse); Son et al. ECCV 2024 (MaskedKD); Wah et al. 2011 (CUB)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261121
RUNS = {
    "maskedkd": {1: "final_maskedkd_k29_s1_1101_104303_b676",
                 2: "final_maskedkd_k29_s2_1101_125958_5b02",
                 3: "final_maskedkd_k29_s3_1101_140937_b00f"},
    "full": {1: "cub_final_full_k196_s1_1030_153055_d87b",
             2: "final_full_k196_s2_1101_125958_cc04",
             3: "final_full_k196_s3_1101_140937_e8fa"},
    "ce": {1: "cub_final_ce_k0_s1_1030_153055_1256",
           2: "final_ce_k0_s2_1103_013642_841b",
           3: "final_ce_k0_s3_1103_013646_bb9a"},
}
CONTRASTS = {"mask_loss": ("full", "maskedkd"), "mask_vs_ce": ("maskedkd", "ce"), "kd_gain": ("full", "ce")}


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _boot_mean(v, rng):
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
    return [round(100 * float(v.mean()), 3), round(100 * float(numpy.quantile(bs, 0.025)), 3),
            round(100 * float(numpy.quantile(bs, 0.975)), 3)]


def _margin(F, y, n_cls):
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
    own_sim = (F * own).sum(1)
    sims[numpy.arange(len(y)), y] = -numpy.inf
    return own_sim - sims.max(1)


def _nc1(F, y, keep):
    F = F / numpy.linalg.norm(F, axis=1, keepdims=True)
    F, y = F[keep], y[keep]
    mu = F.mean(0)
    w, b = 0.0, 0.0
    for c in numpy.unique(y):
        Fc = F[y == c]
        mc = Fc.mean(0)
        w += float(((Fc - mc) ** 2).sum())
        b += len(Fc) * float(((mc - mu) ** 2).sum())
    return w / b


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
    seeds = sorted(RUNS["full"])
    marg = {a: {s: _margin(feats[a][s], y, n_cls) for s in seeds} for a in RUNS}
    nc1 = {a: {s: _nc1(feats[a][s], y, valid) for s in seeds} for a in RUNS}

    rng = numpy.random.default_rng(SEED)
    sub, per_seed, seed_t, verdict = {}, {}, {}, {}
    for name, (A, B) in CONTRASTS.items():
        M = numpy.mean([marg[A][s] - marg[B][s] for s in seeds], 0)[valid]
        sub[name] = {"margin_x100_boot": _boot_mean(M, rng)}
        ps = {"margin_x100": [round(100 * float((marg[A][s] - marg[B][s])[valid].mean()), 4) for s in seeds],
              "nc1": [round(nc1[A][s] - nc1[B][s], 5) for s in seeds],
              "logit_pct": [round(100 * float((logc[A][s] - logc[B][s])[valid].mean()), 3) for s in seeds]}
        per_seed[name] = ps
        seed_t[name] = {k: _t95(v) for k, v in ps.items()}
        verdict[name] = _verdict(sub[name]["margin_x100_boot"])

    acc = {a: {str(s): {"margin_x100_mean": round(100 * float(marg[a][s][valid].mean()), 3),
                        "nc1": round(nc1[a][s], 5),
                        "logit": round(100 * float(logc[a][s][valid].mean()), 2),
                        "logit_all_measured": round(100 * float(logc[a][s].mean()), 2),
                        "val_acc_last": runs[RUNS[a][s]]["val_acc_last"]}
               for s in seeds} for a in RUNS}

    verdict["P1_full_minus_maskedkd_margin"] = verdict["mask_loss"]
    verdict["P2_maskedkd_minus_ce_margin"] = verdict["mask_vs_ce"]
    out = {"n_val": int(len(y)), "n_valid": int(valid.sum()), "n_dropped_singleton_class": int((~valid).sum()),
           "subsets": sub, "per_seed": per_seed, "per_seed_t": seed_t, "acc": acc, "verdict": verdict}
    api.write_json("summary", out)
