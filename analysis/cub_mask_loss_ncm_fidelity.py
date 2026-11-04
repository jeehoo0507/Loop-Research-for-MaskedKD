"""H-A109 (round 397, analysis only, no training): H-A105 found the masking loss (Full@196 - MaskedKD@29) in the
students' features (leave-one-out NCM), H-A97 found the logit loss concentrated on images where the masked teacher
departs from the full teacher (high KL). Is the feature (NCM) loss concentrated on the same images? Same runs as
H-A97/H-A105 (CUB final, seeds 1-3); masked teacher = H-A97 rule, NCM probe = H-A105 probe.

Per image i and seed s (final weights, val 600, un-augmented):
  kl_s   KL(p_T || q_s), p_T = clean full teacher softmax, q_s = keep-only masked teacher with the maskedkd@29 seed-s
         student's own last-block head-mean CLS attention top-29 (harness MaskedKD rule)
  ncm_s  leave-one-out cosine nearest-class-mean correct on the student's L2-normalized pre-logit feature
  log_s  logit argmax correct
f_i = mean_s kl_s, N_i = mean_s (ncm full_s - ncm maskedkd_s), G_i = same for logits. Quintiles of f (Q1 = most
faithful), fixed from the full sample, over images whose class has >= 2 val images. Primary P1: Q5 - Q1 of N (%p),
paired image bootstrap 2000, numpy default_rng(20261119).
Record: per-quintile N, G, KD gain NCM (Full - CE), masked teacher acc, teacher confidence; Spearman(f, N) with
bootstrap CI; Q5 - Q1 of G (re-measure of H-A97); per-seed Q5 - Q1 of N (own-seed kl quintiles) with t 95% CI (df 2);
measured val acc vs val_acc_last. Labels enter only correctness and the NCM class means; no labels enter any mask.
"""
import numpy
import torch

META = {
    "question": "on CUB (final, seeds 1-3), is the feature (leave-one-out NCM) loss of MaskedKD@29 relative to "
                "Full@196 concentrated on images where the masked teacher's soft label departs from the full teacher's?",
    "method": "keep-only masked teacher with each maskedkd student's own last-block CLS top-29; per-image KL "
              "quintiles; NCM loss Q5 - Q1 with paired image bootstrap and per-seed t CI",
    "paper": "Hinton et al. 2015 (KD); Son et al. ECCV 2024 (MaskedKD); Allen-Zhu & Li ICLR 2023 (multi-view KD); "
             "Kang et al. ICLR 2020 (NCM); Wah et al. 2011 (CUB-200-2011)",
}

K = 29
BATCH = 50
N_BOOT = 2000
SEED = 20261119
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


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _spearman(a, b):
    ra = numpy.argsort(numpy.argsort(a)).astype(float)
    rb = numpy.argsort(numpy.argsort(b)).astype(float)
    return float(numpy.corrcoef(ra, rb)[0, 1])


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _quint(v):
    edges = numpy.quantile(v, [0.2, 0.4, 0.6, 0.8])
    return numpy.searchsorted(edges, v, side="right"), edges


def _qmeans(v, q):
    return [float(v[q == j].mean()) if (q == j).any() else float("nan") for j in range(5)]


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


def _ci(v, b, scale):
    return [round(scale * v, 4), round(scale * float(numpy.quantile(b, 0.025)), 4),
            round(scale * float(numpy.quantile(b, 0.975)), 4)]


def _verdict(ci):
    if ci[1] > 0:
        return "ncm_loss_in_low_fidelity"
    if ci[2] < 0:
        return "ncm_loss_in_high_fidelity"
    return "uncertain"


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    T = api.teacher.eval()

    batches, logp_t, conf = [], [], []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            lp = api.student_forward(T, x).logits.float().log_softmax(-1)
        logp_t.append(lp)
        conf.append(lp.exp().max(-1).values)
        batches.append((x, y))
    conf = torch.cat(conf).cpu().numpy()
    y_all = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)

    logc, feats = {a: {} for a in RUNS}, {a: {} for a in RUNS}
    kl, t_acc = {}, {}
    n_cls = None
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            depth = len(S.blocks)
            cor, fs, kls, tac = [], [], [], []
            for (x, y), lp in zip(batches, logp_t):
                with torch.no_grad():
                    al = [depth - 1] if a == "maskedkd" else []
                    so = api.student_forward(S, x, token_layers=[depth - 1], attn_layers=al)
                    f = S.forward_head(S.norm(so.tokens[depth - 1]), pre_logits=True)
                    if a == "maskedkd":
                        keep = _topk(so.attn[depth - 1].float().mean(1)[:, 0, 1:], K)
                        lq = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                        kls.append((lp.exp() * (lp - lq)).sum(-1))
                        tac.append((lq.argmax(-1) == y).float())
                fs.append(f.float().cpu())
                cor.append((so.logits.float().argmax(-1) == y).float())
                n_cls = so.logits.shape[-1]
            logc[a][s] = torch.cat(cor).cpu().numpy()
            feats[a][s] = torch.cat(fs).numpy().astype(numpy.float64)
            if a == "maskedkd":
                kl[s] = torch.cat(kls).cpu().numpy()
                t_acc[s] = torch.cat(tac).cpu().numpy()
            del S

    counts = numpy.bincount(y_all, minlength=n_cls)
    valid = counts[y_all] >= 2
    ncm = {a: {s: _loo_ncm(feats[a][s], y_all, n_cls) for s in RUNS[a]} for a in RUNS}
    seeds = sorted(RUNS["maskedkd"])
    v = valid
    N_seed = {s: (ncm["full"][s] - ncm["maskedkd"][s])[v] for s in seeds}
    N = numpy.mean([N_seed[s] for s in seeds], 0)
    G = numpy.mean([(logc["full"][s] - logc["maskedkd"][s])[v] for s in seeds], 0)
    Nkd = numpy.mean([(ncm["full"][s] - ncm["ce"][s])[v] for s in seeds], 0)
    f = numpy.mean([kl[s] for s in seeds], 0)[v]
    ta = numpy.mean([t_acc[s] for s in seeds], 0)[v]
    cf = conf[v]
    n = len(N)
    q, edges = _quint(f)

    qN, qG = _qmeans(N, q), _qmeans(G, q)
    rho = _spearman(f, N)
    rng = numpy.random.default_rng(SEED)
    bN, bG, br = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        mN, mG = _qmeans(N[idx], q[idx]), _qmeans(G[idx], q[idx])
        bN.append(mN[4] - mN[0])
        bG.append(mG[4] - mG[0])
        br.append(_spearman(f[idx], N[idx]))
    ci = _ci(qN[4] - qN[0], bN, 100)
    ci_g = _ci(qG[4] - qG[0], bG, 100)
    ci_rho = _ci(rho, br, 1)

    per_seed = {}
    for s in seeds:
        qs, _ = _quint(kl[s][v])
        mm = _qmeans(N_seed[s], qs)
        per_seed[s] = mm[4] - mm[0]
    seed_t = _t95([100 * per_seed[s] for s in seeds])

    verdict = _verdict(ci)
    out = {"k": K, "n_val": int(len(y_all)), "n_valid": int(n), "n_dropped_singleton_class": int((~valid).sum()),
           "ncm_loss_mean_pct": round(100 * float(N.mean()), 3),
           "logit_loss_mean_pct": round(100 * float(G.mean()), 3),
           "val_acc": {a: {str(s): [round(100 * float(logc[a][s].mean()), 2),
                                    runs[RUNS[a][s]]["val_acc_last"]] for s in seeds} for a in RUNS},
           "ncm_acc": {a: {str(s): round(100 * float(ncm[a][s][v].mean()), 2) for s in seeds} for a in RUNS},
           "kl_quintile_edges": [round(float(e), 4) for e in edges],
           "quintile_n": [int((q == j).sum()) for j in range(5)],
           "quintile_kl": [round(x, 4) for x in _qmeans(f, q)],
           "quintile_ncm_loss_pct": [round(100 * x, 3) for x in qN],
           "quintile_logit_loss_pct": [round(100 * x, 3) for x in qG],
           "quintile_kd_gain_ncm_pct": [round(100 * x, 3) for x in _qmeans(Nkd, q)],
           "quintile_masked_teacher_acc_pct": [round(100 * x, 2) for x in _qmeans(ta, q)],
           "quintile_teacher_conf": [round(x, 4) for x in _qmeans(cf, q)],
           "P1_ncm_loss_q5_minus_q1_pct_boot": ci,
           "logit_loss_q5_minus_q1_pct_boot": ci_g,
           "spearman_kl_ncm_loss_boot": ci_rho,
           "spearman_kl_teacher_conf": round(_spearman(f, cf), 4),
           "per_seed_ncm_q5_minus_q1_pct": {str(s): round(100 * per_seed[s], 3) for s in seeds},
           "per_seed_ncm_q5_minus_q1_t": seed_t,
           "verdict": {"ncm_q5_minus_q1": verdict, "P1_ncm_loss_in_low_fidelity": verdict == "ncm_loss_in_low_fidelity",
                       "logit_q5_minus_q1": _verdict(ci_g)}}
    api.write_json("summary", out)
