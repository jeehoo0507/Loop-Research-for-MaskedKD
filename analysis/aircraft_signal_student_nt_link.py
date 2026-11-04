"""H-A132 (round 408, analysis only, no training): Aircraft replication of H-A131
(analysis/cub_signal_student_nt_link.py, code unchanged except runs, bootstrap seed and text; H-A128/H-A130 found
+0.132 nats on Aircraft). Original text: H-A127/H-A129 found that on CUB at K=29 the MaskedKD student's
temperature-scaled non-target class distribution is further from the full teacher's than Rollout@29's (+0.120 nats,
mostly on the teacher's top-5 confusable classes). Is that loss on the same images where the masked teacher (the KD
signal the selector produces) loses non-target shape? If yes, the student-side loss is traceable image by image to
what the selector hid from the teacher (Hinton et al. 2015; Zhao et al. CVPR 2022 NCKD; Son et al. ECCV 2024).

Per seed s: student logits of MaskedKD@29 and Rollout@29 (same runs as H-A127), 2-fold index-parity temperature as
in H-A127 -> per image non-target KL(full teacher || student) = S_m, S_r. Signal: the final student of the same run
picks 29 patches by its own rule (MaskedKD: last-block head-mean CLS attention; Rollout: Abnar & Zuidema rollout with
0.5 A + 0.5 I, as harness/baselines.py), the teacher sees only those patches -> per image non-target
KL(full teacher || masked teacher) at T=1 (config.TAU) = G_m, G_r.
dS_i = mean_s (S_m - S_r), dG_i = mean_s (G_m - G_r).
Primary P1: Spearman(dS, dG) over val images, image bootstrap 2000, numpy default_rng(20261141), 95% quantile CI.
P2: mean dG (nats), same bootstrap. Record: per-seed Spearman, within-method Spearman(S_m, G_m) and (S_r, G_r),
mean dS (must equal H-A127), top-5 non-target overlap of each masked teacher with the full teacher.
Limit: clean val views and final students; training used augmented views and the student of that step.
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3, K=29), are the images where the MaskedKD student loses more non-target "
                "shape than Rollout the same images where the MaskedKD-masked teacher loses more non-target shape?",
    "method": "per image non-target KL(full teacher || student) after held-out temperature scaling and non-target "
              "KL(full teacher || teacher on the 29 patches the run's own final student selects); paired MaskedKD - "
              "Rollout differences; Spearman across images with image bootstrap; seed t CI",
    "paper": "Hinton et al. 2015 (dark knowledge); Zhao et al. CVPR 2022 (DKD, NCKD); Abnar & Zuidema ACL 2020 "
             "(attention rollout); Guo et al. ICML 2017 (temperature scaling); Son et al. ECCV 2024 (MaskedKD); "
             "Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261142
K = 29
TOPK = 5
T_GRID = numpy.exp(numpy.linspace(numpy.log(0.1), numpy.log(10.0), 400))
RUNS = {
    "maskedkd": {1: "aircraft_final_maskedkd_k29_s1_1102_132227_6542",
                 2: "aircraft_final_maskedkd_k29_s2_1102_180321_52e2",
                 3: "aircraft_final_maskedkd_k29_s3_1102_180336_347e"},
    "rollout": {1: "aircraft_final_rollout_k29_s1_1102_132227_c3d7",
                2: "aircraft_final_rollout_k29_s2_1102_180325_20b6",
                3: "aircraft_final_rollout_k29_s3_1102_180341_8606"},
}


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _logp(L, T):
    Z = L / T
    Z = Z - Z.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _nll(L, y, T):
    return -_logp(L, T)[numpy.arange(len(y)), y]


def _fit_T(L, y):
    losses = [float(_nll(L, y, T).mean()) for T in T_GRID]
    return float(T_GRID[int(numpy.argmin(losses))])


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


def _scores(api, S, x, method):
    if method == "maskedkd":
        out = api.student_forward(S, x, attn_layers=[-1])
        return out, out.attn[len(S.blocks) - 1].float().mean(1)[:, 0, 1:]
    out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
    r = None
    for a in out.attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return out, r[:, 0, 1:]


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    Tm = api.teacher.eval()
    batches, tl = [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    t_logp = _logp(torch.cat(tl).numpy().astype(numpy.float64), 1.0)
    n = len(y)
    folds = [numpy.arange(n) % 2 == 0, numpy.arange(n) % 2 == 1]
    tn = _nt_logp(t_logp, y)
    t_top = numpy.argsort(-tn, 1)[:, :TOPK]

    S_kl, G_kl, per_run = {a: {} for a in RUNS}, {a: {} for a in RUNS}, {a: {} for a in RUNS}
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            ls, gs = [], []
            for x, _ in batches:
                with torch.no_grad():
                    out, sc = _scores(api, S, x, a)
                    ls.append(out.logits.float().cpu())
                    keep = _topk_mask(sc, K)
                    gs.append(api.teacher_forward_keepmask(Tm, x, keep).float().cpu())
            L = torch.cat(ls).numpy().astype(numpy.float64)
            G = torch.cat(gs).numpy().astype(numpy.float64)
            del S
            torch.cuda.empty_cache()
            s_lp_t, Ts = numpy.zeros_like(L), []
            for fit, ev in ((folds[0], folds[1]), (folds[1], folds[0])):
                T = _fit_T(L[fit], y[fit])
                Ts.append(round(T, 4))
                s_lp_t[ev] = _logp(L[ev], T)
            S_kl[a][s] = _nt_kl(tn, _nt_logp(s_lp_t, y))
            gn = _nt_logp(_logp(G, 1.0), y)
            G_kl[a][s] = _nt_kl(tn, gn)
            g_top = numpy.argsort(-gn, 1)[:, :TOPK]
            ov = numpy.mean([len(set(t_top[i]) & set(g_top[i])) / TOPK for i in range(n)])
            per_run[a][str(s)] = {"T_fit_even_fit_odd": Ts,
                                  "student_nt_kl": round(float(S_kl[a][s].mean()), 4),
                                  "signal_nt_kl": round(float(G_kl[a][s].mean()), 4),
                                  "signal_top5_overlap": round(float(ov), 4),
                                  "signal_acc": round(100 * float((G.argmax(1) == y).mean()), 2),
                                  "spearman_student_vs_signal": round(_spearman(S_kl[a][s], G_kl[a][s]), 4),
                                  "acc_measured": round(100 * float((L.argmax(1) == y).mean()), 2),
                                  "val_acc_last": runs[rid]["val_acc_last"]}

    seeds = sorted(RUNS["maskedkd"])
    dS = numpy.mean([S_kl["maskedkd"][s] - S_kl["rollout"][s] for s in seeds], 0)
    dG = numpy.mean([G_kl["maskedkd"][s] - G_kl["rollout"][s] for s in seeds], 0)
    Sm, Sr = numpy.mean([S_kl["maskedkd"][s] for s in seeds], 0), numpy.mean([S_kl["rollout"][s] for s in seeds], 0)
    Gm, Gr = numpy.mean([G_kl["maskedkd"][s] for s in seeds], 0), numpy.mean([G_kl["rollout"][s] for s in seeds], 0)
    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def boot(fn):
        b = numpy.array([fn(ix) for ix in idxs])
        return [round(float(numpy.quantile(b, 0.025)), 4), round(float(numpy.quantile(b, 0.975)), 4)]

    rho = _spearman(dS, dG)
    rho_ci = boot(lambda ix: _spearman(dS[ix], dG[ix]))
    dG_ci = boot(lambda ix: dG[ix].mean())
    dS_ci = boot(lambda ix: dS[ix].mean())
    within = {"maskedkd": [round(_spearman(Sm, Gm), 4)] + boot(lambda ix: _spearman(Sm[ix], Gm[ix])),
              "rollout": [round(_spearman(Sr, Gr), 4)] + boot(lambda ix: _spearman(Sr[ix], Gr[ix]))}
    per_seed_rho = [round(_spearman(S_kl["maskedkd"][s] - S_kl["rollout"][s],
                                    G_kl["maskedkd"][s] - G_kl["rollout"][s]), 4) for s in seeds]

    def verdict(lo, hi):
        return "positive" if lo > 0 else ("negative" if hi < 0 else "uncertain")

    api.write_json("summary", {
        "n_val": int(n), "K": K, "T_grid": [0.1, 10.0, len(T_GRID)],
        "P1_spearman_dS_dG": [round(rho, 4)] + rho_ci,
        "P2_mean_dG_signal": [round(float(dG.mean()), 4)] + dG_ci,
        "mean_dS_student": [round(float(dS.mean()), 4)] + dS_ci,
        "per_seed_spearman_dS_dG": per_seed_rho, "per_seed_t_spearman": _t95(per_seed_rho),
        "per_seed_t_dG": _t95([float((G_kl["maskedkd"][s] - G_kl["rollout"][s]).mean()) for s in seeds]),
        "within_method_spearman_student_vs_signal": within,
        "per_run": per_run,
        "verdict": {"P1": verdict(*rho_ci), "P2": verdict(*dG_ci)}})
