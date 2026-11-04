"""H-A136 (round 410, analysis only, no training): Aircraft replication of H-A135
(analysis/cub_signal_student_nt_link_snap.py, code unchanged except runs, bootstrap seed and text; H-A132 final-student
rho +0.24). Original text: H-A131 found on CUB that the images where the MaskedKD@29 student
loses more non-target shape than Rollout@29 (dS) are weakly the images where the masked teacher loses more
non-target shape (dG): Spearman +0.21 [+0.13, +0.29]; H-A133 found the same on crop+flip views. Both used the FINAL
student to pick the 29 patches, but during training the KD signal was produced by the student of that step. Does the
signal picked by the mid-training students (e25, e50 snapshots) carry the same per-image link to the final student's
loss?

Same runs, rules, temperature folds and KL as H-A131 (analysis/cub_signal_student_nt_link.py, inlined), clean val views.
S = non-target KL(full teacher || temperature-scaled FINAL student) (= H-A131). For each epoch e in {25, 50, final}:
G_e = non-target KL(full teacher || teacher on the 29 patches the run's own student at epoch e selects), T=1.
dS_i = mean_s (S_m - S_r); dG_e,i = mean_s (G_m,e - G_r,e); dG_mid = (dG_25 + dG_50) / 2.
Primary P1: Spearman(dS, dG_mid), image bootstrap 2000, numpy default_rng(20261145), 95% quantile CI.
P2: mean dG_25 and mean dG_50 (nats), same bootstrap. Record: Spearman(dS, dG_e) per e, rho_mid - rho_final on the
same draws, Spearman(dG_25, dG_final), snapshot students' own dS_e and Spearman(dS_e, dG_e), per-seed values,
masked-teacher and student accuracy per epoch.
Limit: snapshots are 25 epochs apart; clean views (H-A133 found no view effect).
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3, K=29), is the per-image link between the MaskedKD-minus-Rollout final student "
                "non-target shape loss and the masked-teacher non-target shape loss also there when the patches are "
                "picked by the e25 and e50 snapshot students that produced the KD signal during training?",
    "method": "per image non-target KL(full teacher || final student) after held-out temperature scaling and non-target "
              "KL(full teacher || teacher on the 29 patches the run's own e25 / e50 / final student selects); paired "
              "MaskedKD - Rollout differences; Spearman with image bootstrap; snapshot vs final on the same draws",
    "paper": "Hinton et al. 2015 (dark knowledge); Zhao et al. CVPR 2022 (DKD, NCKD); Abnar & Zuidema ACL 2020 "
             "(attention rollout); Guo et al. ICML 2017 (temperature scaling); Son et al. ECCV 2024 (MaskedKD); "
             "Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261146
K = 29
EPOCHS = (25, 50, None)
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


def _ename(e):
    return "final" if e is None else f"e{e}"


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid]["snapshots"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    Tm = api.teacher.eval()
    batches, tl = [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    tn = _nt_logp(_logp(torch.cat(tl).numpy().astype(numpy.float64), 1.0), y)
    n = len(y)
    folds = [numpy.arange(n) % 2 == 0, numpy.arange(n) % 2 == 1]

    S_kl = {_ename(e): {a: {} for a in RUNS} for e in EPOCHS}
    G_kl = {_ename(e): {a: {} for a in RUNS} for e in EPOCHS}
    per_run = {a: {} for a in RUNS}
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            per_run[a][str(s)] = {"val_acc_last": runs[rid]["val_acc_last"]}
            for e in EPOCHS:
                S = api.student(rid, epoch=e)
                ls, gs = [], []
                for x, _ in batches:
                    with torch.no_grad():
                        out, sc = _scores(api, S, x, a)
                        ls.append(out.logits.float().cpu())
                        gs.append(api.teacher_forward_keepmask(Tm, x, _topk_mask(sc, K)).float().cpu())
                L = torch.cat(ls).numpy().astype(numpy.float64)
                G = torch.cat(gs).numpy().astype(numpy.float64)
                del S
                torch.cuda.empty_cache()
                s_lp_t, Ts = numpy.zeros_like(L), []
                for fit, ev in ((folds[0], folds[1]), (folds[1], folds[0])):
                    T = _fit_T(L[fit], y[fit])
                    Ts.append(round(T, 4))
                    s_lp_t[ev] = _logp(L[ev], T)
                en = _ename(e)
                S_kl[en][a][s] = _nt_kl(tn, _nt_logp(s_lp_t, y))
                G_kl[en][a][s] = _nt_kl(tn, _nt_logp(_logp(G, 1.0), y))
                per_run[a][str(s)][en] = {"T_fit_even_fit_odd": Ts,
                                          "student_nt_kl": round(float(S_kl[en][a][s].mean()), 4),
                                          "signal_nt_kl": round(float(G_kl[en][a][s].mean()), 4),
                                          "signal_acc": round(100 * float((G.argmax(1) == y).mean()), 2),
                                          "student_acc": round(100 * float((L.argmax(1) == y).mean()), 2)}

    seeds = sorted(RUNS["maskedkd"])
    dSe = {en: numpy.mean([S_kl[en]["maskedkd"][s] - S_kl[en]["rollout"][s] for s in seeds], 0) for en in S_kl}
    dGe = {en: numpy.mean([G_kl[en]["maskedkd"][s] - G_kl[en]["rollout"][s] for s in seeds], 0) for en in G_kl}
    dS = dSe["final"]
    dG_mid = 0.5 * (dGe["e25"] + dGe["e50"])
    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def boot(fn):
        b = numpy.array([fn(ix) for ix in idxs])
        return [round(float(numpy.quantile(b, 0.025)), 4), round(float(numpy.quantile(b, 0.975)), 4)]

    def verdict(lo, hi):
        return "positive" if lo > 0 else ("negative" if hi < 0 else "uncertain")

    rho_mid = _spearman(dS, dG_mid)
    rho_mid_ci = boot(lambda ix: _spearman(dS[ix], dG_mid[ix]))
    rho_e = {en: [round(_spearman(dS, dGe[en]), 4)] + boot(lambda ix, en=en: _spearman(dS[ix], dGe[en][ix]))
             for en in dGe}
    diff_ci = boot(lambda ix: _spearman(dS[ix], dG_mid[ix]) - _spearman(dS[ix], dGe["final"][ix]))
    mean_dG = {en: [round(float(dGe[en].mean()), 4)] + boot(lambda ix, en=en: dGe[en][ix].mean()) for en in dGe}
    mean_dS = {en: [round(float(dSe[en].mean()), 4)] + boot(lambda ix, en=en: dSe[en][ix].mean()) for en in dSe}
    own = {en: [round(_spearman(dSe[en], dGe[en]), 4)] + boot(lambda ix, en=en: _spearman(dSe[en][ix], dGe[en][ix]))
           for en in dSe}
    stab = {en: round(_spearman(dGe[en], dGe["final"]), 4) for en in ("e25", "e50")}
    per_seed = {en: [round(_spearman(S_kl["final"]["maskedkd"][s] - S_kl["final"]["rollout"][s],
                                     G_kl[en]["maskedkd"][s] - G_kl[en]["rollout"][s]), 4) for s in seeds]
                for en in G_kl}
    per_seed_dG = {en: _t95([float((G_kl[en]["maskedkd"][s] - G_kl[en]["rollout"][s]).mean()) for s in seeds])
                   for en in G_kl}

    api.write_json("summary", {
        "n_val": int(n), "K": K, "T_grid": [0.1, 10.0, len(T_GRID)],
        "P1_spearman_dS_final_dG_mid": [round(rho_mid, 4)] + rho_mid_ci,
        "P2_mean_dG": {"e25": mean_dG["e25"], "e50": mean_dG["e50"]},
        "spearman_dS_final_dG_by_epoch": rho_e,
        "rho_mid_minus_final": [round(rho_mid - rho_e["final"][0], 4)] + diff_ci,
        "mean_dG_by_epoch": mean_dG, "mean_dS_by_epoch": mean_dS,
        "spearman_own_epoch_dS_dG": own,
        "spearman_dG_epoch_vs_final": stab,
        "per_seed_spearman_dS_final_dG": per_seed, "per_seed_t_dG": per_seed_dG,
        "per_run": per_run,
        "verdict": {"P1": verdict(*rho_mid_ci), "P2_e25": verdict(*mean_dG["e25"][1:]),
                    "P2_e50": verdict(*mean_dG["e50"][1:]), "mid_minus_final": verdict(*diff_ci)}})
