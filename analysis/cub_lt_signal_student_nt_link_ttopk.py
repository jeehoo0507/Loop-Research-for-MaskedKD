"""H-A145 (round 415, analysis only, no training): H-A131/H-A132 (analysis/cub_signal_student_nt_link.py) with the
teacher-selected set as the comparator. H-A131/132 found that images where the MaskedKD@29 student loses more
non-target shape than Rollout@29 are the images where the MaskedKD-masked teacher loses more (Spearman +0.21 / +0.24).
Rollout still selects by student attention. Does the same image-level link hold against teacher_topk@29, whose kept
set is the teacher's own rollout top-29 (no student in the selection)? CUB has no teacher_topk run, so this uses
cub_lt (same CUB val/teacher; train tail classes 100-199 cut to 20%), seeds 1-3.

Per seed s and method a in {maskedkd, rollout, teacher_topk}: student logits (teacher_topk: normal fused forward),
2-fold index-parity temperature as H-A127 -> per image non-target KL(full teacher || student) = S_a. Signal: the
run's final student picks 29 patches by its rule (maskedkd: last-block head-mean CLS attention; rollout: student
rollout 0.5 A + 0.5 I; teacher_topk: teacher rollout of the whole clean image, same formula as harness/teacher_cache.py),
teacher sees only those patches -> non-target KL(full teacher || masked teacher), T=1 = G_a.
dS_i = mean_s (S_maskedkd - S_teacher_topk), dG_i likewise.
Primary P1: Spearman(dS, dG) over val images, image bootstrap 2000, numpy default_rng(20261153), 95% quantile CI.
P2: mean dG (nats), same bootstrap. Record: the same for maskedkd - rollout and rollout - teacher_topk, mean dS,
per-seed Spearman and seed t, within-method Spearman(S, G), signal acc/top-5 overlap, tail/head student acc.
Limit: clean val views and final students; teacher_topk's signal is the same for all seeds (teacher map fixed).
"""
import numpy
import torch

META = {
    "question": "on cub_lt (final, seeds 1-3, K=29), are the images where the MaskedKD student loses more non-target "
                "shape than the teacher_topk student the same images where the MaskedKD-masked teacher loses more "
                "non-target shape than the teacher-top-29-masked teacher?",
    "method": "per image non-target KL(full teacher || student) after held-out temperature scaling and non-target "
              "KL(full teacher || teacher on the 29 patches of the run's own selection rule); paired MaskedKD - "
              "teacher_topk differences; Spearman across images with image bootstrap; seed t CI",
    "paper": "Hinton et al. 2015 (dark knowledge); Zhao et al. CVPR 2022 (DKD, NCKD); Abnar & Zuidema ACL 2020 "
             "(attention rollout); Guo et al. ICML 2017 (temperature scaling); Son et al. ECCV 2024 (MaskedKD); "
             "Wah et al. 2011 (CUB-200-2011)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261153
K = 29
TOPK = 5
TAIL_FROM = 100
T_GRID = numpy.exp(numpy.linspace(numpy.log(0.1), numpy.log(10.0), 400))
RUNS = {
    "maskedkd": {1: "cub_lt_final_maskedkd_k29_s1_1103_234609_76d1",
                 2: "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 3: "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"},
    "rollout": {1: "cub_lt_final_rollout_k29_s1_1103_234609_8382",
                2: "cub_lt_final_rollout_k29_s2_1103_234647_9b4c",
                3: "cub_lt_final_rollout_k29_s3_1103_234704_0edf"},
    "teacher_topk": {1: "cub_lt_final_teacher_topk_k29_s1_1103_234627_9ef0",
                     2: "cub_lt_final_teacher_topk_k29_s2_1103_234655_cfa0",
                     3: "cub_lt_final_teacher_topk_k29_s3_1103_234716_6d91"},
}
PAIRS = (("maskedkd", "teacher_topk"), ("maskedkd", "rollout"), ("rollout", "teacher_topk"))


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


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _scores(api, S, x, method, tmap):
    if method == "maskedkd":
        out = api.student_forward(S, x, attn_layers=[-1])
        return out, out.attn[len(S.blocks) - 1].float().mean(1)[:, 0, 1:]
    if method == "teacher_topk":
        return api.student_forward(S, x), tmap
    out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
    return out, _rollout(out.attn)


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    Tm = api.teacher.eval()
    batches, tl, tr = [], [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
            out = api.student_forward(Tm, x, attn_layers=list(range(len(Tm.blocks))))
            tr.append(_rollout(out.attn).float())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    t_logp = _logp(torch.cat(tl).numpy().astype(numpy.float64), 1.0)
    n = len(y)
    tail = y >= TAIL_FROM
    folds = [numpy.arange(n) % 2 == 0, numpy.arange(n) % 2 == 1]
    tn = _nt_logp(t_logp, y)
    t_top = numpy.argsort(-tn, 1)[:, :TOPK]

    S_kl, G_kl, per_run = {a: {} for a in RUNS}, {a: {} for a in RUNS}, {a: {} for a in RUNS}
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            ls, gs = [], []
            for (x, _), tmap in zip(batches, tr):
                with torch.no_grad():
                    out, sc = _scores(api, S, x, a, tmap)
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
            corr = L.argmax(1) == y
            per_run[a][str(s)] = {"T_fit_even_fit_odd": Ts,
                                  "student_nt_kl": round(float(S_kl[a][s].mean()), 4),
                                  "signal_nt_kl": round(float(G_kl[a][s].mean()), 4),
                                  "signal_top5_overlap": round(float(ov), 4),
                                  "signal_acc": round(100 * float((G.argmax(1) == y).mean()), 2),
                                  "spearman_student_vs_signal": round(_spearman(S_kl[a][s], G_kl[a][s]), 4),
                                  "acc_measured": round(100 * float(corr.mean()), 2),
                                  "acc_tail_measured": round(100 * float(corr[tail].mean()), 2),
                                  "acc_head_measured": round(100 * float(corr[~tail].mean()), 2),
                                  "val_acc_last": runs[rid]["val_acc_last"]}

    seeds = sorted(RUNS["maskedkd"])
    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def boot(fn):
        b = numpy.array([fn(ix) for ix in idxs])
        return [round(float(numpy.quantile(b, 0.025)), 4), round(float(numpy.quantile(b, 0.975)), 4)]

    def verdict(lo, hi):
        return "positive" if lo > 0 else ("negative" if hi < 0 else "uncertain")

    pairs = {}
    for a, b in PAIRS:
        dS = numpy.mean([S_kl[a][s] - S_kl[b][s] for s in seeds], 0)
        dG = numpy.mean([G_kl[a][s] - G_kl[b][s] for s in seeds], 0)
        per_seed_rho = [round(_spearman(S_kl[a][s] - S_kl[b][s], G_kl[a][s] - G_kl[b][s]), 4) for s in seeds]
        pairs[f"{a}-{b}"] = {
            "spearman_dS_dG": [round(_spearman(dS, dG), 4)] + boot(lambda ix: _spearman(dS[ix], dG[ix])),
            "mean_dG_signal": [round(float(dG.mean()), 4)] + boot(lambda ix: dG[ix].mean()),
            "mean_dS_student": [round(float(dS.mean()), 4)] + boot(lambda ix: dS[ix].mean()),
            "spearman_dS_dG_tail_only": round(_spearman(dS[tail], dG[tail]), 4),
            "per_seed_spearman_dS_dG": per_seed_rho, "per_seed_t_spearman": _t95(per_seed_rho),
            "per_seed_t_dG": _t95([float((G_kl[a][s] - G_kl[b][s]).mean()) for s in seeds]),
            "per_seed_t_dS": _t95([float((S_kl[a][s] - S_kl[b][s]).mean()) for s in seeds])}
    within = {}
    for a in RUNS:
        Sa = numpy.mean([S_kl[a][s] for s in seeds], 0)
        Ga = numpy.mean([G_kl[a][s] for s in seeds], 0)
        within[a] = [round(_spearman(Sa, Ga), 4)] + boot(lambda ix: _spearman(Sa[ix], Ga[ix]))
    main = pairs["maskedkd-teacher_topk"]

    api.write_json("summary", {
        "n_val": int(n), "n_tail_val": int(tail.sum()), "K": K, "T_grid": [0.1, 10.0, len(T_GRID)],
        "P1_spearman_dS_dG_maskedkd_minus_teacher_topk": main["spearman_dS_dG"],
        "P2_mean_dG_signal_maskedkd_minus_teacher_topk": main["mean_dG_signal"],
        "pairs": pairs,
        "within_method_spearman_student_vs_signal": within,
        "per_run": per_run,
        "verdict": {"P1": verdict(*main["spearman_dS_dG"][1:]), "P2": verdict(*main["mean_dG_signal"][1:])}})
