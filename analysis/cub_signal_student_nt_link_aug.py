"""H-A133 (round 409, analysis only, no training): H-A131 found on CUB clean val views that the images where the
MaskedKD@29 student loses more non-target shape than Rollout@29 (dS) are weakly the images where the MaskedKD-masked
teacher loses more non-target shape (dG): Spearman +0.21 [+0.13, +0.29]. Training used crop+flip views, and clean-view
masked-teacher statistics can mislead (H-A77). Does the same link hold on training views, and is it stronger there?

Same runs, rules, temperature folds and KL as H-A131 (analysis/cub_signal_student_nt_link.py, inlined). Views: V = 2
fixed RandomResizedCrop+flip views per val image (harness GpuAugment formula, numpy default_rng 20261121, as H-A77),
no RandAugment, erasing or Mixup. Per view: reference = full teacher on that view; S = non-target KL(full teacher ||
temperature-scaled student) on that view (T fit on augmented logits of the other index-parity fold); G = non-target
KL(full teacher || teacher on the 29 patches the run's own final student selects on that view). Per image: mean of
the V views. dS_i = mean_s (S_m - S_r), dG_i = mean_s (G_m - G_r). The clean-view dS, dG (= H-A131) are recomputed.
Primary P1: Spearman(dS_aug, dG_aug), image bootstrap 2000, numpy default_rng(20261143), 95% quantile CI.
P2: mean dG_aug (nats), same bootstrap. Record: rho_aug - rho_clean on the same bootstrap draws, per-seed Spearman,
mean dS_aug, within-method Spearman, masked-teacher accuracy on views.
Limit: final students, not the student of the training step; 2 fixed views.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on CUB (final, seeds 1-3, K=29), on training-style crop+flip views, are the images where the MaskedKD "
                "student loses more non-target shape than Rollout the same images where the MaskedKD-masked teacher "
                "loses more non-target shape?",
    "method": "per image and view non-target KL(full teacher || student) after held-out temperature scaling and "
              "non-target KL(full teacher || teacher on the 29 patches the run's own final student selects), on 2 fixed "
              "RandomResizedCrop+flip views; paired MaskedKD - Rollout differences; Spearman with image bootstrap; "
              "clean-view comparison on the same draws",
    "paper": "Hinton et al. 2015 (dark knowledge); Zhao et al. CVPR 2022 (DKD, NCKD); Abnar & Zuidema ACL 2020 "
             "(attention rollout); Guo et al. ICML 2017 (temperature scaling); Son et al. ECCV 2024 (MaskedKD); "
             "Wah et al. 2011 (CUB)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261143
K = 29
TOPK = 5
V = 2
IMG = 224
T_GRID = numpy.exp(numpy.linspace(numpy.log(0.1), numpy.log(10.0), 400))
RUNS = {
    "maskedkd": {1: "final_maskedkd_k29_s1_1101_104303_b676",
                 2: "final_maskedkd_k29_s2_1101_125958_5b02",
                 3: "final_maskedkd_k29_s3_1101_140937_b00f"},
    "rollout": {1: "final_rollout_k29_s1_1101_104303_3861",
                2: "final_rollout_k29_s2_1101_125958_2ad2",
                3: "final_rollout_k29_s3_1101_140937_8573"},
}


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _logp(L, T):
    Z = L / T
    Z = Z - Z.max(-1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(-1, keepdims=True))


def _fit_T(L, y):
    idx = numpy.arange(len(y))
    losses = [float(-_logp(L, T)[idx, y].mean()) for T in T_GRID]
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


def _grids(n, device):
    """V crop+flip affine grids per val image, harness GpuAugment RandomResizedCrop formula, fixed draws (H-A77)."""
    gen = numpy.random.default_rng(20261121)
    area = gen.uniform(0.08, 1.0, (V, n))
    r = numpy.exp(gen.uniform(math.log(3 / 4), math.log(4 / 3), (V, n)))
    w = numpy.minimum(numpy.sqrt(area * r), 1.0)
    h = numpy.minimum(numpy.sqrt(area / r), 1.0)
    cx = (gen.uniform(0, 1, (V, n)) * 2 - 1) * (1 - w)
    cy = (gen.uniform(0, 1, (V, n)) * 2 - 1) * (1 - h)
    flip = numpy.where(gen.uniform(0, 1, (V, n)) < 0.5, -1.0, 1.0)
    th = numpy.zeros((V, n, 2, 3))
    th[:, :, 0, 0] = w * flip
    th[:, :, 0, 2] = cx
    th[:, :, 1, 1] = h
    th[:, :, 1, 2] = cy
    return torch.tensor(th, dtype=torch.float32, device=device)


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    Tm = api.teacher.eval()
    xs, ys = [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
    y_t = torch.cat(ys)
    n = len(y_t)
    y = y_t.cpu().numpy().astype(int)
    th = _grids(n, y_t.device)
    # views[0] = clean, views[1..V] = augmented; each a list of batches aligned with xs
    views = [xs]
    for v in range(V):
        vb, off = [], 0
        for x in xs:
            b = len(x)
            grid = F.affine_grid(th[v, off:off + b], (b, 3, IMG, IMG), align_corners=False)
            vb.append(F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros",
                                    align_corners=False).to(x.dtype))
            off += b
        views.append(vb)
    tn = []
    t_acc = []
    for vb in views:
        with torch.no_grad():
            tl = torch.cat([Tm(x).float().cpu() for x in vb]).numpy().astype(numpy.float64)
        t_acc.append(round(100 * float((tl.argmax(1) == y).mean()), 2))
        tn.append(_nt_logp(_logp(tl, 1.0), y))
    folds = [numpy.arange(n) % 2 == 0, numpy.arange(n) % 2 == 1]

    S_kl = {c: {a: {} for a in RUNS} for c in ("clean", "aug")}
    G_kl = {c: {a: {} for a in RUNS} for c in ("clean", "aug")}
    per_run = {a: {} for a in RUNS}
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            Ls, Gs = [], []
            for vb in views:
                ls, gs = [], []
                for x in vb:
                    with torch.no_grad():
                        out, sc = _scores(api, S, x, a)
                        ls.append(out.logits.float().cpu())
                        gs.append(api.teacher_forward_keepmask(Tm, x, _topk_mask(sc, K)).float().cpu())
                Ls.append(torch.cat(ls).numpy().astype(numpy.float64))
                Gs.append(torch.cat(gs).numpy().astype(numpy.float64))
            del S
            torch.cuda.empty_cache()
            # temperature: clean fit on clean logits (= H-A131); aug fit on both aug views of the fit fold
            Ts = {"clean": [], "aug": []}
            s_lp = [numpy.zeros_like(L) for L in Ls]
            for fit, ev in ((folds[0], folds[1]), (folds[1], folds[0])):
                T = _fit_T(Ls[0][fit], y[fit])
                Ts["clean"].append(round(T, 4))
                s_lp[0][ev] = _logp(Ls[0][ev], T)
                Ta = _fit_T(numpy.concatenate([Ls[v][fit] for v in range(1, V + 1)]),
                            numpy.concatenate([y[fit]] * V))
                Ts["aug"].append(round(Ta, 4))
                for v in range(1, V + 1):
                    s_lp[v][ev] = _logp(Ls[v][ev], Ta)
            s_v = [_nt_kl(tn[v], _nt_logp(s_lp[v], y)) for v in range(V + 1)]
            g_v = [_nt_kl(tn[v], _nt_logp(_logp(Gs[v], 1.0), y)) for v in range(V + 1)]
            S_kl["clean"][a][s], G_kl["clean"][a][s] = s_v[0], g_v[0]
            S_kl["aug"][a][s] = numpy.mean(s_v[1:], 0)
            G_kl["aug"][a][s] = numpy.mean(g_v[1:], 0)
            per_run[a][str(s)] = {
                "T_fit_even_fit_odd": Ts,
                "student_nt_kl": {c: round(float(S_kl[c][a][s].mean()), 4) for c in ("clean", "aug")},
                "signal_nt_kl": {c: round(float(G_kl[c][a][s].mean()), 4) for c in ("clean", "aug")},
                "signal_acc_clean_v1_v2": [round(100 * float((G.argmax(1) == y).mean()), 2) for G in Gs],
                "student_acc_clean_v1_v2": [round(100 * float((L.argmax(1) == y).mean()), 2) for L in Ls],
                "spearman_student_vs_signal_aug": round(_spearman(S_kl["aug"][a][s], G_kl["aug"][a][s]), 4),
                "val_acc_last": runs[rid]["val_acc_last"]}

    seeds = sorted(RUNS["maskedkd"])
    d = {}
    for c in ("clean", "aug"):
        d[c] = (numpy.mean([S_kl[c]["maskedkd"][s] - S_kl[c]["rollout"][s] for s in seeds], 0),
                numpy.mean([G_kl[c]["maskedkd"][s] - G_kl[c]["rollout"][s] for s in seeds], 0))
    dS, dG = d["aug"]
    cS, cG = d["clean"]
    Sm = numpy.mean([S_kl["aug"]["maskedkd"][s] for s in seeds], 0)
    Sr = numpy.mean([S_kl["aug"]["rollout"][s] for s in seeds], 0)
    Gm = numpy.mean([G_kl["aug"]["maskedkd"][s] for s in seeds], 0)
    Gr = numpy.mean([G_kl["aug"]["rollout"][s] for s in seeds], 0)
    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def boot(fn):
        b = numpy.array([fn(ix) for ix in idxs])
        return [round(float(numpy.quantile(b, 0.025)), 4), round(float(numpy.quantile(b, 0.975)), 4)]

    rho = _spearman(dS, dG)
    rho_ci = boot(lambda ix: _spearman(dS[ix], dG[ix]))
    rho_c = _spearman(cS, cG)
    rho_c_ci = boot(lambda ix: _spearman(cS[ix], cG[ix]))
    rho_diff_ci = boot(lambda ix: _spearman(dS[ix], dG[ix]) - _spearman(cS[ix], cG[ix]))
    dG_ci = boot(lambda ix: dG[ix].mean())
    dS_ci = boot(lambda ix: dS[ix].mean())
    within = {"maskedkd": [round(_spearman(Sm, Gm), 4)] + boot(lambda ix: _spearman(Sm[ix], Gm[ix])),
              "rollout": [round(_spearman(Sr, Gr), 4)] + boot(lambda ix: _spearman(Sr[ix], Gr[ix]))}
    per_seed_rho = [round(_spearman(S_kl["aug"]["maskedkd"][s] - S_kl["aug"]["rollout"][s],
                                    G_kl["aug"]["maskedkd"][s] - G_kl["aug"]["rollout"][s]), 4) for s in seeds]

    def verdict(lo, hi):
        return "positive" if lo > 0 else ("negative" if hi < 0 else "uncertain")

    api.write_json("summary", {
        "n_val": int(n), "K": K, "V": V, "T_grid": [0.1, 10.0, len(T_GRID)],
        "teacher_acc_clean_v1_v2": t_acc,
        "P1_spearman_dS_dG_aug": [round(rho, 4)] + rho_ci,
        "P2_mean_dG_signal_aug": [round(float(dG.mean()), 4)] + dG_ci,
        "mean_dS_student_aug": [round(float(dS.mean()), 4)] + dS_ci,
        "clean_spearman_dS_dG": [round(rho_c, 4)] + rho_c_ci,
        "clean_mean_dS_dG": [round(float(cS.mean()), 4), round(float(cG.mean()), 4)],
        "rho_aug_minus_clean": [round(rho - rho_c, 4)] + rho_diff_ci,
        "per_seed_spearman_dS_dG_aug": per_seed_rho, "per_seed_t_spearman_aug": _t95(per_seed_rho),
        "per_seed_t_dG_aug": _t95([float((G_kl["aug"]["maskedkd"][s] - G_kl["aug"]["rollout"][s]).mean())
                                   for s in seeds]),
        "within_method_spearman_student_vs_signal_aug": within,
        "per_run": per_run,
        "verdict": {"P1": verdict(*rho_ci), "P2": verdict(*dG_ci), "aug_minus_clean": verdict(*rho_diff_ci)}})
