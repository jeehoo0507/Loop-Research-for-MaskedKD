"""H-A140 (round 412, analysis only, no training): Aircraft replication of H-A139
(analysis/cub_mask_keep_churn_snap.py, code unchanged except runs, bootstrap seed and text; on Aircraft H-A138 recorded
maskedkd Jaccard 0.52-0.54, rollout 0.78-0.80). Original text: H-A137 found that the MaskedKD@29 kept set changes much more from
e25 to final than the Rollout@29 one (Jaccard 0.57-0.59 vs 0.79-0.80) while the masked-teacher KL grows only for
maskedkd, but it stored only run-mean Jaccard. Do the images whose kept set changes more (churn = 1 - Jaccard) lose
more masked-teacher signal from e25 to final?

Runs = H-A137 (maskedkd@29, rollout@29, seeds 1-3, snapshots e25, e50, final), clean val views, same score rules.
Per image, run, epoch: keep = top-29 patches of the selection score; G = KL(full teacher || teacher on keep), all
classes, T = 1. churn_i = mean_s (1 - Jaccard(keep_e25, keep_final)); dG_i = mean_s (G_final - G_e25).
Primary P1: Spearman(churn, dG) for maskedkd, image bootstrap 2000, numpy default_rng(20261149), 95% quantile CI.
P2: partial Spearman for maskedkd: ranks of churn and dG each residualised (least squares) on [1, rank(G_e25),
rank(dconc)], correlation of the residuals, same bootstrap (dconc = conc_final - conc_e25 as in H-A137).
Record: the same for rollout, mean churn per rule, mean dG by maskedkd churn tercile, per-seed image Spearman,
churn e25 -> e50 and e50 -> final, Spearman(churn, G_e25), masked-teacher accuracy per epoch.
Limit: snapshots 25 epochs apart; correlation, not a proof of mechanism.
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3, K=29), do images whose MaskedKD kept set changes more from e25 to final "
                "lose more masked-teacher signal?",
    "method": "per image Jaccard of the top-29 kept sets at e25 and final snapshots and change of KL(full teacher || "
              "teacher on the kept patches); seed-mean per image; Spearman and partial Spearman with image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (attention rollout); Hinton et al. 2015; "
             "Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261150
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


def _partial(a, b, controls):
    X = numpy.column_stack([numpy.ones(len(a))] + [_rank(c) for c in controls])
    ra, rb = _rank(a), _rank(b)
    ea = ra - X @ numpy.linalg.lstsq(X, ra, rcond=None)[0]
    eb = rb - X @ numpy.linalg.lstsq(X, rb, rcond=None)[0]
    return float(numpy.corrcoef(ea, eb)[0, 1])


def _topk_mask(scores, k):
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(1, scores.topk(k, dim=1).indices, True)
    return keep


def _scores(api, S, x, method):
    if method == "maskedkd":
        out = api.student_forward(S, x, attn_layers=[-1])
        return out.attn[len(S.blocks) - 1].float().mean(1)[:, 0, 1:]
    out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
    r = None
    for a in out.attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _ename(e):
    return "final" if e is None else f"e{e}"


def _jac(k0, k1):
    return ((k0 & k1).sum(1).float() / (k0 | k1).sum(1).float()).numpy().astype(numpy.float64)


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
    tlp = _logp(torch.cat(tl).numpy().astype(numpy.float64))
    n = len(y)

    conc = {a: {} for a in RUNS}
    Gkl = {a: {} for a in RUNS}
    keeps = {a: {} for a in RUNS}
    per_run = {a: {} for a in RUNS}
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            per_run[a][str(s)] = {"val_acc_last": runs[rid]["val_acc_last"]}
            for e in EPOCHS:
                S = api.student(rid, epoch=e)
                cs, gs, ks = [], [], []
                for x, _ in batches:
                    with torch.no_grad():
                        sc = _scores(api, S, x, a).clamp_min(0)
                        p = sc / sc.sum(1, keepdim=True).clamp_min(1e-12)
                        H = -(p * p.clamp_min(1e-12).log()).sum(1)
                        cs.append((1 - H / numpy.log(p.shape[1])).cpu())
                        keep = _topk_mask(sc, K)
                        ks.append(keep.cpu())
                        gs.append(api.teacher_forward_keepmask(Tm, x, keep).float().cpu())
                del S
                torch.cuda.empty_cache()
                G = torch.cat(gs).numpy().astype(numpy.float64)
                en = _ename(e)
                g = (numpy.exp(tlp) * (tlp - _logp(G))).sum(1)
                conc[a].setdefault(en, {})[s] = torch.cat(cs).numpy().astype(numpy.float64)
                Gkl[a].setdefault(en, {})[s] = g
                keeps[a].setdefault(en, {})[s] = torch.cat(ks)
                per_run[a][str(s)][en] = {"signal_kl": round(float(g.mean()), 4),
                                          "signal_acc": round(100 * float((G.argmax(1) == y).mean()), 2)}

    seeds = sorted(RUNS["maskedkd"])
    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def boot(fn):
        b = numpy.array([fn(ix) for ix in idxs])
        return [round(float(numpy.quantile(b, 0.025)), 4), round(float(numpy.quantile(b, 0.975)), 4)]

    def verdict(lo, hi):
        return "positive" if lo > 0 else ("negative" if hi < 0 else "uncertain")

    churn_s = {a: {s: 1 - _jac(keeps[a]["e25"][s], keeps[a]["final"][s]) for s in seeds} for a in RUNS}
    churn = {a: numpy.mean([churn_s[a][s] for s in seeds], 0) for a in RUNS}
    dG_s = {a: {s: Gkl[a]["final"][s] - Gkl[a]["e25"][s] for s in seeds} for a in RUNS}
    dG = {a: numpy.mean([dG_s[a][s] for s in seeds], 0) for a in RUNS}
    G25 = {a: numpy.mean([Gkl[a]["e25"][s] for s in seeds], 0) for a in RUNS}
    dconc = {a: numpy.mean([conc[a]["final"][s] - conc[a]["e25"][s] for s in seeds], 0) for a in RUNS}

    rho = {a: [round(_spearman(churn[a], dG[a]), 4)] + boot(lambda ix, a=a: _spearman(churn[a][ix], dG[a][ix]))
           for a in RUNS}
    prho = {a: [round(_partial(churn[a], dG[a], [G25[a], dconc[a]]), 4)]
            + boot(lambda ix, a=a: _partial(churn[a][ix], dG[a][ix], [G25[a][ix], dconc[a][ix]])) for a in RUNS}
    m_churn = {a: [round(float(churn[a].mean()), 4)] + boot(lambda ix, a=a: churn[a][ix].mean()) for a in RUNS}
    order = numpy.argsort(churn["maskedkd"], kind="stable")
    terc = [round(float(dG["maskedkd"][t].mean()), 4) for t in numpy.array_split(order, 3)]
    per_seed_rho = {a: [round(_spearman(churn_s[a][s], dG_s[a][s]), 4) for s in seeds] for a in RUNS}
    phase = {a: {"e25_e50": round(float(numpy.mean([1 - _jac(keeps[a]["e25"][s], keeps[a]["e50"][s]).mean()
                                                     for s in seeds])), 4),
                 "e50_final": round(float(numpy.mean([1 - _jac(keeps[a]["e50"][s], keeps[a]["final"][s]).mean()
                                                       for s in seeds])), 4)} for a in RUNS}
    churn_G25 = {a: round(_spearman(churn[a], G25[a]), 4) for a in RUNS}

    api.write_json("summary", {
        "n_val": int(n), "K": K,
        "P1_spearman_churn_dG_maskedkd": rho["maskedkd"],
        "P2_partial_spearman_churn_dG_maskedkd": prho["maskedkd"],
        "spearman_churn_dG": rho, "partial_spearman_churn_dG": prho, "mean_churn": m_churn,
        "maskedkd_dG_by_churn_tercile_low_mid_high": terc,
        "per_seed_spearman_churn_dG": per_seed_rho, "mean_churn_by_phase": phase,
        "spearman_churn_G_e25": churn_G25,
        "per_run": per_run,
        "verdict": {"P1": verdict(*rho["maskedkd"][1:]), "P2": verdict(*prho["maskedkd"][1:])}})
