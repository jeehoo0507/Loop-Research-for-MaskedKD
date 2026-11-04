"""H-A137 (round 411, analysis only, no training): H-A135 recorded that on CUB the masked teacher fed the 29 patches
picked by the MaskedKD@29 student gets worse late in training (seed-mean accuracy 62.4 (e25) -> 62.3 (e50) -> 57.3
(final)) while the Rollout@29 one stays near 70. One candidate mechanism: the student's last-block CLS attention
concentrates on a few patches as training goes on, so most of the 29 picks become low-attention filler. Does the
selection score concentrate from e25 to final, and do the images whose score concentrates more lose more signal?

Runs = H-A135 (maskedkd@29, rollout@29, seeds 1-3, snapshots e25, e50, final), clean val views, same score rules.
Per image, run, epoch: p = selection score over the 196 patches normalised to sum 1; conc = 1 - H(p) / log(196);
G = KL(full teacher || teacher on the 29 top-score patches), all classes, T = 1.
dconc_i = mean_s (conc_final - conc_e25); dG_i = mean_s (G_final - G_e25)   (maskedkd runs).
Primary P1: Spearman(dconc, dG) for maskedkd, image bootstrap 2000, numpy default_rng(20261147), 95% quantile CI.
P2: mean dconc for maskedkd, same bootstrap. Record: the same for rollout, mean dG, maskedkd - rollout of mean dconc,
Jaccard of the e25 and final kept sets, top-29 score mass, conc and G per epoch, cross-sectional Spearman(conc, G)
per epoch, per-seed t of mean dconc, masked-teacher accuracy per epoch.
Limit: snapshots 25 epochs apart; entropy over head-mean attention (maskedkd) or rollout row (rollout), the scales
differ between rules, so only within-rule changes are compared.
"""
import numpy
import torch

META = {
    "question": "on CUB (final, seeds 1-3, K=29), does the MaskedKD student's last-block CLS attention concentrate "
                "from e25 to final, and do images whose attention concentrates more lose more masked-teacher signal?",
    "method": "per image normalised entropy of the selection score and KL(full teacher || teacher on the 29 top-score "
              "patches) at e25 / e50 / final snapshots; within-image change e25 -> final averaged over seeds; "
              "Spearman with image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (attention rollout); Darcet et al. ICLR 2024 "
             "(registers, attention artefacts); Hinton et al. 2015; Wah et al. 2011 (CUB)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261147
K = 29
EPOCHS = (25, 50, None)
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


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


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
                cs, ms, gs, ks = [], [], [], []
                for x, _ in batches:
                    with torch.no_grad():
                        sc = _scores(api, S, x, a).clamp_min(0)
                        p = sc / sc.sum(1, keepdim=True).clamp_min(1e-12)
                        H = -(p * p.clamp_min(1e-12).log()).sum(1)
                        cs.append((1 - H / numpy.log(p.shape[1])).cpu())
                        keep = _topk_mask(sc, K)
                        ms.append((p * keep).sum(1).cpu())
                        ks.append(keep.cpu())
                        gs.append(api.teacher_forward_keepmask(Tm, x, keep).float().cpu())
                del S
                torch.cuda.empty_cache()
                G = torch.cat(gs).numpy().astype(numpy.float64)
                en = _ename(e)
                c = torch.cat(cs).numpy().astype(numpy.float64)
                g = (numpy.exp(tlp) * (tlp - _logp(G))).sum(1)
                conc[a].setdefault(en, {})[s] = c
                Gkl[a].setdefault(en, {})[s] = g
                keeps[a].setdefault(en, {})[s] = torch.cat(ks)
                per_run[a][str(s)][en] = {"conc": round(float(c.mean()), 4),
                                          "top29_mass": round(float(torch.cat(ms).mean()), 4),
                                          "signal_kl": round(float(g.mean()), 4),
                                          "signal_acc": round(100 * float((G.argmax(1) == y).mean()), 2)}

    seeds = sorted(RUNS["maskedkd"])
    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def boot(fn):
        b = numpy.array([fn(ix) for ix in idxs])
        return [round(float(numpy.quantile(b, 0.025)), 4), round(float(numpy.quantile(b, 0.975)), 4)]

    def verdict(lo, hi):
        return "positive" if lo > 0 else ("negative" if hi < 0 else "uncertain")

    dconc = {a: numpy.mean([conc[a]["final"][s] - conc[a]["e25"][s] for s in seeds], 0) for a in RUNS}
    dG = {a: numpy.mean([Gkl[a]["final"][s] - Gkl[a]["e25"][s] for s in seeds], 0) for a in RUNS}
    rho = {a: [round(_spearman(dconc[a], dG[a]), 4)] + boot(lambda ix, a=a: _spearman(dconc[a][ix], dG[a][ix]))
           for a in RUNS}
    m_dconc = {a: [round(float(dconc[a].mean()), 4)] + boot(lambda ix, a=a: dconc[a][ix].mean()) for a in RUNS}
    m_dG = {a: [round(float(dG[a].mean()), 4)] + boot(lambda ix, a=a: dG[a][ix].mean()) for a in RUNS}
    dd = dconc["maskedkd"] - dconc["rollout"]
    cross = {a: {en: round(_spearman(numpy.mean([conc[a][en][s] for s in seeds], 0),
                                     numpy.mean([Gkl[a][en][s] for s in seeds], 0)), 4) for en in conc[a]}
             for a in RUNS}
    jac = {}
    for a in RUNS:
        v = []
        for s in seeds:
            k0, k1 = keeps[a]["e25"][s], keeps[a]["final"][s]
            v.append(float(((k0 & k1).sum(1).float() / (k0 | k1).sum(1).float()).mean()))
        jac[a] = [round(x, 4) for x in v]
    per_seed = {a: _t95([float((conc[a]["final"][s] - conc[a]["e25"][s]).mean()) for s in seeds]) for a in RUNS}

    api.write_json("summary", {
        "n_val": int(n), "K": K,
        "P1_spearman_dconc_dG_maskedkd": rho["maskedkd"],
        "P2_mean_dconc_maskedkd": m_dconc["maskedkd"],
        "spearman_dconc_dG": rho, "mean_dconc": m_dconc, "mean_dG": m_dG,
        "maskedkd_minus_rollout_mean_dconc": [round(float(dd.mean()), 4)] + boot(lambda ix: dd[ix].mean()),
        "cross_sectional_spearman_conc_G": cross,
        "jaccard_e25_final_per_seed": jac, "per_seed_t_dconc": per_seed,
        "per_run": per_run,
        "verdict": {"P1": verdict(*rho["maskedkd"][1:]), "P2": verdict(*m_dconc["maskedkd"][1:])}})
