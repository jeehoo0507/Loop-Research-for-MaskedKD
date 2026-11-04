"""H-A99 (round 388, analysis only, no training): replication of H-A98 on Aircraft. Does the H-A95 coupling
(MaskedKD@29 cost concentrated on high masked-teacher-KL images) survive when image difficulty, proxied by full
teacher confidence, is held fixed? Same code as H-A98; only runs (H-A95's) and RNG seed differ.

Per image i and seed s (final weights, seeds 1-3, val 700, un-augmented):
  kl_s   KL(p_T || q_s), p_T = clean full teacher softmax, q_s = keep-only masked teacher with the maskedkd@29 seed-s
         student's own last-block head-mean CLS attention top-29 (harness MaskedKD rule)
  cost_s full_s correct - maskedkd_s correct
f_i = mean_s kl_s, m_i = mean_s cost_s, c_i = full teacher max prob. Terciles of c (label-free strata). Within each
tercile, split at the within-tercile median of f (high half vs low half). Primary contrast D = mean over terciles of
(mean m on high-f half - mean m on low-f half) (%p). Strata and halves fixed from the full sample; paired image
bootstrap 2000, numpy default_rng(20261110).
Record: per-tercile high/low cost, KL, masked teacher acc, conf; per-tercile contrast; partial Spearman(f, m | c)
via rank residuals; secondary D with strata = CE student correctness (number of CE seeds correct, 0..3 -> groups
{0}, {1,2}, {3}); per-seed D with t 95% CI (df 2). Labels enter only correctness; no positions or labels enter any mask.
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3; replication of H-A98 on CUB), is the per-image accuracy cost of MaskedKD@29 relative to Full@196 larger "
                "on high masked-teacher-KL images within full-teacher-confidence terciles (difficulty held fixed)?",
    "method": "keep-only masked teacher with each maskedkd student's own last-block CLS top-29; per-image KL; "
              "within-confidence-tercile median split of KL; stratified high-minus-low cost with image bootstrap",
    "paper": "Hinton et al. 2015 (KD); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 (keep-only); "
             "Maji et al. 2013 (FGVC-Aircraft)",
}

K = 29
BATCH = 50
N_BOOT = 2000
SEED = 20261110
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


def _terc(v):
    edges = numpy.quantile(v, [1 / 3, 2 / 3])
    return numpy.searchsorted(edges, v, side="right"), edges


def _halves(f, strata, n_strata):
    # 1 = high-f half within each stratum (above the within-stratum median), 0 = low half
    h = numpy.zeros(len(f), dtype=int)
    for j in range(n_strata):
        sel = strata == j
        if sel.any():
            h[sel] = (f[sel] > numpy.median(f[sel])).astype(int)
    return h


def _cell(v, strata, h, j, k):
    sel = (strata == j) & (h == k)
    return float(v[sel].mean()) if sel.any() else float("nan")


def _strat_d(m, strata, h, n_strata):
    per = [_cell(m, strata, h, j, 1) - _cell(m, strata, h, j, 0) for j in range(n_strata)]
    return float(numpy.nanmean(per)), per


def _resid_rank(a, c):
    ra = numpy.argsort(numpy.argsort(a)).astype(float)
    rc = numpy.argsort(numpy.argsort(c)).astype(float)
    beta = numpy.polyfit(rc, ra, 1)
    return ra - numpy.polyval(beta, rc)


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

    correct = {a: {} for a in RUNS}
    kl, t_acc = {}, {}
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            depth = len(S.blocks)
            cor, kls, tac = [], [], []
            for (x, y), lp in zip(batches, logp_t):
                with torch.no_grad():
                    if a == "maskedkd":
                        so = api.student_forward(S, x, attn_layers=[depth - 1])
                        keep = _topk(so.attn[depth - 1].float().mean(1)[:, 0, 1:], K)
                        lq = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                        kls.append((lp.exp() * (lp - lq)).sum(-1))
                        tac.append((lq.argmax(-1) == y).float())
                    else:
                        so = api.student_forward(S, x)
                cor.append((so.logits.float().argmax(-1) == y).float())
            correct[a][s] = torch.cat(cor).cpu().numpy()
            if a == "maskedkd":
                kl[s] = torch.cat(kls).cpu().numpy()
                t_acc[s] = torch.cat(tac).cpu().numpy()
            del S

    seeds = sorted(RUNS["maskedkd"])
    cost_seed = {s: correct["full"][s] - correct["maskedkd"][s] for s in seeds}
    m = numpy.mean([cost_seed[s] for s in seeds], 0)
    f = numpy.mean([kl[s] for s in seeds], 0)
    g = numpy.mean([correct["full"][s] - correct["ce"][s] for s in seeds], 0)
    ta = numpy.mean([t_acc[s] for s in seeds], 0)
    n = len(m)
    t3, t_edges = _terc(conf)
    h = _halves(f, t3, 3)
    ce_n = numpy.sum([correct["ce"][s] for s in seeds], 0)
    c3 = numpy.where(ce_n == 0, 0, numpy.where(ce_n == 3, 2, 1))
    h_ce = _halves(f, c3, 3)

    d, per = _strat_d(m, t3, h, 3)
    d_ce, per_ce = _strat_d(m, c3, h_ce, 3)
    pr = float(numpy.corrcoef(_resid_rank(f, conf), _resid_rank(m, conf))[0, 1])
    rng = numpy.random.default_rng(SEED)
    bd, bce, bpr = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        bd.append(_strat_d(m[idx], t3[idx], h[idx], 3)[0])
        bce.append(_strat_d(m[idx], c3[idx], h_ce[idx], 3)[0])
        bpr.append(float(numpy.corrcoef(_resid_rank(f[idx], conf[idx]), _resid_rank(m[idx], conf[idx]))[0, 1]))

    def _ci(v, b, scale):
        return [round(scale * v, 4), round(scale * float(numpy.quantile(b, 0.025)), 4),
                round(scale * float(numpy.quantile(b, 0.975)), 4)]

    ci = _ci(d, bd, 100)
    ci_ce = _ci(d_ce, bce, 100)
    ci_pr = _ci(pr, bpr, 1)

    per_seed = {}
    for s in seeds:
        hs = _halves(kl[s], t3, 3)
        per_seed[s] = _strat_d(cost_seed[s], t3, hs, 3)[0]
    seed_t = _t95([100 * per_seed[s] for s in seeds])

    lo, hi = ci[1], ci[2]
    if lo > 0:
        verdict = "survives_conf_strata"
    elif hi < 0:
        verdict = "reverses_within_strata"
    else:
        verdict = "uncertain"

    def _cells(v, strata, hh):
        return [[round(_cell(v, strata, hh, j, k), 4) for k in (0, 1)] for j in range(3)]

    out = {"k": K, "n_val": n, "mask_cost_mean_pct": round(100 * float(m.mean()), 3),
           "kd_gain_mean_pct": round(100 * float(g.mean()), 3),
           "val_acc": {a: {str(s): [round(100 * float(correct[a][s].mean()), 2),
                                    runs[RUNS[a][s]]["val_acc_last"]] for s in seeds} for a in RUNS},
           "conf_tercile_edges": [round(float(e), 4) for e in t_edges],
           "tercile_n": [int((t3 == j).sum()) for j in range(3)],
           "tercile_low_high_cost_pct": [[round(100 * c, 3) for c in row] for row in _cells(m, t3, h)],
           "tercile_low_high_kl": _cells(f, t3, h),
           "tercile_low_high_masked_teacher_acc": _cells(ta, t3, h),
           "tercile_low_high_conf": _cells(conf, t3, h),
           "tercile_contrast_pct": [round(100 * v, 3) for v in per],
           "D_conf_strata_pct_boot": ci,
           "ce_strata_n": [int((c3 == j).sum()) for j in range(3)],
           "ce_strata_contrast_pct": [round(100 * v, 3) for v in per_ce],
           "D_ce_strata_pct_boot": ci_ce,
           "partial_spearman_kl_cost_given_conf_boot": ci_pr,
           "spearman_kl_teacher_conf": round(_spearman(f, conf), 4),
           "per_seed_D_pct": {str(s): round(100 * per_seed[s], 3) for s in seeds},
           "per_seed_D_t": seed_t,
           "verdict": {"D_conf_strata": verdict, "P1_survives": verdict == "survives_conf_strata"}}
    api.write_json("summary", out)
