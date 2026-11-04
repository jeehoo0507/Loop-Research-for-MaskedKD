"""H-A87 (round 376, analysis only, no training): negative control for H-A86. On Waterbirds the tokens only the online
maskedkd@29 student keeps (M minus F) lie further from the image centre than those only the frozen student keeps
(group 2 +0.135, group 0 +0.229), read as "the loop pulls attention to the background". On cub_lt the gate-3 test
found no behavioural loop (H-A66 online - frozen tail +2.22, unclear). If the outward shift is a sign of the loop, it
should be small on cub_lt; if it shows up just as strongly, it is a generic online-vs-frozen drift and the H-A86
background reading loses its support.

Same token rule and pieces as H-A86 (last-block head-mean CLS attention, top-K, K = 29, final snapshot, un-augmented
val image, same seed), cub_lt students. Position enters only the statistics, never any mask.
Per image: mean normalised centre distance (0 = centre, 1 = corner) and outer-ring fraction (2-token border) of M\\F,
F\\M and I; d_M = q_y(M) - q_y(I), d_F = q_y(F) - q_y(I). Images with an empty difference set are left out of that
image's statistic. Per seed and group (all, 0 = head, 1 = tail): mean over images; per-image Spearman; paired t 95%
CI (df 2).
"""
import numpy
import torch

META = {
    "question": "on cub_lt (final, seeds 1-3), where gate 3 found no behavioural loop, are the tokens kept only by the "
                "online maskedkd@29 student further from the image centre than those kept only by the frozen student "
                "(negative control for the Waterbirds H-A86 outward shift)?",
    "method": "last-block CLS attention top-K; set differences of the two students' kept tokens; grid centre distance "
              "and outer-ring fraction; keep-only masked teacher marginal q_y; paired t CI over seeds",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 (keep-only); Wah et al. 2011 (CUB-200-2011)",
}

K = 29
BATCH = 50
GRID = 14
GROUPS = ("all", "0", "1")
ARMS = {
    "maskedkd": {1: "cub_lt_final_maskedkd_k29_s1_1103_234609_76d1",
                 2: "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 3: "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"},
    "frozen": {1: "cub_lt_final_maskedkd_frozen_k29_s1_1104_074618_8245",
               2: "cub_lt_final_maskedkd_frozen_k29_s2_1104_074622_647c",
               3: "cub_lt_final_maskedkd_frozen_k29_s3_1104_074627_dd2d"},
}


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _t95(d):
    d = [v for v in d if v == v]
    if len(d) < 2:
        return [None, None, None, len(d)]
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _qy(T, api, x, y, keep):
    with torch.no_grad():
        lq = api.teacher_forward_keepmask(T, x, keep).float()
    return lq.softmax(-1).gather(1, y[:, None])[:, 0]


def _rank(a):
    r = numpy.empty(len(a))
    r[numpy.argsort(a, kind="stable")] = numpy.arange(len(a))
    return r


def _spearman(a, b):
    ok = ~(numpy.isnan(a) | numpy.isnan(b))
    if ok.sum() < 3:
        return float("nan")
    return float(numpy.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])


def _set_stats(keep, dist, ring):
    n = keep.sum(1).float()
    d = (keep.float() * dist).sum(1) / n
    r = (keep.float() * ring).sum(1) / n
    d[n == 0] = float("nan")
    r[n == 0] = float("nan")
    return d, r, n


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in ARMS.items()})
    T = api.teacher.eval()
    groups = api.val_groups().numpy()
    masks = {"all": numpy.ones(len(groups), dtype=bool)}
    masks.update({str(g): groups == g for g in range(2)})

    ii, jj = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
    c = (GRID - 1) / 2.0
    dist = (((ii - c) ** 2 + (jj - c) ** 2).sqrt() / (2 * c * c) ** 0.5).flatten()
    ring = ((ii < 2) | (ii >= GRID - 2) | (jj < 2) | (jj >= GRID - 2)).float().flatten()

    per = {}
    for s in (1, 2, 3):
        Sm, Sf = api.student(ARMS["maskedkd"][s]), api.student(ARMS["frozen"][s])
        acc = {k: [] for k in ("dist_MnF", "dist_FnM", "dist_I", "ring_MnF", "ring_FnM", "ring_I", "n_MnF", "d_M",
                               "d_F")}
        for x, y in api.val_batches(batch=BATCH):
            with torch.no_grad():
                om = api.student_forward(Sm, x, attn_layers=[len(Sm.blocks) - 1])
                of = api.student_forward(Sf, x, attn_layers=[len(Sf.blocks) - 1])
            km = _topk(om.attn[len(Sm.blocks) - 1].float().mean(1)[:, 0, 1:], K)
            kf = _topk(of.attn[len(Sf.blocks) - 1].float().mean(1)[:, 0, 1:], K)
            ki = km & kf
            dv, rv = dist.to(km.device), ring.to(km.device)
            for name, kk in (("MnF", km & ~kf), ("FnM", kf & ~km), ("I", ki)):
                d, r, n = _set_stats(kk, dv, rv)
                acc["dist_" + name].append(d)
                acc["ring_" + name].append(r)
                if name == "MnF":
                    acc["n_MnF"].append(n)
            qi = _qy(T, api, x, y, ki)
            acc["d_M"].append(_qy(T, api, x, y, km) - qi)
            acc["d_F"].append(_qy(T, api, x, y, kf) - qi)
        per[s] = {k: torch.cat(v).float().cpu().numpy() for k, v in acc.items()}

    res = {}
    for g in GROUPS:
        m = masks[g]
        rows = {k: [] for k in ("dist_MnF", "dist_FnM", "dist_I", "dist_MnF_minus_FnM", "ring_MnF", "ring_FnM",
                                "ring_MnF_minus_FnM", "n_MnF", "d_M", "d_F", "dF_minus_dM", "rho_dist_MnF_dM",
                                "rho_ring_MnF_dM", "rho_dist_FnM_dF")}
        for s in (1, 2, 3):
            p = {k: v[m] for k, v in per[s].items()}
            for k in ("dist_MnF", "dist_FnM", "dist_I", "ring_MnF", "ring_FnM", "n_MnF", "d_M", "d_F"):
                rows[k].append(float(numpy.nanmean(p[k])))
            rows["dist_MnF_minus_FnM"].append(float(numpy.nanmean(p["dist_MnF"] - p["dist_FnM"])))
            rows["ring_MnF_minus_FnM"].append(float(numpy.nanmean(p["ring_MnF"] - p["ring_FnM"])))
            rows["dF_minus_dM"].append(float(numpy.nanmean(p["d_F"] - p["d_M"])))
            rows["rho_dist_MnF_dM"].append(_spearman(p["dist_MnF"], p["d_M"]))
            rows["rho_ring_MnF_dM"].append(_spearman(p["ring_MnF"], p["d_M"]))
            rows["rho_dist_FnM_dF"].append(_spearman(p["dist_FnM"], p["d_F"]))
        res[g] = {"n": int(m.sum()), "per_seed": {k: [round(v, 4) for v in vs] for k, vs in rows.items()},
                  "t95": {k: _t95(vs) for k, vs in rows.items()}}
    p1 = res["all"]["t95"]["dist_MnF_minus_FnM"]
    p2 = res["1"]["t95"]["dist_MnF_minus_FnM"]
    out = {"k": K, "n_val": int(len(groups)), "by_group": res,
           "verdict": {"P1_all_dist_diff_ci_includes0_or_mean_lt_0.068":
                           bool(p1[1] is not None and (p1[1] <= 0 or p1[0] < 0.068)),
                       "P2_tail_dist_diff_mean_lt_0.068": bool(p2[0] is not None and p2[0] < 0.068)}}
    api.write_json("summary", out)
