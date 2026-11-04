"""H-A86 (round 375, analysis only, no training): H-A85 found that on Waterbirds group 2 the tokens only the online
maskedkd@29 student keeps (M minus F) lower the masked teacher's q_y (d_M = -0.144), and suggested they are background
tokens. There are no FG/BG labels in the analysis API, so this uses the token's grid distance from the image centre as
a proxy (Waterbirds birds are mostly centred; see QUEUE 10-31 17:30). Do M minus F tokens lie further from the centre
than F minus M tokens, and within an image, does a further-out M minus F set go with a more negative d_M?

Same students, token rule and pieces as H-A85 (last-block head-mean CLS attention, top-K, K = 29, final snapshot,
un-augmented val image, same seed). Position enters only the statistics, never any mask.
Per image: mean normalised centre distance (0 = centre, 1 = corner) and outer-ring fraction (2-token border) of M\\F
and F\\M; d_M = q_y(M) - q_y(I). Images with an empty difference set are left out of that image's statistic.
Per seed and group: mean over images; per-image Spearman(dist(M\\F), d_M) within the group; paired t 95% CI (df 2).
"""
import numpy
import torch

META = {
    "question": "on Waterbirds group 2 (final, seeds 1-3), are the tokens kept only by the online maskedkd@29 student "
                "further from the image centre than those kept only by the frozen student, and does a further-out "
                "online-only set go with a lower marginal masked-teacher q_y?",
    "method": "last-block CLS attention top-K; set differences of the two students' kept tokens; grid centre distance "
              "and outer-ring fraction; keep-only masked teacher marginal q_y; paired t CI over seeds",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 (keep-only); Sagawa et al. ICLR 2020 "
             "(Waterbirds)",
}

K = 29
BATCH = 50
GRID = 14
GROUPS = ("all", "0", "1", "2", "3")
ARMS = {
    "maskedkd": {1: "waterbirds_final_maskedkd_k29_s1_1031_201502_4be8",
                 2: "waterbirds_final_maskedkd_k29_s2_1103_170931_92bd",
                 3: "waterbirds_final_maskedkd_k29_s3_1103_170935_907a"},
    "frozen": {1: "waterbirds_final_maskedkd_frozen_k29_s1_1103_171021_f885",
               2: "waterbirds_final_maskedkd_frozen_k29_s2_1103_171026_2b13",
               3: "waterbirds_final_maskedkd_frozen_k29_s3_1103_171031_b65e"},
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
    masks.update({str(g): groups == g for g in range(4)})

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
                                "ring_MnF_minus_FnM", "n_MnF", "rho_dist_MnF_dM", "rho_ring_MnF_dM",
                                "rho_dist_FnM_dF")}
        for s in (1, 2, 3):
            p = {k: v[m] for k, v in per[s].items()}
            for k in ("dist_MnF", "dist_FnM", "dist_I", "ring_MnF", "ring_FnM", "n_MnF"):
                rows[k].append(float(numpy.nanmean(p[k])))
            rows["dist_MnF_minus_FnM"].append(float(numpy.nanmean(p["dist_MnF"] - p["dist_FnM"])))
            rows["ring_MnF_minus_FnM"].append(float(numpy.nanmean(p["ring_MnF"] - p["ring_FnM"])))
            rows["rho_dist_MnF_dM"].append(_spearman(p["dist_MnF"], p["d_M"]))
            rows["rho_ring_MnF_dM"].append(_spearman(p["ring_MnF"], p["d_M"]))
            rows["rho_dist_FnM_dF"].append(_spearman(p["dist_FnM"], p["d_F"]))
        res[g] = {"n": int(m.sum()), "per_seed": {k: [round(v, 4) for v in vs] for k, vs in rows.items()},
                  "t95": {k: _t95(vs) for k, vs in rows.items()}}
    p1, p2 = res["2"]["t95"]["dist_MnF_minus_FnM"], res["2"]["t95"]["rho_dist_MnF_dM"]
    out = {"k": K, "n_val": int(len(groups)), "by_group": res,
           "verdict": {"P1_group2_dist_diff_ci_low_gt0": bool(p1[1] is not None and p1[1] > 0),
                       "P2_group2_rho_ci_high_lt0": bool(p2[2] is not None and p2[2] < 0)}}
    api.write_json("summary", out)
