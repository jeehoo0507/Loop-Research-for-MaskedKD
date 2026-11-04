"""H-A88 (round 377, analysis only, no training): H-A86 found that on Waterbirds the tokens the online maskedkd@29
student keeps, and its frozen twin does not, lie further from the image centre (a background proxy; no FG/BG labels in
the analysis API), and H-A87 found no such shift on cub_lt where gate 3 saw no loop. Does the same proxy line up with
worst-group accuracy across the gate-1 selectors? Per run: the mean normalised grid centre distance of the K = 29
tokens its rule keeps on group 2 (waterbird on land) val images; then Spearman over the 18 runs against the run's
val WGA, and the paired same-seed v0 (trustfilt) - maskedkd difference.

Same students and token rules as H-A79 (analysis/wb_aug_targets.py), clean val images only, final snapshot. The
teacher-region rule uses the clean teacher rollout. Position enters only the statistics, never any mask.
Spearman over the 18 runs with run bootstrap 95% CI (2000 draws, numpy default_rng 20261123). Paired t 95% CI (df 2).
"""
import math

import numpy
import torch

META = {
    "question": "on Waterbirds K=29 (gate 1, seeds 1-3), does the mean grid centre distance of the tokens each "
                "selector keeps on group 2 rank the runs' val WGA, and does v0 keep more central tokens than "
                "maskedkd there?",
    "method": "attention rollout (teacher all blocks; student all / blocks 6..11); last-block CLS attention; top-K; "
              "grid centre distance and outer-ring fraction of kept tokens by group; Spearman with run bootstrap; "
              "paired t CI",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Sagawa et al. ICLR 2020 "
             "(Waterbirds groups)",
}

K = 29
START = 6
WIDEN = 1.5
NBOOT = 2000
BATCH = 50
GRID = 14
ARMS = {
    "maskedkd": {1: "waterbirds_final_maskedkd_k29_s1_1031_201502_4be8",
                 2: "waterbirds_final_maskedkd_k29_s2_1103_170931_92bd",
                 3: "waterbirds_final_maskedkd_k29_s3_1103_170935_907a"},
    "rollout": {1: "waterbirds_final_rollout_k29_s1_1031_201502_e135",
                2: "waterbirds_final_rollout_k29_s2_1103_170940_2b72",
                3: "waterbirds_final_rollout_k29_s3_1103_170944_dd2c"},
    "teacher_topk": {1: "waterbirds_final_teacher_topk_k29_s1_1103_170914_f015",
                     2: "waterbirds_final_teacher_topk_k29_s2_1103_170918_4535",
                     3: "waterbirds_final_teacher_topk_k29_s3_1103_170923_d12a"},
    "trustmask_v1": {1: "waterbirds_final_trustmask_v1_k29_s1_1103_170855_2e3c",
                     2: "waterbirds_final_trustmask_v1_k29_s2_1103_170951_279c",
                     3: "waterbirds_final_trustmask_v1_k29_s3_1103_170956_fdca"},
    "trustfilt": {1: "waterbirds_final_trustfilt_k29_s1_1103_171121_24be",
                  2: "waterbirds_final_trustfilt_k29_s2_1103_171126_b33d",
                  3: "waterbirds_final_trustfilt_k29_s3_1103_171130_db65"},
    "lateroll": {1: "waterbirds_final_lateroll_k29_s1_1104_025003_b396",
                 2: "waterbirds_final_lateroll_k29_s2_1104_025008_c94e",
                 3: "waterbirds_final_lateroll_k29_s3_1104_025012_629b"},
}
GROUPS = ("all", "0", "1", "2", "3")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    m = r[:, 0, 1:]
    return m / m.sum(1, keepdim=True)


def _mask(api, arm, S, x, tm):
    if arm == "teacher_topk":
        return _topk(tm, K)
    depth = len(S.blocks)
    with torch.no_grad():
        so = api.student_forward(S, x, attn_layers=list(range(depth)))
    last = so.attn[depth - 1].float().mean(1)[:, 0, 1:]
    if arm == "maskedkd":
        return _topk(last, K)
    if arm == "rollout":
        return _topk(_rollout([so.attn[i] for i in range(depth)]), K)
    late = _rollout([so.attn[i] for i in range(START, depth)])
    if arm in ("trustmask_v1", "lateroll"):
        return _topk(late, K)
    allowed = _topk(tm, min(int(math.ceil(WIDEN * K)), tm.shape[1]))
    return _topk(late.masked_fill(~allowed, -1.0), K)


def _rank(v):
    r = numpy.empty(len(v))
    r[numpy.argsort(v, kind="stable")] = numpy.arange(len(v))
    return r


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(numpy.corrcoef(ra, rb)[0, 1])


def _rho_ci(a, b, gen):
    n = len(a)
    bs = []
    for _ in range(NBOOT):
        i = gen.integers(0, n, n)
        r = _spearman(a[i], b[i])
        if r == r:
            bs.append(r)
    bs = numpy.array(bs)
    return [round(_spearman(a, b), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4), int(n)]


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in ARMS.items()})
    T = api.teacher.eval()
    depth_t = len(T.blocks)
    groups = api.val_groups().numpy()
    masks = {"all": numpy.ones(len(groups), dtype=bool)}
    masks.update({str(g): groups == g for g in range(4)})

    ii, jj = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
    c = (GRID - 1) / 2.0
    dist = (((ii - c) ** 2 + (jj - c) ** 2).sqrt() / (2 * c * c) ** 0.5).flatten()
    ring = ((ii < 2) | (ii >= GRID - 2) | (jj < 2) | (jj >= GRID - 2)).float().flatten()

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        batches.append((x, _rollout(out.attn)))

    rows = []
    for a, ids in ARMS.items():
        for seed, rid in sorted(ids.items()):
            S = api.student(rid)
            ds, rs = [], []
            for x, tm in batches:
                keep = _mask(api, a, S, x, tm).float()
                dv, rv = dist.to(keep.device), ring.to(keep.device)
                ds.append((keep * dv).sum(1) / K)
                rs.append((keep * rv).sum(1) / K)
            d = torch.cat(ds).float().cpu().numpy()
            r = torch.cat(rs).float().cpu().numpy()
            row = {"arm": a, "seed": seed, "run": rid, "val_wga": runs[rid]["val_wga_last"],
                   "val_acc": runs[rid]["val_acc_last"]}
            for g in GROUPS:
                row["dist_" + g] = round(float(d[masks[g]].mean()), 4)
                row["ring_" + g] = round(float(r[masks[g]].mean()), 4)
            row["dist_2_minus_0"] = round(row["dist_2"] - row["dist_0"], 4)
            rows.append(row)

    gen = numpy.random.default_rng(20261123)
    col = lambda k: numpy.array([r[k] for r in rows], dtype=float)
    vw, va = col("val_wga"), col("val_acc")
    rho = {"dist_2_vs_val_wga": _rho_ci(col("dist_2"), vw, gen),
           "ring_2_vs_val_wga": _rho_ci(col("ring_2"), vw, gen),
           "dist_all_vs_val_wga": _rho_ci(col("dist_all"), vw, gen),
           "dist_2_minus_0_vs_val_wga": _rho_ci(col("dist_2_minus_0"), vw, gen),
           "dist_all_vs_val_acc": _rho_ci(col("dist_all"), va, gen)}
    by_arm = {}
    for a in ARMS:
        rr = [r for r in rows if r["arm"] == a]
        keys = [k for k in rr[0] if k.startswith(("dist_", "ring_"))] + ["val_wga", "val_acc"]
        by_arm[a] = {k: round(float(numpy.mean([r[k] for r in rr])), 4) for k in keys}
    get = lambda a, s, k: next(r[k] for r in rows if r["arm"] == a and r["seed"] == s)
    pair = {}
    for p, ref in (("trustfilt", "maskedkd"), ("trustfilt", "teacher_topk"), ("trustfilt", "lateroll"),
                   ("rollout", "maskedkd"), ("teacher_topk", "maskedkd"), ("trustmask_v1", "maskedkd"),
                   ("lateroll", "maskedkd")):
        for k in ("dist_2", "ring_2", "dist_0", "dist_all", "val_wga"):
            pair[f"{p}-{ref}:{k}"] = _t95([get(p, s, k) - get(ref, s, k) for s in (1, 2, 3)])
    p1 = rho["dist_2_vs_val_wga"]
    p2 = pair["trustfilt-maskedkd:dist_2"]
    out = {"k": K, "n_val": int(len(groups)), "group_counts": [int((groups == g).sum()) for g in range(4)],
           "n_runs": len(rows), "rho": rho, "paired_t95": pair, "by_arm": by_arm, "runs": rows,
           "verdict": {"P1_rho_dist_2_vs_val_wga_ci_high_lt0": bool(p1[2] < 0),
                       "P2_trustfilt_minus_maskedkd_dist_2_ci_high_lt0": bool(p2[2] < 0)}}
    api.write_json("summary", out)
