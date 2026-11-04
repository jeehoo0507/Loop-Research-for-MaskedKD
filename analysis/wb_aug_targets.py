"""H-A79 (round 368, analysis only, no training): the Waterbirds version of H-A77/H-A78. On Waterbirds gate 1 (final
K=29, seeds 1-3), do training-view (crop+flip) masked-teacher targets rank the runs' val WGA and overall val accuracy
better than clean-view targets, and does v0's worst-group target advantage over teacher_topk hold in the training
view?

Same views and token rules as H-A77 (analysis/cub_lt_aug_targets.py; V = 2 fixed crop+flip views per val image,
numpy default_rng 20261119, harness crop formula, no RandAugment, erasing or Mixup). Per run and view type:
  q_y(i)   masked teacher true-class probability
  wg       min over the 4 Waterbirds groups of the group mean q_y (aug: over both views)
  all      mean q_y over all val images
Targets = run val_wga_last and val_acc_last. Teacher-region rules use the clean teacher rollout (as the cache) warped
by grid_sample with the crop grid and patch-averaged, exactly as GpuAugment does for ctx.teacher_map; student rules
use the student on the view.

Students (pinned run ids), Waterbirds final seeds 1, 2, 3, K = 29. Spearman over the 18 runs with run bootstrap 95%
CI (2000 draws, numpy default_rng 20261120). Paired same-seed t 95% CI (df 2). No labels, groups or positions enter
any mask; groups and labels enter only the statistics.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on Waterbirds K=29 (gate 1, seeds 1-3), do augmented-view (crop+flip) masked-teacher targets rank "
                "run val WGA and overall val accuracy better than clean-view targets, and is v0's worst-group "
                "target above teacher_topk's in the training view?",
    "method": "attention rollout (teacher all blocks, warped as the training cache; student all / blocks 6..11); "
              "last-block CLS attention; keep-only masked teacher on RandomResizedCrop+flip views sliced by group; "
              "Spearman with run bootstrap; paired t CI",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Szegedy et al. 2015 "
             "(RandomResizedCrop); Beyer et al. CVPR 2022 (consistent teaching); Sagawa et al. ICLR 2020 "
             "(Waterbirds groups)",
}

K = 29
START = 6
WIDEN = 1.5
NBOOT = 2000
BATCH = 50
V = 2
IMG = 224
G = 14
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
KEYS = ("q_y_all_clean", "q_y_all_aug", "q_y_wg_clean", "q_y_wg_aug")


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


def _grids(n, device):
    """V crop+flip affine grids per val image, harness GpuAugment RandomResizedCrop formula, fixed draws."""
    gen = numpy.random.default_rng(20261119)
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


def _diff_rho_ci(a1, a2, b, gen):
    """rho(a2, b) - rho(a1, b), same run bootstrap draws."""
    n = len(b)
    bs = []
    for _ in range(NBOOT):
        i = gen.integers(0, n, n)
        d = _spearman(a2[i], b[i]) - _spearman(a1[i], b[i])
        if d == d:
            bs.append(d)
    bs = numpy.array(bs)
    return [round(_spearman(a2, b) - _spearman(a1, b), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4), int(n)]


def _group_stats(qy, g):
    means = [float(qy[g == k].mean().item()) for k in range(4)]
    return means, min(means)


def run(api):
    runs = api.runs()
    used = {a: {str(s): [rid, runs[rid]["method"], runs[rid]["keep"], runs[rid]["seed"], runs[rid]["tier"],
                         runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]] for s, rid in sorted(ids.items())}
            for a, ids in ARMS.items()}
    api.write_json("students", used)
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    xs, ys = [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
    y_all = torch.cat(ys)
    n = len(y_all)
    g_all = api.val_groups().to(y_all.device)
    th = _grids(n, y_all.device)

    views = []
    off = 0
    g_aug = []
    p_full = {"clean": [], "aug": []}
    for x, y in zip(xs, ys):
        b = len(y)
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        tm = _rollout(out.attn)
        views.append(("clean", x, tm, y))
        p_full["clean"].append(out.logits.float().softmax(-1).gather(1, y[:, None])[:, 0])
        tmap = tm.reshape(b, 1, G, G)
        for v in range(V):
            grid = F.affine_grid(th[v, off:off + b], (b, 3, IMG, IMG), align_corners=False)
            xa = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            ma = F.grid_sample(tmap.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            ma = F.adaptive_avg_pool2d(ma, G).flatten(1)
            with torch.no_grad():
                pa = api.student_forward(T, xa).logits
            p_full["aug"].append(pa.float().softmax(-1).gather(1, y[:, None])[:, 0])
            views.append(("aug", xa, ma, y))
            g_aug.append(g_all[off:off + b])
        off += b
    g_aug = torch.cat(g_aug)
    gv = {"clean": g_all, "aug": g_aug}
    full_ref = {}
    for name in ("clean", "aug"):
        p = torch.cat(p_full[name])
        means, wg = _group_stats(p, gv[name])
        full_ref[f"p_y_all_{name}"] = round(p.mean().item(), 4)
        full_ref[f"p_y_groups_{name}"] = [round(m, 4) for m in means]
        full_ref[f"p_y_wg_{name}"] = round(wg, 4)

    rows = []
    for a, ids in ARMS.items():
        for seed, rid in sorted(ids.items()):
            S = api.student(rid)
            acc = {"clean": [], "aug": []}
            for name, x, tm, y in views:
                keep = _mask(api, a, S, x, tm)
                with torch.no_grad():
                    q = api.teacher_forward_keepmask(T, x, keep).float().softmax(-1)
                acc[name].append(q.gather(1, y[:, None])[:, 0])
            row = {"arm": a, "seed": seed, "run": rid, "val_wga": runs[rid]["val_wga_last"],
                   "val_acc": runs[rid]["val_acc_last"]}
            for name in ("clean", "aug"):
                qy = torch.cat(acc[name])
                means, wg = _group_stats(qy, gv[name])
                row[f"q_y_all_{name}"] = round(qy.mean().item(), 4)
                row[f"q_y_wg_{name}"] = round(wg, 4)
                row[f"q_y_groups_{name}"] = [round(m, 4) for m in means]
            rows.append(row)

    gen = numpy.random.default_rng(20261120)
    col = lambda rs, k: numpy.array([r[k] for r in rs], dtype=float)
    vw, va = col(rows, "val_wga"), col(rows, "val_acc")
    rho = {"q_y_wg_clean_vs_val_wga": _rho_ci(col(rows, "q_y_wg_clean"), vw, gen),
           "q_y_wg_aug_vs_val_wga": _rho_ci(col(rows, "q_y_wg_aug"), vw, gen),
           "aug_minus_clean_rho_q_y_wg": _diff_rho_ci(col(rows, "q_y_wg_clean"), col(rows, "q_y_wg_aug"), vw, gen),
           "q_y_all_clean_vs_val_acc": _rho_ci(col(rows, "q_y_all_clean"), va, gen),
           "q_y_all_aug_vs_val_acc": _rho_ci(col(rows, "q_y_all_aug"), va, gen),
           "aug_minus_clean_rho_q_y_all": _diff_rho_ci(col(rows, "q_y_all_clean"), col(rows, "q_y_all_aug"), va,
                                                       gen)}
    by_arm = {}
    for a in ARMS:
        rs = [r for r in rows if r["arm"] == a]
        by_arm[a] = {k: round(float(numpy.mean([r[k] for r in rs])), 4) for k in KEYS + ("val_wga", "val_acc")}
        by_arm[a]["n_seeds"] = len(rs)

    def t95(d):
        mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
        return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]

    get = lambda a, s, k: next(r[k] for r in rows if r["arm"] == a and r["seed"] == s)
    pair = {}
    for p, ref in (("trustfilt", "teacher_topk"), ("trustfilt", "maskedkd"), ("trustfilt", "lateroll"),
                   ("teacher_topk", "maskedkd"), ("rollout", "maskedkd"), ("trustmask_v1", "maskedkd"),
                   ("lateroll", "maskedkd")):
        for k in ("q_y_wg", "q_y_all"):
            for v in ("clean", "aug"):
                pair[f"{p}-{ref}:{k}_{v}"] = t95([get(p, s, f"{k}_{v}") - get(ref, s, f"{k}_{v}")
                                                  for s in (1, 2, 3)])
            pair[f"{p}-{ref}:{k}_aug_minus_clean"] = t95(
                [(get(p, s, f"{k}_aug") - get(ref, s, f"{k}_aug"))
                 - (get(p, s, f"{k}_clean") - get(ref, s, f"{k}_clean")) for s in (1, 2, 3)])
        pair[f"{p}-{ref}:val_wga"] = t95([get(p, s, "val_wga") - get(ref, s, "val_wga") for s in (1, 2, 3)])
        pair[f"{p}-{ref}:val_acc"] = t95([get(p, s, "val_acc") - get(ref, s, "val_acc") for s in (1, 2, 3)])
    p1 = rho["q_y_wg_aug_vs_val_wga"]
    p2 = pair["trustfilt-teacher_topk:q_y_wg_aug"]
    out = {"n_val": int(n), "group_counts": [int((g_all == k).sum().item()) for k in range(4)],
           "views_per_image": V, "n_runs": len(rows), "full_teacher": full_ref, "rho": rho, "paired_t95": pair,
           "by_arm": by_arm, "runs": rows,
           "verdict": {"P1_rho_q_y_wg_aug_vs_val_wga_ge_0.5_and_gt_clean":
                       bool(p1[0] >= 0.5 and p1[0] > rho["q_y_wg_clean_vs_val_wga"][0]),
                       "P2_trustfilt_minus_teacher_topk_q_y_wg_aug_ci_low_gt0": bool(p2[1] > 0)}}
    api.write_json("summary", out)
