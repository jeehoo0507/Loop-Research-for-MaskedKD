"""H-A77 (round 366, analysis only, no training): on cub_lt K=29 (gate 2), does the teacher-region rules' deficit in
masked-teacher tail targets (H-A71, H-A75, H-A76: measured on un-augmented val) shrink on training-style views?

H-A71..H-A76 measured the masked teacher on clean centred val images, where teacher_topk / v0 / trustfilt_attn keep
less of the true-class probability than maskedkd, yet their run tail accuracy is not lower. Training never shows the
clean view: every step is a RandomResizedCrop(scale 0.08..1) + flip view, and the teacher-region rules get the
cached clean teacher rollout warped with the same crop (harness/teacher_cache.py, harness/data.py GpuAugment).
Here each val image gets V = 2 fixed crop+flip views (same draws for every run; numpy default_rng 20261117; the
harness crop formula; no RandAugment, erasing or Mixup). Per run and view type (clean / aug):
  q_y(i)   masked teacher true-class probability,   t_acc(i)  [argmax masked teacher == y]
tail = mean over val images of classes 100..199 (aug: over both views).
Teacher-region rules use the clean teacher rollout (as the cache) warped by grid_sample with the crop grid and
patch-averaged, exactly as GpuAugment does for ctx.teacher_map; student rules use the student on the view.

Students (pinned run ids, as in H-A75/H-A76), cub_lt final seeds 1, 2, 3, K = 29. Spearman over the 21 runs with run
bootstrap 95% CI (2000 draws, numpy default_rng 20261116). Paired same-seed t 95% CI (df 2) vs maskedkd. No labels,
groups or positions enter any mask; labels enter only the statistics.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on cub_lt K=29, does the teacher-region rules' deficit in masked-teacher tail true-class probability "
                "vs maskedkd (seen on clean val) shrink on training-style crop+flip views, and do augmented-view "
                "targets rank run tail accuracy better than clean-view targets?",
    "method": "attention rollout (teacher all blocks, warped as the training cache; student all / blocks 6..11); "
              "last-block CLS attention; keep-only masked teacher on RandomResizedCrop+flip views; Spearman with run "
              "bootstrap; paired t CI",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Szegedy et al. 2015 "
             "(RandomResizedCrop); Beyer et al. CVPR 2022 (consistent teaching: KD on the same augmented view); "
             "Wah et al. 2011 (CUB)",
}

K = 29
START = 6
WIDEN = 1.5
NBOOT = 2000
BATCH = 50
TAIL0 = 100
V = 2
IMG = 224
G = 14
ARMS = {
    "maskedkd": {1: "cub_lt_final_maskedkd_k29_s1_1103_234609_76d1",
                 2: "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 3: "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"},
    "rollout": {1: "cub_lt_final_rollout_k29_s1_1103_234609_8382",
                2: "cub_lt_final_rollout_k29_s2_1103_234647_9b4c",
                3: "cub_lt_final_rollout_k29_s3_1103_234704_0edf"},
    "teacher_topk": {1: "cub_lt_final_teacher_topk_k29_s1_1103_234627_9ef0",
                     2: "cub_lt_final_teacher_topk_k29_s2_1103_234655_cfa0",
                     3: "cub_lt_final_teacher_topk_k29_s3_1103_234716_6d91"},
    "trustmask_v1": {1: "cub_lt_final_trustmask_v1_k29_s1_1103_234609_10df",
                     2: "cub_lt_final_trustmask_v1_k29_s2_1103_234647_4e79",
                     3: "cub_lt_final_trustmask_v1_k29_s3_1103_234704_d748"},
    "trustfilt": {1: "cub_lt_final_trustfilt_k29_s1_1103_234609_dbe3",
                  2: "cub_lt_final_trustfilt_k29_s2_1103_234647_43e9",
                  3: "cub_lt_final_trustfilt_k29_s3_1103_234704_8732"},
    "lateroll": {1: "cub_lt_final_lateroll_k29_s1_1104_084157_2b15",
                 2: "cub_lt_final_lateroll_k29_s2_1104_084204_9f09",
                 3: "cub_lt_final_lateroll_k29_s3_1104_084209_a3f8"},
    "trustfilt_attn": {1: "cub_lt_final_trustfilt_attn_k29_s1_1104_103757_81f3",
                       2: "cub_lt_final_trustfilt_attn_k29_s2_1104_103804_0d6e",
                       3: "cub_lt_final_trustfilt_attn_k29_s3_1104_103809_befb"},
}
KEYS = ("q_y_tail_clean", "q_y_tail_aug", "t_acc_tail_clean", "t_acc_tail_aug", "q_y_head_clean", "q_y_head_aug")


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
    s = last if arm == "trustfilt_attn" else late
    return _topk(s.masked_fill(~allowed, -1.0), K)


def _grids(n, device):
    """V crop+flip affine grids per val image, harness GpuAugment RandomResizedCrop formula, fixed draws."""
    gen = numpy.random.default_rng(20261117)
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
    th = _grids(n, y_all.device)

    # views: (name, x, cached-like teacher map, labels); clean first, then V aug views per batch
    views = []
    off = 0
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
        off += b
    tail_c = y_all >= TAIL0
    p_y_full = {k: torch.cat(v) for k, v in p_full.items()}
    y_aug = torch.cat([vw[3] for vw in views if vw[0] == "aug"])
    tail_a = y_aug >= TAIL0
    full_ref = {"p_y_tail_clean": round(p_y_full["clean"][tail_c].mean().item(), 4),
                "p_y_tail_aug": round(p_y_full["aug"][tail_a].mean().item(), 4)}

    rows = []
    for a, ids in ARMS.items():
        for seed, rid in sorted(ids.items()):
            S = api.student(rid)
            acc = {"clean": ([], []), "aug": ([], [])}
            for name, x, tm, y in views:
                keep = _mask(api, a, S, x, tm)
                with torch.no_grad():
                    q = api.teacher_forward_keepmask(T, x, keep).float().softmax(-1)
                acc[name][0].append(q.gather(1, y[:, None])[:, 0])
                acc[name][1].append((q.argmax(1) == y).float())
            row = {"arm": a, "seed": seed, "run": rid, "val_tail": runs[rid]["val_wga_last"],
                   "val_acc": runs[rid]["val_acc_last"]}
            for name, ti in (("clean", tail_c), ("aug", tail_a)):
                qy, ta = torch.cat(acc[name][0]), torch.cat(acc[name][1])
                row[f"q_y_tail_{name}"] = round(qy[ti].mean().item(), 4)
                row[f"q_y_head_{name}"] = round(qy[~ti].mean().item(), 4)
                row[f"t_acc_tail_{name}"] = round(ta[ti].mean().item(), 4)
            rows.append(row)

    gen = numpy.random.default_rng(20261116)
    col = lambda rs, k: numpy.array([r[k] for r in rs], dtype=float)
    vt = col(rows, "val_tail")
    rho = {"q_y_tail_clean_vs_val_tail": _rho_ci(col(rows, "q_y_tail_clean"), vt, gen),
           "q_y_tail_aug_vs_val_tail": _rho_ci(col(rows, "q_y_tail_aug"), vt, gen),
           "t_acc_tail_clean_vs_val_tail": _rho_ci(col(rows, "t_acc_tail_clean"), vt, gen),
           "t_acc_tail_aug_vs_val_tail": _rho_ci(col(rows, "t_acc_tail_aug"), vt, gen),
           "aug_minus_clean_rho_q_y_tail": _diff_rho_ci(col(rows, "q_y_tail_clean"), col(rows, "q_y_tail_aug"), vt,
                                                        gen)}
    by_arm = {}
    for a in ARMS:
        rs = [r for r in rows if r["arm"] == a]
        by_arm[a] = {k: round(float(numpy.mean([r[k] for r in rs])), 4) for k in KEYS + ("val_tail",)}
        by_arm[a]["n_seeds"] = len(rs)

    def t95(d):
        mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
        return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]

    get = lambda a, s, k: next(r[k] for r in rows if r["arm"] == a and r["seed"] == s)
    pair = {}
    for p in ("teacher_topk", "trustfilt", "trustfilt_attn", "trustmask_v1", "lateroll", "rollout"):
        for k in ("q_y_tail", "t_acc_tail"):
            for v in ("clean", "aug"):
                pair[f"{p}-maskedkd:{k}_{v}"] = t95([get(p, s, f"{k}_{v}") - get("maskedkd", s, f"{k}_{v}")
                                                     for s in (1, 2, 3)])
            pair[f"{p}-maskedkd:{k}_aug_minus_clean"] = t95(
                [(get(p, s, f"{k}_aug") - get("maskedkd", s, f"{k}_aug"))
                 - (get(p, s, f"{k}_clean") - get("maskedkd", s, f"{k}_clean")) for s in (1, 2, 3)])
    p1 = pair["teacher_topk-maskedkd:q_y_tail_aug_minus_clean"]
    p2 = rho["q_y_tail_aug_vs_val_tail"]
    out = {"n_val": int(n), "n_tail_val": int(tail_c.sum().item()), "views_per_image": V, "n_runs": len(rows),
           "full_teacher": full_ref, "rho": rho, "paired_t95": pair, "by_arm": by_arm, "runs": rows,
           "verdict": {"P1_teacher_topk_minus_maskedkd_q_y_tail_aug_minus_clean_ci_low_gt0": bool(p1[1] > 0),
                       "P2_rho_q_y_tail_aug_vs_val_tail_ge_0.6_and_gt_clean":
                       bool(p2[0] >= 0.6 and p2[0] > rho["q_y_tail_clean_vs_val_tail"][0])}}
    api.write_json("summary", out)
