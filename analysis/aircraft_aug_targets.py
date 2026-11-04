"""H-A80 (round 369, analysis only, no training): the Aircraft version of H-A78. On Aircraft final K=29 (seeds 1-3,
no groups, primary metric = mean accuracy), do training-view (crop+flip) masked-teacher targets rank the runs' val
accuracy better than clean-view targets, and does the teacher-region rules' target deficit vs maskedkd shrink in the
training view, as on cub_lt?

Same views and token rules as H-A77/H-A79 (V = 2 fixed crop+flip views per val image, numpy default_rng 20261121,
harness crop formula, no RandAugment, erasing or Mixup). Per run and view type:
  q_y_all   mean masked-teacher true-class probability over val images (aug: over both views)
  t_acc_all mean masked-teacher argmax == label
Target = run val_acc_last. Teacher-region rules use the clean teacher rollout (as the cache) warped by grid_sample
with the crop grid and patch-averaged, exactly as GpuAugment does for ctx.teacher_map; student rules use the student
on the view.

Students (pinned run ids), Aircraft final seeds 1, 2, 3, K = 29. Spearman over the 18 runs with run bootstrap 95% CI
(2000 draws, numpy default_rng 20261122). Paired same-seed t 95% CI (df 2). No labels or positions enter any mask;
labels enter only the statistics.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on Aircraft K=29 (final, seeds 1-3), do augmented-view (crop+flip) masked-teacher targets rank run "
                "val accuracy better than clean-view targets, and does teacher_topk's target deficit vs maskedkd "
                "shrink in the training view?",
    "method": "attention rollout (teacher all blocks, warped as the training cache; student all / blocks 6..11); "
              "last-block CLS attention; keep-only masked teacher on RandomResizedCrop+flip views; Spearman with "
              "run bootstrap; paired t CI",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Szegedy et al. 2015 "
             "(RandomResizedCrop); Beyer et al. CVPR 2022 (consistent teaching); Maji et al. 2013 (FGVC-Aircraft)",
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
    "maskedkd": {1: "aircraft_final_maskedkd_k29_s1_1102_132227_6542",
                 2: "aircraft_final_maskedkd_k29_s2_1102_180321_52e2",
                 3: "aircraft_final_maskedkd_k29_s3_1102_180336_347e"},
    "rollout": {1: "aircraft_final_rollout_k29_s1_1102_132227_c3d7",
                2: "aircraft_final_rollout_k29_s2_1102_180325_20b6",
                3: "aircraft_final_rollout_k29_s3_1102_180341_8606"},
    "teacher_topk": {1: "aircraft_final_teacher_topk_k29_s1_1103_074554_cd8b",
                     2: "aircraft_final_teacher_topk_k29_s2_1103_074559_69a5",
                     3: "aircraft_final_teacher_topk_k29_s3_1103_074603_470b"},
    "trustmask_v1": {1: "aircraft_final_trustmask_v1_k29_s1_1103_083809_92a4",
                     2: "aircraft_final_trustmask_v1_k29_s2_1103_083818_f341",
                     3: "aircraft_final_trustmask_v1_k29_s3_1103_083822_5bcd"},
    "trustfilt": {1: "aircraft_final_trustfilt_k29_s1_1103_074527_357e",
                  2: "aircraft_final_trustfilt_k29_s2_1103_074535_db76",
                  3: "aircraft_final_trustfilt_k29_s3_1103_074540_2203"},
    "lateroll": {1: "aircraft_final_lateroll_k29_s1_1103_065756_f0d9",
                 2: "aircraft_final_lateroll_k29_s2_1103_065801_38e1",
                 3: "aircraft_final_lateroll_k29_s3_1103_065805_2130"},
}
KEYS = ("q_y_all_clean", "q_y_all_aug", "t_acc_all_clean", "t_acc_all_aug")


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
                         runs[rid]["val_acc_last"]] for s, rid in sorted(ids.items())}
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

    views = []
    off = 0
    p_full = {"clean": [], "aug": []}
    a_full = {"clean": [], "aug": []}
    for x, y in zip(xs, ys):
        b = len(y)
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        tm = _rollout(out.attn)
        views.append(("clean", x, tm, y))
        lg = out.logits.float()
        p_full["clean"].append(lg.softmax(-1).gather(1, y[:, None])[:, 0])
        a_full["clean"].append((lg.argmax(-1) == y).float())
        tmap = tm.reshape(b, 1, G, G)
        for v in range(V):
            grid = F.affine_grid(th[v, off:off + b], (b, 3, IMG, IMG), align_corners=False)
            xa = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            ma = F.grid_sample(tmap.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            ma = F.adaptive_avg_pool2d(ma, G).flatten(1)
            with torch.no_grad():
                la = api.student_forward(T, xa).logits.float()
            p_full["aug"].append(la.softmax(-1).gather(1, y[:, None])[:, 0])
            a_full["aug"].append((la.argmax(-1) == y).float())
            views.append(("aug", xa, ma, y))
        off += b
    full_ref = {}
    for name in ("clean", "aug"):
        full_ref[f"p_y_all_{name}"] = round(torch.cat(p_full[name]).mean().item(), 4)
        full_ref[f"t_acc_all_{name}"] = round(torch.cat(a_full[name]).mean().item(), 4)

    rows = []
    for a, ids in ARMS.items():
        for seed, rid in sorted(ids.items()):
            S = api.student(rid)
            qs = {"clean": [], "aug": []}
            hs = {"clean": [], "aug": []}
            for name, x, tm, y in views:
                keep = _mask(api, a, S, x, tm)
                with torch.no_grad():
                    lq = api.teacher_forward_keepmask(T, x, keep).float()
                qs[name].append(lq.softmax(-1).gather(1, y[:, None])[:, 0])
                hs[name].append((lq.argmax(-1) == y).float())
            row = {"arm": a, "seed": seed, "run": rid, "val_acc": runs[rid]["val_acc_last"]}
            for name in ("clean", "aug"):
                row[f"q_y_all_{name}"] = round(torch.cat(qs[name]).mean().item(), 4)
                row[f"t_acc_all_{name}"] = round(torch.cat(hs[name]).mean().item(), 4)
            rows.append(row)

    gen = numpy.random.default_rng(20261122)
    col = lambda rs, k: numpy.array([r[k] for r in rs], dtype=float)
    va = col(rows, "val_acc")
    rho = {"q_y_all_clean_vs_val_acc": _rho_ci(col(rows, "q_y_all_clean"), va, gen),
           "q_y_all_aug_vs_val_acc": _rho_ci(col(rows, "q_y_all_aug"), va, gen),
           "aug_minus_clean_rho_q_y_all": _diff_rho_ci(col(rows, "q_y_all_clean"), col(rows, "q_y_all_aug"), va,
                                                       gen),
           "t_acc_all_clean_vs_val_acc": _rho_ci(col(rows, "t_acc_all_clean"), va, gen),
           "t_acc_all_aug_vs_val_acc": _rho_ci(col(rows, "t_acc_all_aug"), va, gen)}
    by_arm = {}
    for a in ARMS:
        rs = [r for r in rows if r["arm"] == a]
        by_arm[a] = {k: round(float(numpy.mean([r[k] for r in rs])), 4) for k in KEYS + ("val_acc",)}
        by_arm[a]["n_seeds"] = len(rs)

    def t95(d):
        mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
        return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]

    get = lambda a, s, k: next(r[k] for r in rows if r["arm"] == a and r["seed"] == s)
    pair = {}
    for p, ref in (("teacher_topk", "maskedkd"), ("trustfilt", "maskedkd"), ("rollout", "maskedkd"),
                   ("trustmask_v1", "maskedkd"), ("lateroll", "maskedkd"), ("trustfilt", "lateroll"),
                   ("trustfilt", "teacher_topk")):
        for k in ("q_y_all", "t_acc_all"):
            for v in ("clean", "aug"):
                pair[f"{p}-{ref}:{k}_{v}"] = t95([get(p, s, f"{k}_{v}") - get(ref, s, f"{k}_{v}")
                                                  for s in (1, 2, 3)])
            pair[f"{p}-{ref}:{k}_aug_minus_clean"] = t95(
                [(get(p, s, f"{k}_aug") - get(ref, s, f"{k}_aug"))
                 - (get(p, s, f"{k}_clean") - get(ref, s, f"{k}_clean")) for s in (1, 2, 3)])
        pair[f"{p}-{ref}:val_acc"] = t95([get(p, s, "val_acc") - get(ref, s, "val_acc") for s in (1, 2, 3)])
    p1 = rho["q_y_all_aug_vs_val_acc"]
    p2 = pair["teacher_topk-maskedkd:q_y_all_aug_minus_clean"]
    out = {"n_val": int(n), "views_per_image": V, "n_runs": len(rows), "full_teacher": full_ref, "rho": rho,
           "paired_t95": pair, "by_arm": by_arm, "runs": rows,
           "verdict": {"P1_rho_q_y_all_aug_vs_val_acc_ge_0.5_and_gt_clean":
                       bool(p1[0] >= 0.5 and p1[0] > rho["q_y_all_clean_vs_val_acc"][0]),
                       "P2_teacher_topk_minus_maskedkd_q_y_all_aug_minus_clean_ci_low_gt0": bool(p2[1] > 0)}}
    api.write_json("summary", out)
