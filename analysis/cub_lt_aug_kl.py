"""H-A173 (round 437, analysis only, no training): follow-up of H-A171/H-A172. On cub_lt final K=29 the trustfilt arm
has the same clean-val mean masked-teacher KL as maskedkd (1.152 vs 1.157) and a median difference that is uncertain
(H-A172), yet +2.3 %p val accuracy. H-A77/78 found that the teacher-region deficit seen on clean centred val views
shrinks on training views (crop+flip). Does the same hold for KL: on training views, is trustfilt's masked teacher
closer to the full teacher (on the same view) than maskedkd's?
Views: H-A77/78 views exactly (V = 2 fixed RandomResizedCrop+flip views per val image, numpy default_rng 20261117,
harness crop formula, no RandAugment, erasing or Mixup). Teacher-region rules use the clean teacher all-block rollout
warped by grid_sample with the crop grid and patch-averaged (as GpuAugment does for ctx.teacher_map); student rules
use the student on the view. Reference = full teacher on the same view (KD target of Full KD on that view).
Per run: mean KL(full teacher || masked teacher) T=1 over all aug views (and over tail-class views), plus the clean-val
mean (check vs H-A171).
P1: trustfilt - maskedkd aug mean KL, paired by seed, t 95% CI (df 2).
P2: Spearman(aug mean KL_run, val_acc_last) over 18 runs; one-sided permutation p (10000, numpy default_rng(20261186)).
Record: (trustfilt - maskedkd) aug - clean; trustfilt - lateroll aug; aug median; tail aug mean; Spearman of clean mean
(H-A171 replica) and of aug tail mean vs tail accuracy; Spearman without maskedkd.
Limit: val images under train-like views (no train split in the API), final students; correlation only.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on cub_lt final K=29 runs, on training-like crop+flip views is trustfilt's masked teacher closer "
                "(KL) to the full teacher on the same view than maskedkd's, and does the aug-view KL rank run accuracy?",
    "method": "keep-only masked-teacher faithfulness (KL to the full teacher on the same RandomResizedCrop+flip view) "
              "per token rule; student/teacher attention rollout (teacher map warped as the training cache) and "
              "last-block CLS attention; paired t CI; Spearman across runs with permutation test",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); Son et al. "
             "ECCV 2024 (MaskedKD); Beyer et al. CVPR 2022 (consistent teaching); Hinton et al. 2015 (KD); "
             "Wah et al. 2011 (CUB-200-2011)",
}

BATCH = 50
N_PERM = 10000
SEED = 20261186
T_CRIT = 4.303
K = 29
WIDEN = 1.5
TAIL = 1
V = 2
IMG = 224
G = 14
RUNS = {
    "maskedkd": ["cub_lt_final_maskedkd_k29_s1_1103_234609_76d1", "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"],
    "rollout": ["cub_lt_final_rollout_k29_s1_1103_234609_8382", "cub_lt_final_rollout_k29_s2_1103_234647_9b4c",
                "cub_lt_final_rollout_k29_s3_1103_234704_0edf"],
    "lateroll": ["cub_lt_final_lateroll_k29_s1_1104_084157_2b15", "cub_lt_final_lateroll_k29_s2_1104_084204_9f09",
                 "cub_lt_final_lateroll_k29_s3_1104_084209_a3f8"],
    "trustmask_v1": ["cub_lt_final_trustmask_v1_k29_s1_1103_234609_10df",
                     "cub_lt_final_trustmask_v1_k29_s2_1103_234647_4e79",
                     "cub_lt_final_trustmask_v1_k29_s3_1103_234704_d748"],
    "trustfilt": ["cub_lt_final_trustfilt_k29_s1_1103_234609_dbe3", "cub_lt_final_trustfilt_k29_s2_1103_234647_43e9",
                  "cub_lt_final_trustfilt_k29_s3_1103_234704_8732"],
    "teacher_topk": ["cub_lt_final_teacher_topk_k29_s1_1103_234627_9ef0",
                     "cub_lt_final_teacher_topk_k29_s2_1103_234655_cfa0",
                     "cub_lt_final_teacher_topk_k29_s3_1103_234716_6d91"],
}


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _kl(tp, sp):
    pt = numpy.exp(tp)
    return (pt * (tp - sp)).sum(1)


def _rank(a):
    r = numpy.empty(len(a))
    r[numpy.argsort(a, kind="stable")] = numpy.arange(len(a))
    return r


def _spearman(a, b):
    return float(numpy.corrcoef(_rank(a), _rank(b))[0, 1])


def _perm(a, b, rng):
    rho = _spearman(a, b)
    null = numpy.array([_spearman(a, rng.permutation(b)) for _ in range(N_PERM)])
    return [round(rho, 4), round(float((null <= rho).mean()), 4), round(float((null >= rho).mean()), 4)]


def _topk_mask(scores, k):
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(1, scores.topk(k, dim=1).indices, True)
    return keep


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _filt(s, t, k):
    m = int(math.ceil(WIDEN * k))
    allowed = t.argsort(1, descending=True).argsort(1) < m
    s = s.masked_fill(~allowed, -1.0)
    return s.argsort(1, descending=True).argsort(1) < k


def _grids(n, device):
    """V crop+flip affine grids per val image, harness GpuAugment RandomResizedCrop formula, fixed draws (H-A77)."""
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


def _keep(api, arm, S, x, tm):
    if arm == "teacher_topk":
        return _topk_mask(tm, K)
    if arm == "maskedkd":
        out = api.student_forward(S, x, attn_layers=[len(S.blocks) - 1])
        return _topk_mask(out.attn[-1].float().mean(1)[:, 0, 1:], K)
    if arm == "rollout":
        out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
        return _topk_mask(_rollout(out.attn), K)
    out = api.student_forward(S, x, attn_layers=list(range(6, len(S.blocks))))
    lr = _rollout([a for a in out.attn if a is not None])
    return _filt(lr, tm, K) if arm == "trustfilt" else _topk_mask(lr, K)


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid].get("val_wga_last")]
                                for arm in RUNS for rid in RUNS[arm]})
    groups = api.val_groups().cpu().numpy()
    Tm = api.teacher.eval()
    xs, ys = [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
    n = sum(len(y) for y in ys)
    th = _grids(n, ys[0].device)
    tail_c = groups[:n] == TAIL

    # views: (kind, x, teacher map); clean batches first per batch then V aug views (H-A78 order)
    views, tl = [], {"clean": [], "aug": []}
    tail_a = []
    off = 0
    for x, y in zip(xs, ys):
        b = len(y)
        with torch.no_grad():
            out = api.student_forward(Tm, x, attn_layers=list(range(len(Tm.blocks))))
            tm = _rollout(out.attn).float()
            tl["clean"].append(Tm(x).float().cpu())
        views.append(("clean", x, tm))
        tmap = (tm / tm.sum(1, keepdim=True)).reshape(b, 1, G, G)
        for v in range(V):
            grid = F.affine_grid(th[v, off:off + b], (b, 3, IMG, IMG), align_corners=False)
            xa = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            ma = F.grid_sample(tmap.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            ma = F.adaptive_avg_pool2d(ma, G).flatten(1)
            with torch.no_grad():
                tl["aug"].append(Tm(xa).float().cpu())
            views.append(("aug", xa, ma))
            tail_a.append(tail_c[off:off + b])
        off += b
    tail_a = numpy.concatenate(tail_a)
    tlp = {k: _logp(torch.cat(v).numpy().astype(numpy.float64)) for k, v in tl.items()}

    per_run, rows = {}, []
    for arm, ids in RUNS.items():
        for rid in ids:
            S = None if arm == "teacher_topk" else api.student(rid)
            gl = {"clean": [], "aug": []}
            for kind, x, tm in views:
                with torch.no_grad():
                    keep = _keep(api, arm, S, x, tm)
                    gl[kind].append(api.teacher_forward_keepmask(Tm, x, keep).float().cpu())
            if S is not None:
                del S
                torch.cuda.empty_cache()
            kl = {k: _kl(tlp[k], _logp(torch.cat(v).numpy().astype(numpy.float64))) for k, v in gl.items()}
            rec = {"aug_mean": float(kl["aug"].mean()), "aug_median": float(numpy.median(kl["aug"])),
                   "aug_tail_mean": float(kl["aug"][tail_a].mean()), "clean_mean": float(kl["clean"].mean()),
                   "clean_tail_mean": float(kl["clean"][tail_c].mean())}
            acc = float(runs[rid]["val_acc_last"])
            tacc = float(runs[rid]["val_wga_last"])
            per_run[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": round(acc, 2),
                            "val_tail_last": tacc, **{k: round(v, 4) for k, v in rec.items()}}
            rows.append({"arm": arm, "seed": runs[rid]["seed"], "acc": acc, "tail": tacc,
                         "aug_minus_clean": rec["aug_mean"] - rec["clean_mean"], **rec})

    rng = numpy.random.default_rng(SEED)
    arms = list(RUNS)
    stats = ("aug_mean", "aug_median", "aug_tail_mean", "clean_mean", "clean_tail_mean", "aug_minus_clean")
    am = {a: {k: float(numpy.mean([r[k] for r in rows if r["arm"] == a])) for k in stats + ("acc", "tail")}
          for a in arms}

    def col(key, skip=None):
        return numpy.array([r[key] for r in rows if r["arm"] != skip])

    def paired(a, b, key):
        da = {r["seed"]: r[key] for r in rows if r["arm"] == a}
        db = {r["seed"]: r[key] for r in rows if r["arm"] == b}
        d = numpy.array([da[s] - db[s] for s in sorted(da)])
        h = T_CRIT * d.std(ddof=1) / math.sqrt(len(d))
        return [round(float(d.mean()), 4), round(float(d.mean() - h), 4), round(float(d.mean() + h), 4),
                [round(float(v), 4) for v in d]]

    def ci_verdict(c):
        return "negative" if c[2] < 0 else ("positive" if c[1] > 0 else "uncertain")

    def verdict(p):
        return "negative" if p[1] < 0.05 else ("positive" if p[2] < 0.05 else "uncertain")

    P1 = paired("trustfilt", "maskedkd", "aug_mean")
    P2 = _perm(col("aug_mean"), col("acc"), rng)
    rec_pairs = {f"trustfilt_minus_maskedkd_{k}": paired("trustfilt", "maskedkd", k)
                 for k in ("clean_mean", "aug_minus_clean", "aug_median", "aug_tail_mean")}
    rec_pairs.update({f"trustfilt_minus_lateroll_{k}": paired("trustfilt", "lateroll", k)
                      for k in ("aug_mean", "clean_mean")})
    rec_pairs["teacher_topk_minus_maskedkd_aug_mean"] = paired("teacher_topk", "maskedkd", "aug_mean")
    rec_pairs["teacher_topk_minus_maskedkd_aug_minus_clean"] = paired("teacher_topk", "maskedkd", "aug_minus_clean")
    rec_sp = {"spearman_clean_mean_acc_runs": _perm(col("clean_mean"), col("acc"), rng),
              "spearman_aug_median_acc_runs": _perm(col("aug_median"), col("acc"), rng),
              "spearman_aug_tail_mean_tail_runs": _perm(col("aug_tail_mean"), col("tail"), rng),
              "spearman_aug_mean_acc_runs_no_maskedkd": _perm(col("aug_mean", "maskedkd"), col("acc", "maskedkd"),
                                                              rng)}

    api.write_json("summary", {
        "n_val": int(n), "n_tail": int(tail_c.sum()), "views_per_image": V, "n_aug_views": int(len(tail_a)),
        "K": K, "n_runs": len(rows),
        "P1_trustfilt_minus_maskedkd_aug_mean_kl": P1,
        "P2_spearman_aug_mean_kl_acc_runs": P2,
        "record": {"pairs": rec_pairs, "spearman": rec_sp,
                   "arm_means": {a: {k: round(v, 4) for k, v in am[a].items()} for a in arms}},
        "per_run": per_run,
        "verdict": {"P1": ci_verdict(P1), "P2": verdict(P2)},
        "note": "pairs are [mean diff, t95 low, t95 high (df 2), per-seed diffs s1..s3]; permutation entries are "
                "[rho, p_lower, p_upper]; one rng shared in order P2 then record spearman keys in listed order; "
                "clean_mean should equal H-A171's mean KL"})
