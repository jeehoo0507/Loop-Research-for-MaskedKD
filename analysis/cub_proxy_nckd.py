"""H-C8: same points and test as H-C4 (analysis/cub_proxy_validation.py), but with the non-target part of the soft
label: DKD NCKD (KL of the renormalized non-target class distributions) and top-5 class miss rate.

Why: in H-C4 (analysis_cub_proxy_validation_1031_165024_bc29) ToMe had the worst TCKD at every K yet mid-to-high
student accuracy. ToMe merges instead of dropping tokens, so it may keep the class-relation ("dark knowledge")
part of the soft label. If NCKD orders the points better than TCKD, the next method's target is NCKD.
Only _stats and the stat names differ from H-C4; the H-C4 docstring follows.

H-C4: on CUB, does the masked teacher's soft-label fidelity (KL, DKD TCKD) rank the students' mean val accuracy
across selectors at the same K?

Basis: H-C2a..c measured only teacher-side fidelity; H-C2d (2 final seeds, K=12) gave rollout - maskedkd +0.33 %p,
inside noise. On Waterbirds the full||masked teacher KL ranked students well (analysis_proxy_validation_1030_170914_
4668, Spearman 0.94) but against WGA and with 2 classes. This checks the link on the search dataset with the main
metric before any new score is built on it.

Points: every finished CUB screen seed-0 run of maskedkd / rollout / lateroll / grid / center / random / tome at
K = 12, 29, 49. Each point's masked teacher is scored on the 600 val images
  self   : with the run's own final student (student-dependent selectors), else the selector's fixed rule
  common : with the CUB final full-KD student (seed 1) for student-dependent selectors
and reduced to mean KL(full teacher || masked teacher) and mean TCKD. ToMe runs the teacher with merging at the
harness budget r (copied from analysis/cub_lateroll_refs.py). Spearman (average ranks) with val_acc_last:
within each K and their mean (primary, preregistered in PROPOSALS H-C4), and pooled over all points. A
within-K permutation test (numpy default_rng) gives the null for the mean within-K correlation.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on CUB, does the masked teacher's non-target fidelity (DKD NCKD, top-5 miss) to the full teacher "
                "rank-predict student mean val accuracy across selectors at K=12,29,49, and does it explain ToMe?",
    "method": "keep-only masked-teacher faithfulness (KL, DKD target-class KD term) under each selector's mask, "
              "validated against trained students by within-K Spearman correlation and permutation test",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Zhao et al. CVPR 2022 (DKD); Abnar & Zuidema ACL "
             "2020 (rollout); Bolya et al. ICLR 2023 (ToMe); Son et al. ECCV 2024 (MaskedKD)",
}

KS = (12, 29, 49)
METHODS = ("maskedkd", "rollout", "lateroll", "grid", "center", "random", "tome")
STUDENT_DEP = ("maskedkd", "rollout", "lateroll")
SIDE = 14
EPS = 1e-8
NPERM = 10000
TEACHER_DIM = 768


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _center(B, n, k, dev):
    r = torch.arange(n, device=dev)
    c = (SIDE - 1) / 2
    d = ((r // SIDE - c) ** 2 + (r % SIDE - c) ** 2).float() + r.float() * 1e-6
    return _topk((-d).expand(B, n), k)


def _grid(B, n, k, dev):
    idx = torch.linspace(0, n - 1, k, device=dev).round().long()
    return torch.zeros(B, n, dtype=torch.bool, device=dev).index_fill_(1, idx, True)


def _random(B, n, k, dev, gen):
    return _topk(torch.from_numpy(gen.random((B, n))).float().to(dev), k)


# ---- ToMe (copied from harness/tome.py via analysis/cub_lateroll_refs.py) ----
def _bsm(metric, r):
    t = metric.shape[1]
    r = min(r, (t - 1) // 2)
    if r <= 0:
        return lambda x, mode="mean": x
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        scores[..., 0, :] = -math.inf
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :].sort(dim=1)[0]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x, mode="mean"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    return merge


def _attn_prop(m, x, size):
    B, N, Cd = x.shape
    qkv = m.qkv(x).reshape(B, N, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = m.q_norm(q), m.k_norm(k)
    bias = size.log()[:, None, None, :, 0]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
    out = out.transpose(1, 2).reshape(B, N, Cd)
    if hasattr(m, "norm"):
        out = m.norm(out)
    return m.proj_drop(m.proj(out)), k.mean(1)


def _tome_forward(model, x, r):
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    h = model.norm_pre(h)
    size = torch.ones(h.shape[0], h.shape[1], 1, device=h.device, dtype=h.dtype)
    for blk in model.blocks:
        a, metric = _attn_prop(blk.attn, blk.norm1(h), size)
        h = h + blk.drop_path1(blk.ls1(a))
        merge = _bsm(metric, r)
        h = merge(h * size, mode="sum")
        size = merge(size, mode="sum")
        h = h / size
        h = h + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(h))))
    h = model.norm(h)
    return model.forward_head(h)


def _cost(t):
    return 12 * t * TEACHER_DIM * TEACHER_DIM + 2 * t * t * TEACHER_DIM


def _schedule(r, n0, depth=12):
    out, n = [], n0
    for _ in range(depth):
        out.append(n)
        n -= min(r, (n - 1) // 2)
    return out


def _r_for_budget(keep, n_patch, depth=12):
    target = depth * _cost(int(keep) + 1)
    for r in range(0, n_patch):
        if sum(_cost(t) for t in _schedule(r, n_patch + 1, depth)) <= target:
            return r
    return n_patch // 2
# ---- end ToMe ----


def _scores(api, student, x):
    """maskedkd = last-block CLS attention; rollout = blocks 0..11; lateroll = blocks 6..11 (selectors/lateroll.py)."""
    depth = len(student.blocks)
    with torch.no_grad():
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
        A = [a.float().mean(1) for a in out.attn]
        B, N, _ = A[0].shape
        eye = torch.eye(N, device=A[0].device).expand(B, N, N)
        R, res = None, {"maskedkd": A[-1][:, 0, 1:]}
        for s in range(depth - 1, -1, -1):
            a = 0.5 * A[s] + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else R @ a
            if s == 6:
                res["lateroll"] = R[:, 0, 1:].clone()
        res["rollout"] = R[:, 0, 1:]
    return res


def _stats(full_lp, mask_lp, y):
    pf = full_lp.exp()
    pm = mask_lp.exp()
    kl = (pf * (full_lp - mask_lp)).sum(-1)
    # H-C8: non-target part of DKD (NCKD: KL of the renormalized non-target distributions) and the share of the
    # full teacher's top-5 classes missing from the masked teacher's top-5. Returned in place of (KL, TCKD).
    off = torch.ones_like(pf, dtype=torch.bool).scatter_(1, y[:, None], False)
    qf = (pf * off) / (pf * off).sum(-1, keepdim=True).clamp_min(EPS)
    qm = (pm * off) / (pm * off).sum(-1, keepdim=True).clamp_min(EPS)
    nckd = (qf * ((qf + EPS).log() - (qm + EPS).log())).sum(-1)
    t5f, t5m = pf.topk(5, dim=-1).indices, pm.topk(5, dim=-1).indices
    miss = 1 - (t5f[:, :, None] == t5m[:, None, :]).any(-1).float().mean(-1)
    del kl
    return nckd.sum().item(), miss.sum().item()


def _rank(v):
    v = numpy.asarray(v, dtype=numpy.float64)
    r = numpy.empty(len(v))
    r[v.argsort(kind="stable")] = numpy.arange(len(v), dtype=numpy.float64)
    for val in numpy.unique(v):
        m = v == val
        r[m] = r[m].mean()
    return r


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def run(api):
    runs = api.runs()
    pts, common_id = [], None
    for rid, r in sorted(runs.items()):
        if r.get("dataset") != "cub":
            continue
        m = str(r["method"]).split("@")[0]
        if r["tier"] == "final" and m == "full" and r["seed"] == 1:
            common_id = rid
        if r["tier"] == "screen" and r["seed"] == 0 and m in METHODS and int(r.get("keep") or 0) in KS:
            pts.append({"id": rid, "method": m, "k": int(r["keep"]), "acc": r["val_acc_last"]})
    T = api.teacher.eval()
    n = api.num_patches
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            batches.append((x, y, T(x).float().log_softmax(-1)))
    N = sum(b[0].shape[0] for b in batches)
    dev = batches[0][0].device
    common = api.student(common_id) if common_id else None
    tome_r = {k: _r_for_budget(k, n) for k in KS}

    def fidelity(method, k, student, seed):
        gen = numpy.random.default_rng(seed)
        kl = tc = 0.0
        for x, y, full in batches:
            B = x.shape[0]
            with torch.no_grad():
                if method == "tome":
                    lm = _tome_forward(T, x, tome_r[k]).float().log_softmax(-1)
                else:
                    if method in STUDENT_DEP:
                        keep = _topk(_scores(api, student, x)[method], k)
                    else:
                        keep = {"grid": lambda: _grid(B, n, k, dev), "center": lambda: _center(B, n, k, dev),
                                "random": lambda: _random(B, n, k, dev, gen)}[method]()
                    lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
            a, b = _stats(full, lm, y)
            kl, tc = kl + a, tc + b
        return round(kl / N, 4), round(tc / N, 4)

    fixed_cache = {}
    for p in pts:
        if p["method"] in STUDENT_DEP:
            st = api.student(p["id"])
            p["nckd_self"], p["top5miss_self"] = fidelity(p["method"], p["k"], st, 0)
            del st
            if common is not None:
                p["nckd_common"], p["top5miss_common"] = fidelity(p["method"], p["k"], common, 0)
        else:
            key = (p["method"], p["k"])
            if key not in fixed_cache:
                fixed_cache[key] = fidelity(p["method"], p["k"], None, 20261031)
            p["nckd_self"], p["top5miss_self"] = fixed_cache[key]
            p["nckd_common"], p["top5miss_common"] = fixed_cache[key]
    api.write_json("points", pts)

    res = {"n_points": len(pts), "common_student": common_id, "tome_r": tome_r, "n_val": N}
    gen = numpy.random.default_rng(20261031)
    for stat in ("nckd_self", "top5miss_self", "nckd_common", "top5miss_common"):
        per_k, groups = {}, []
        for k in KS:
            g = [p for p in pts if p["k"] == k and stat in p]
            if len(g) >= 3:
                per_k[str(k)] = {"rho": round(_spearman([p[stat] for p in g], [p["acc"] for p in g]), 4),
                                 "n": len(g)}
                groups.append((numpy.array([p[stat] for p in g]), numpy.array([p["acc"] for p in g])))
        obs = float(numpy.mean([per_k[k]["rho"] for k in per_k])) if per_k else float("nan")
        null = []
        for _ in range(NPERM if groups else 0):
            null.append(numpy.mean([_spearman(s, gen.permutation(a)) for s, a in groups]))
        p_val = (1 + sum(v <= obs for v in null)) / (1 + len(null)) if null else float("nan")
        allp = [p for p in pts if stat in p]
        res[stat] = {"within_k": per_k, "mean_within_k_rho": round(obs, 4), "perm_p_one_sided_neg": round(p_val, 5),
                     "pooled_rho": round(_spearman([p[stat] for p in allp], [p["acc"] for p in allp]), 4)
                     if len(allp) >= 3 else None}
    api.write_json("summary", res)
