"""H-C2c: on CUB, does late-block rollout (roll6, blocks 6..11; selectors/lateroll.py) keep the masked teacher's
soft label better than every claim reference at the same K, with paired confidence intervals, and how much of the
gap to the teacher's own attention (oracle, analysis only) does it close?

Basis: analysis_cub_rollout_depth_1031_100143_a4b6 (H-C2b) compared roll6 with full rollout, center and random only,
as point means. The claim rule (config.CLAIM_REFS) also needs grid and tome, and H-C3 should not be submitted on a
point mean alone. Here, for the three CUB final students (seed 1) and K = 49, 25, 12:
  1. per-image KL / TCKD of the masked teacher for roll6, roll0 (= harness rollout), roll11 (= maskedkd ranking),
     and student-independent refs grid, center, random, tome (teacher-side merging at the harness budget r), and
     oracle (teacher's own last-block CLS attention; upper reference, never a claim).
  2. paired bootstrap (2000 resamples over val images) 95 % CI of mean TCKD(roll6) - TCKD(ref) and of KL.
  3. head-room: share of the TCKD gap roll0 -> oracle closed by roll6.
ToMe is re-implemented from harness/tome.py (analysis files may import only harness.analysis_api); same matching,
proportional attention and r_for_budget rule.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on CUB, does roll6 (late-block rollout) keep the masked teacher's TCKD/KL lower than maskedkd, "
                "rollout, grid, center, random and tome at the same K, with paired bootstrap CIs, and how close "
                "is it to the teacher-attention oracle?",
    "method": "keep-only masked-teacher faithfulness (KL, DKD TCKD) with paired bootstrap CIs; attention rollout "
              "start-block variant; ToMe teacher at matched budget; teacher-attention oracle as upper reference",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Zhao et al. CVPR 2022 (DKD); Bolya et al. ICLR 2023 (ToMe); "
             "Son et al. ECCV 2024 (MaskedKD); Efron & Tibshirani 1993 (bootstrap)",
}

KS = (49, 25, 12)
SIDE = 14
EPS = 1e-8
NBOOT = 2000
TEACHER_DIM = 768
SCORES = ("roll6", "roll0", "roll11")
REFS = ("grid", "center", "random", "tome", "oracle")


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


def _hash(i, j):
    return torch.frac(torch.sin(i * 12.9898 + j * 78.233) * 43758.5453)


def _random(B, n, k, dev, off):
    i = torch.arange(off, off + B, device=dev, dtype=torch.float64)[:, None]
    j = torch.arange(n, device=dev, dtype=torch.float64)[None, :]
    return _topk(_hash(i, j).float(), k)


# ---- ToMe (copied from harness/tome.py) ----
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


def _rollouts(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad():
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
        A = [a.float().mean(1) for a in out.attn]
        B, N, _ = A[0].shape
        eye = torch.eye(N, device=A[0].device).expand(B, N, N)
        res, R = {}, None
        for s in range(depth - 1, -1, -1):
            a = 0.5 * A[s] + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else R @ a
            if f"roll{s}" in SCORES:
                res[f"roll{s}"] = R[:, 0, 1:].clone()
    return res


def _stats(full_lp, mask_lp, y):
    pf, pm = full_lp.exp(), mask_lp.exp()
    kl = (pf * (full_lp - mask_lp)).sum(-1)
    pt_f = pf.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    pt_m = pm.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()
    return kl, tckd


def _boot_idx(N, dev):
    # v1 (analysis_cub_lateroll_refs_1031_101304_5f82) drew these from the sin hash; the indices were not uniform
    # (CIs did not contain the mean), so its CIs and verdict are void. A seeded numpy generator is used instead.
    rng = numpy.random.default_rng(20261031)
    return torch.from_numpy(rng.integers(0, N, size=(NBOOT, N))).to(dev)    # [NBOOT, N], fixed


def _ci(d, idx):
    m = d[idx].mean(1).sort().values
    lo, hi = m[int(0.025 * NBOOT)].item(), m[int(0.975 * NBOOT) - 1].item()
    assert lo <= d.mean().item() <= hi, "bootstrap CI does not contain the mean"
    return [round(d.mean().item(), 4), round(lo, 4), round(hi, 4)]


def run(api):
    runs = api.runs()
    studs = {}
    for rid, r in sorted(runs.items()):
        if r.get("dataset") != "cub" or r["seed"] != 1 or r["tier"] != "final":
            continue
        studs[f"{r['method']}{int(r.get('keep') or 0)}_s1"] = rid
    T = api.teacher.eval()
    n = api.num_patches
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
            ta = api.student_forward(T, x, attn_layers=[-1]).attn[-1].float().mean(1)[:, 0, 1:]
        batches.append((x, y, full, ta))
    N = sum(b[0].shape[0] for b in batches)
    dev = batches[0][0].device
    idx = _boot_idx(N, dev)
    tome_r = {k: _r_for_budget(k, n) for k in KS}
    tome_ratio = {k: round(sum(_cost(t) for t in _schedule(tome_r[k], n + 1)) / (12 * _cost(k + 1)), 3) for k in KS}

    per = {}                                                      # (name, k) -> (kl [N], tckd [N])

    def add(key, kl, tc):
        o = per.setdefault(key, ([], []))
        o[0].append(kl)
        o[1].append(tc)

    off = 0
    for x, y, full, ta in batches:
        B = x.shape[0]
        for k in KS:
            for m in REFS:
                with torch.no_grad():
                    if m == "tome":
                        lm = _tome_forward(T, x, tome_r[k]).float().log_softmax(-1)
                    else:
                        keep = {"grid": lambda: _grid(B, n, k, dev), "center": lambda: _center(B, n, k, dev),
                                "random": lambda: _random(B, n, k, dev, off), "oracle": lambda: _topk(ta, k)}[m]()
                        lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                add((m, k), *_stats(full, lm, y))
        off += B
    for sname, rid in studs.items():
        student = api.student(rid)
        for x, y, full, ta in batches:
            sc = _rollouts(api, student, x)
            for k in KS:
                for m in SCORES:
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, _topk(sc[m], k)).float().log_softmax(-1)
                    add((f"{sname}:{m}", k), *_stats(full, lm, y))
    per = {key: (torch.cat(v[0]), torch.cat(v[1])) for key, v in per.items()}

    means = {f"{a}@{k}": {"kl": round(v[0].mean().item(), 4), "tckd": round(v[1].mean().item(), 4)}
             for (a, k), v in per.items()}
    api.write_json("means", means)

    out, verdict = {}, {}
    for sname in studs:
        out[sname], verdict[sname] = {}, {}
        for k in KS:
            r6kl, r6tc = per[(f"{sname}:roll6", k)]
            row = {}
            for ref in ("roll0", "roll11") + REFS:
                kl, tc = per[(f"{sname}:{ref}", k)] if ref in SCORES else per[(ref, k)]
                row[ref] = {"d_tckd_ci": _ci(r6tc - tc, idx), "d_kl_ci": _ci(r6kl - kl, idx)}
            o_tc = per[("oracle", k)][1].mean().item()
            r0_tc = per[(f"{sname}:roll0", k)][1].mean().item()
            row["headroom_closed"] = round((r0_tc - r6tc.mean().item()) / max(r0_tc - o_tc, EPS), 4)
            out[sname][str(k)] = row
            # preregistered reading (PROPOSALS H-C2c): CI upper bound of TCKD difference < 0
            verdict[sname][str(k)] = {ref: row[ref]["d_tckd_ci"][2] < 0
                                      for ref in ("roll0", "roll11", "grid", "center", "random", "tome")}
    api.write_json("paired_ci", out)
    api.write_json("verdict", verdict)
    api.write_json("summary", {"students": studs, "n_val": N, "ks": list(KS), "nboot": NBOOT,
                               "tome_r": tome_r, "tome_cost_ratio": tome_ratio, "verdict": verdict})
