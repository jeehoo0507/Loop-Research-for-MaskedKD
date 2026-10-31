"""H-C10: does MaskedKD's last-block CLS attention spend its top-K on high-norm "sink" tokens of the student, and is
that why rollout keeps a more faithful masked teacher at the same K?

Basis: on CUB seed 0 rollout beats maskedkd at every K (K=49 81.67 vs 80.50, K=29 80.83 vs 79.17, K=12 78.33 vs 76.00;
RESULTS_TABLE "Where token choice matters"), and H-C4 (analysis_cub_proxy_validation_1031_165024_bc29) tied the
masked teacher's TCKD to student accuracy within K. Darcet et al. (ICLR 2024, "Vision Transformers Need Registers")
report that ViTs store global information in a few high-norm patch tokens that take much of the late CLS attention
but carry little local content. If the DeiT-Ti student does this, last-block attention picks tokens that are
informative to the student but not to the teacher, while rollout (mixed with identity over 12 blocks) dilutes them.

Sinks (fixed before results): per image, patch tokens whose L2 norm at the block-10 output (= input of the block
whose attention MaskedKD reads) exceeds median + 3 * MAD of that image's 196 patch norms. Position-free.

Per K in (12, 29, 49) and per student (each seed-0 maskedkd / rollout screen run's own student, plus the CUB final
full-KD s1 student as common):
  sink_count            mean sinks per image
  share_sink_mkd/roll   fraction of the maskedkd / rollout top-K that are sinks
  attn_mass_sink        share of last-block CLS->patch attention that falls on sinks
  TCKD / KL             masked teacher (keep-only) for maskedkd, maskedkd_nosink (sinks scored -inf, next tokens
                        fill), rollout
  closure               (TCKD_mkd - TCKD_nosink) / (TCKD_mkd - TCKD_roll), image bootstrap 95% CI (2000, numpy
                        default_rng 20261031) on the common student
Side check (round 92 note on ToMe): the copied ToMe forward with r=0 must equal the plain teacher forward; also the
budget r and token schedule for each K, top-1 agreement and entropy at that r.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on CUB, does MaskedKD's last-block CLS attention put its top-K on high-norm sink tokens of the "
                "student, and does excluding them close the masked-teacher TCKD gap to rollout?",
    "method": "high-norm artifact token detection (per-image median+3MAD of token norms), attention mass on "
              "artifacts, keep-only masked-teacher faithfulness (KL, DKD TCKD); ToMe r=0 sanity check",
    "paper": "Darcet et al. ICLR 2024 (ViTs need registers); Abnar & Zuidema ACL 2020 (rollout); Zhao et al. "
             "CVPR 2022 (DKD); Son et al. ECCV 2024 (MaskedKD); Bolya et al. ICLR 2023 (ToMe)",
}

KS = (12, 29, 49)
EPS = 1e-8
NBOOT = 2000
TEACHER_DIM = 768
MAD_K = 3.0


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _per_img(full_lp, mask_lp, y):
    pf = full_lp.exp()
    pm = mask_lp.exp()
    kl = (pf * (full_lp - mask_lp)).sum(-1)
    pt_f = pf.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    pt_m = pm.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()
    return kl.float().cpu().numpy(), tckd.float().cpu().numpy()


# ---- ToMe (same code as analysis/cub_proxy_cwrong.py, copied from harness/tome.py) ----
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


def _student_view(api, S, x):
    """Last-block CLS attention, rollout over all blocks, sink mask from block-10 output norms."""
    depth = len(S.blocks)
    with torch.no_grad():
        out = api.student_forward(S, x, attn_layers=list(range(depth)), token_layers=[depth - 2])
        A = [a.float().mean(1) for a in out.attn]
        B, N, _ = A[0].shape
        eye = torch.eye(N, device=A[0].device).expand(B, N, N)
        R = None
        for s in range(depth - 1, -1, -1):
            a = 0.5 * A[s] + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else R @ a
        nrm = out.tokens[depth - 2][:, 1:].float().norm(dim=-1)
        med = nrm.median(dim=1, keepdim=True).values
        mad = (nrm - med).abs().median(dim=1, keepdim=True).values
        sink = nrm > med + MAD_K * mad
    return A[-1][:, 0, 1:], R[:, 0, 1:], sink


def run(api):
    runs = api.runs()
    common_id, students = None, []
    for rid, r in sorted(runs.items()):
        if r.get("dataset") != "cub":
            continue
        m = str(r["method"]).split("@")[0]
        if r["tier"] == "final" and m == "full" and r["seed"] == 1:
            common_id = rid
        if r["tier"] == "screen" and r["seed"] == 0 and m in ("maskedkd", "rollout") and int(r.get("keep") or 0) in KS:
            students.append((rid, m, int(r["keep"]), r["val_acc_last"]))
    T = api.teacher.eval()
    n = api.num_patches
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            batches.append((x, y, T(x).float().log_softmax(-1)))
    N = sum(b[0].shape[0] for b in batches)

    # ---- side check: ToMe ----
    tome = {"r0_max_abs_logit_diff": 0.0, "r0_top1_agree": 0.0}
    agree = 0
    for x, y, full in batches[:4]:
        with torch.no_grad():
            l0 = _tome_forward(T, x, 0).float()
            tl = T(x).float()
        tome["r0_max_abs_logit_diff"] = max(tome["r0_max_abs_logit_diff"], (l0 - tl).abs().max().item())
        agree += (l0.argmax(-1) == tl.argmax(-1)).sum().item()
    tome["r0_top1_agree"] = agree / sum(b[0].shape[0] for b in batches[:4])
    for k in KS:
        r = _r_for_budget(k, n)
        ag = ent = 0.0
        for x, y, full in batches:
            with torch.no_grad():
                lm = _tome_forward(T, x, r).float().log_softmax(-1)
            ag += (lm.argmax(-1) == full.argmax(-1)).sum().item()
            ent += -(lm.exp() * lm).sum().item()
        tome[str(k)] = {"r": r, "schedule": _schedule(r, n + 1), "top1_agree_full": round(ag / N, 4),
                        "entropy": round(ent / N, 4)}
    api.write_json("tome_check", tome)

    def view(S, k):
        acc = {"sink_count": [], "share_sink_mkd": [], "share_sink_roll": [], "attn_mass_sink": []}
        per = {v: {"kl": [], "tckd": []} for v in ("maskedkd", "maskedkd_nosink", "rollout")}
        for x, y, full in batches:
            a_last, roll, sink = _student_view(api, S, x)
            km = _topk(a_last, k)
            kr = _topk(roll, k)
            kn = _topk(a_last.masked_fill(sink, -1.0), k)    # attention >= 0, so sinks fill only when needed
            acc["sink_count"].append(sink.sum(1).float().cpu().numpy())
            acc["share_sink_mkd"].append(((km & sink).sum(1).float() / k).cpu().numpy())
            acc["share_sink_roll"].append(((kr & sink).sum(1).float() / k).cpu().numpy())
            acc["attn_mass_sink"].append(((a_last * sink).sum(1) / a_last.sum(1).clamp_min(EPS)).cpu().numpy())
            for v, keep in (("maskedkd", km), ("maskedkd_nosink", kn), ("rollout", kr)):
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                kl, tc = _per_img(full, lm, y)
                per[v]["kl"].append(kl)
                per[v]["tckd"].append(tc)
        acc = {a: numpy.concatenate(b) for a, b in acc.items()}
        per = {v: {s: numpy.concatenate(b) for s, b in d.items()} for v, d in per.items()}
        return acc, per

    def summ(acc, per):
        out = {a: round(float(b.mean()), 4) for a, b in acc.items()}
        for v, d in per.items():
            for s, b in d.items():
                out[f"{s}_{v}"] = round(float(b.mean()), 4)
        return out

    res = {"n_val": N, "common_student": common_id, "mad_k": MAD_K, "points": []}
    gen = numpy.random.default_rng(20261031)
    if common_id is not None:
        C = api.student(common_id)
        res["common"] = {}
        for k in KS:
            acc, per = view(C, k)
            s = summ(acc, per)
            t_m, t_n, t_r = per["maskedkd"]["tckd"], per["maskedkd_nosink"]["tckd"], per["rollout"]["tckd"]
            num, den = t_m.mean() - t_n.mean(), t_m.mean() - t_r.mean()
            boots = []
            for _ in range(NBOOT):
                i = gen.integers(0, N, N)
                d = t_m[i].mean() - t_r[i].mean()
                boots.append((t_m[i].mean() - t_n[i].mean()) / d if abs(d) > EPS else float("nan"))
            boots = numpy.array(boots)
            boots = boots[numpy.isfinite(boots)]
            s["closure"] = round(float(num / den), 4) if abs(den) > EPS else None
            s["closure_ci95"] = [round(float(numpy.quantile(boots, q)), 4) for q in (0.025, 0.975)] if len(boots) else None
            d_mn = t_m - t_n
            s["tckd_mkd_minus_nosink_ci95"] = [round(float(numpy.quantile(
                [d_mn[gen.integers(0, N, N)].mean() for _ in range(NBOOT)], q)), 4) for q in (0.025, 0.975)]
            res["common"][str(k)] = s
        del C
    for rid, m, k, va in students:
        S = api.student(rid)
        acc, per = view(S, k)
        s = summ(acc, per)
        s.update({"id": rid, "method": m, "k": k, "acc": va})
        res["points"].append(s)
        del S
    api.write_json("summary", res)
