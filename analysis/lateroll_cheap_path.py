"""H-A16: is lateroll's extra step time (H-A15: lateroll@29 = 1.144 x maskedkd@29, 1.040 x maskedkd@49 at almost the
same GFLOPs) intrinsic to the selection or a cost of how this harness computes it? H-A15 split the extra 11.2 ms into
student forward +4.1 (six blocks leave the fused attention path to expose weights), selector +2.9 (six float32
197x197 matrix products) and gradient +4.4 (backward of the non-fused attention). Two cheaper paths that compute the
same rollout (blocks 6..11, CLS row) are timed against the harness path:
  clsrow: same student forward, but the CLS row of A11..A6 is taken as a vector-matrix chain e0^T A11 ... A6
          (identical in exact arithmetic to row 0 of the matrix product).
  lite:   fused student forward for all blocks except 11 (as maskedkd); block outputs 5..10 are kept and the
          attention weights of blocks 6..10 are recomputed under no_grad from them; then the clsrow chain.
Mask agreement with the harness lateroll path is measured on the same images in eval mode. Selectors and the timing
loop are copied from analysis/step_walltime.py / selectors/lateroll.py (analysis files may not import them).
Nothing is selected or tuned; lateroll and its runs are unchanged (QUEUE 2026-11-02 08:50).
"""
import numpy as np
import torch
import torch.nn.functional as F

META = {
    "question": "alone on the GPU, does computing the same late-half rollout (blocks 6..11, CLS row) without "
                "non-fused attention in the graph and without matrix-matrix products remove lateroll's step-time "
                "overhead over maskedkd at K=29, and are the selected masks the same as the harness lateroll path?",
    "method": "CUDA-event timing of train_step segments (as H-A15), interleaved repeats, medians; mask overlap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Dao et al. NeurIPS 2022 (fused attention); Dehghani et al. "
             "ICLR 2022 (FLOPs vs throughput)",
}

B = 128
TAU = 1.0
ALPHA = 0.5
WARM = 10
STEPS = 30
REPEATS = 3
K = 29
START = 6
CONFIGS = ("full", "maskedkd_29", "maskedkd_49", "lateroll_29", "lateroll_29_clsrow", "lateroll_29_lite")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _norm(a):
    a = a.detach().float().mean(dim=1)
    a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
    return a / a.sum(dim=-1, keepdim=True)


def _rollout_matrix(attn):            # harness lateroll: r = A11 @ ... @ A6, row 0
    r = None
    for a in attn[START:]:
        a = _norm(a)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _rollout_clsrow(attn):            # e0^T A11 A10 ... A6 as a vector chain
    mats = [_norm(a) for a in attn[START:]]
    v = mats[-1][:, 0:1, :]
    for a in reversed(mats[:-1]):
        v = v @ a
    return v[:, 0, 1:]


def _attn_weights_nograd(blk, h):
    with torch.no_grad():
        x = blk.norm1(h.detach())
        m = blk.attn
        Bn, N, Cd = x.shape
        qkv = m.qkv(x).reshape(Bn, N, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        q, k = m.q_norm(q), m.k_norm(k)
        return ((q * m.scale) @ k.transpose(-2, -1)).softmax(dim=-1)


def _forward_select(api, student, x, cfg):
    """Returns (student output, keep mask or None); the selector segment is timed inside via the caller."""
    if cfg == "full":
        out = api.student_forward(student, x, [], [])
        return out, None
    if cfg.startswith("maskedkd"):
        out = api.student_forward(student, x, [11], [])
        return out, None
    if cfg in ("lateroll_29", "lateroll_29_clsrow"):
        out = api.student_forward(student, x, list(range(START, 12)), [])
        return out, None
    out = api.student_forward(student, x, [11], list(range(START - 1, 11)))
    return out, None


def _select(student, cfg, out, n):
    if cfg == "full":
        return torch.ones(out.logits.shape[0], n, dtype=torch.bool, device=out.logits.device)
    if cfg.startswith("maskedkd"):
        return _topk(out.attn[-1].detach().float().mean(dim=1)[:, 0, 1:], int(cfg.split("_")[1]))
    if cfg == "lateroll_29":
        return _topk(_rollout_matrix(out.attn), K)
    if cfg == "lateroll_29_clsrow":
        return _topk(_rollout_clsrow(out.attn), K)
    attn = [None] * 12
    for i in range(START, 11):
        attn[i] = _attn_weights_nograd(student.blocks[i], out.tokens[i - 1])
    attn[11] = out.attn[11]
    return _topk(_rollout_clsrow(attn), K)


def _step(api, student, teacher, params, cfg, x, y, ev):
    ev[0].record()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = _forward_select(api, student, x, cfg)
        ev[1].record()
        keep = _select(student, cfg, out, api.num_patches)
        ev[2].record()
        with torch.no_grad():
            t_logits = api.teacher_forward_keepmask(teacher, x, keep)
        ev[3].record()
        kd = F.kl_div(F.log_softmax(out.logits.float() / TAU, dim=1), F.softmax(t_logits.float() / TAU, dim=1),
                      reduction="none").sum(1) * TAU * TAU
        ce = F.cross_entropy(out.logits.float(), y, reduction="none")
        loss = ((1 - ALPHA) * ce + ALPHA * kd).mean()
    grads = torch.autograd.grad(loss, params)
    ev[4].record()
    del grads


def _masks(api, student, x):
    # harness/analyze.py runs run() inside an outer autocast, so its weight-cast cache lives for the whole job;
    # casts made here under no_grad must not be cached (they would carry no grad into later timed steps)
    student.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        res = {}
        for cfg in ("lateroll_29", "lateroll_29_clsrow", "lateroll_29_lite"):
            out, _ = _forward_select(api, student, x, cfg)
            res[cfg] = _select(student, cfg, out, api.num_patches)
    student.train()
    ref = res["lateroll_29"]
    agree = {}
    for cfg in ("lateroll_29_clsrow", "lateroll_29_lite"):
        ov = (res[cfg] & ref).sum(1).float() / K
        agree[cfg] = {"mean_overlap": round(float(ov.mean()), 5), "min_overlap": round(float(ov.min()), 5),
                      "frac_identical": round(float((ov == 1).float().mean()), 5)}
    return agree


def run(api):
    dev = api.device
    teacher = api.teacher.eval()
    rid = None
    for r_id, r in sorted(api.runs().items()):
        if r.get("dataset") == api.dataset and r["method"] == "full" and r["tier"] == "final" and r["seed"] == 1:
            rid = r_id
            break
    if rid is None:
        rid = sorted(api.runs())[0]
    student = api.student(rid).train()
    params = [p for p in student.parameters() if p.requires_grad]
    batches = iter(api.val_batches(batch=B))
    x, y = next(batches)

    names =["student_fwd", "selector", "teacher", "loss_and_grad", "total"]
    times = {c: {n: [] for n in names} for c in CONFIGS}
    order_rng = np.random.default_rng(0)
    orders = []
    for rep in range(REPEATS):
        order = list(CONFIGS) if rep == 0 else [CONFIGS[i] for i in order_rng.permutation(len(CONFIGS))]
        orders.append(order)
        for cfg in order:
            for i in range(WARM + STEPS):
                ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
                _step(api, student, teacher, params, cfg, x, y, ev)
                torch.cuda.synchronize()
                if i < WARM:
                    continue
                seg = [ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2]), ev[2].elapsed_time(ev[3]),
                       ev[3].elapsed_time(ev[4]), ev[0].elapsed_time(ev[4])]
                for n, v in zip(names, seg):
                    times[cfg][n].append(v)

    # mask agreement on the first 5 val batches (up to 640 images), eval mode, after all timing
    agree_parts = []
    xs = [x]
    for _ in range(4):
        try:
            xs.append(next(batches)[0])
        except StopIteration:
            break
    n_img = 0
    for xb in xs:
        agree_parts.append((xb.shape[0], _masks(api, student, xb)))
        n_img += xb.shape[0]
    agreement = {}
    for cfg in ("lateroll_29_clsrow", "lateroll_29_lite"):
        w = np.array([n for n, _ in agree_parts], dtype=float)
        agreement[cfg] = {
            "mean_overlap": round(float(np.average([a[cfg]["mean_overlap"] for _, a in agree_parts], weights=w)), 5),
            "min_overlap": min(a[cfg]["min_overlap"] for _, a in agree_parts),
            "frac_identical": round(float(np.average([a[cfg]["frac_identical"] for _, a in agree_parts], weights=w)), 5),
        }

    summary ={"run_for_student_weights": rid, "dataset": api.dataset, "batch": B, "warmup": WARM,
               "steps_per_repeat": STEPS, "repeats": REPEATS, "orders": orders,
               "device": torch.cuda.get_device_name(dev), "mask_agreement_images": n_img,
               "mask_agreement_vs_harness_lateroll": agreement, "configs": {}}
    full_ms = float(np.median(times["full"]["total"]))
    for cfg, t in times.items():
        tot = np.asarray(t["total"])
        summary["configs"][cfg] = {
            **{f"{n}_ms_median": round(float(np.median(v)), 3) for n, v in t.items()},
            "total_ms_iqr": [round(float(np.percentile(tot, 25)), 3), round(float(np.percentile(tot, 75)), 3)],
            "total_ms_median_per_repeat": [round(float(np.median(tot[r * STEPS:(r + 1) * STEPS])), 3)
                                           for r in range(REPEATS)],
            "time_ratio_to_full": round(float(np.median(tot)) / full_ms, 4),
        }
    c = summary["configs"]
    tot = {k: v["total_ms_median"] for k, v in c.items()}
    summary["checks"] = {
        "L1_clsrow_mean_overlap": agreement["lateroll_29_clsrow"]["mean_overlap"],
        "L1_lite_mean_overlap": agreement["lateroll_29_lite"]["mean_overlap"],
        "L2_lite_over_maskedkd29": round(tot["lateroll_29_lite"] / tot["maskedkd_29"], 4),
        "L3_lite_over_maskedkd49": round(tot["lateroll_29_lite"] / tot["maskedkd_49"], 4),
        "L4_clsrow_selector_over_harness_selector": round(c["lateroll_29_clsrow"]["selector_ms_median"]
                                                          / c["lateroll_29"]["selector_ms_median"], 4),
        "report_harness_lateroll29_over_maskedkd29": round(tot["lateroll_29"] / tot["maskedkd_29"], 4),
        "report_clsrow_over_maskedkd29": round(tot["lateroll_29_clsrow"] / tot["maskedkd_29"], 4),
    }
    api.write_json("summary", summary)
