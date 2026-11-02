"""H-A15: does the FLOPs saving of a masked teacher show up as wall-clock time of one training step? RESULTS_TABLE
reports total GFLOPs/img (FlopCounterMode) and an img/s column that depends on how many jobs shared the GPU. This
times, alone on the GPU with CUDA events, the parts of train_step (harness/train.py) per method at batch 128 under
bf16 autocast: student forward (with the attention blocks the selector needs), selector, masked teacher forward
(teacher_forward_keepmask), KD/CE loss, and the student gradient (torch.autograd.grad: the validator forbids
.backward(); the optimizer step is method independent and left out). Selectors are copied from harness/baselines.py
and selectors/lateroll.py (analysis files may not import them). Nothing is selected or tuned; lateroll is unchanged
(QUEUE 2026-11-02 08:50).
"""
import numpy as np
import torch
import torch.nn.functional as F

META = {
    "question": "alone on the GPU, how much wall-clock time does one CUB training step take per method and budget "
                "(ce, full, maskedkd/rollout/lateroll/random at K=12,29,49,98), and does the step-time ratio to "
                "full KD follow the total-GFLOPs ratio? Does reading attention weights (non-fused attention in the "
                "student blocks rollout/lateroll need) cost time that FLOPs do not show?",
    "method": "CUDA-event timing of each train_step segment, interleaved repeats, medians",
    "paper": "Son et al. ECCV 2024 (MaskedKD, reports wall-clock savings); Dehghani et al. ICLR 2022 (The "
             "Efficiency Misnomer: FLOPs vs throughput); Abnar & Zuidema ACL 2020 (rollout)",
}

B = 128
TAU = 1.0
ALPHA = 0.5
WARM = 10
STEPS = 30
REPEATS = 3
# (method, K); total GFLOPs/img from RESULTS_TABLE (FlopCounterMode in harness/train.py) for the ratio check
CONFIGS = (("ce", 0), ("full", 196),
           ("maskedkd", 12), ("maskedkd", 29), ("maskedkd", 49), ("maskedkd", 98),
           ("random", 29), ("rollout", 29), ("rollout", 98),
           ("lateroll", 12), ("lateroll", 29), ("lateroll", 49), ("lateroll", 98))
GFLOPS = {"ce_0": 7.64, "full_196": 42.77, "maskedkd_12": 10.07, "maskedkd_29": 12.99, "maskedkd_49": 16.44,
          "maskedkd_98": 25.04, "random_29": 13.00, "rollout_29": 12.99, "rollout_98": 25.04, "lateroll_12": 10.07,
          "lateroll_29": 12.99, "lateroll_49": 16.44, "lateroll_98": 25.04}
NEEDS = {"maskedkd": [11], "rollout": list(range(12)), "lateroll": list(range(6, 12))}


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _rollout(attn, start):
    r = None
    for a in attn[start:]:
        a = a.detach().float().mean(dim=1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _select(method, k, out, gen, n):
    if method == "full":
        return torch.ones(out.logits.shape[0], n, dtype=torch.bool, device=out.logits.device)
    if method == "maskedkd":
        return _topk(out.attn[-1].detach().float().mean(dim=1)[:, 0, 1:], k)
    if method == "random":
        return _topk(torch.rand(out.logits.shape[0], n, device=out.logits.device, generator=gen), k)
    if method == "rollout":
        return _topk(_rollout(out.attn, 0), k)
    return _topk(_rollout(out.attn, 6), k)


def _step(api, student, teacher, params, method, k, x, y, gen, ev):
    ev[0].record()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = api.student_forward(student, x, NEEDS.get(method, []), [])
        ev[1].record()
        if method == "ce":
            ev[2].record()
            ev[3].record()
            loss = F.cross_entropy(out.logits.float(), y)
        else:
            keep = _select(method, k, out, gen, api.num_patches)
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
    return ev


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
    x, y = next(iter(api.val_batches(batch=B)))
    gen = torch.Generator(device=dev)   # default seed; random masks only need to be random, not reproducible
    names = ["student_fwd", "selector", "teacher", "loss_and_grad", "total"]
    times = {f"{m}_{k}": {n: [] for n in names} for m, k in CONFIGS}
    order_rng = np.random.default_rng(0)
    orders = []
    for rep in range(REPEATS):
        order = list(CONFIGS) if rep == 0 else [CONFIGS[i] for i in order_rng.permutation(len(CONFIGS))]
        orders.append([f"{m}_{k}" for m, k in order])
        for m, k in order:
            key = f"{m}_{k}"
            for i in range(WARM + STEPS):
                ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
                _step(api, student, teacher, params, m, k, x, y, gen, ev)
                torch.cuda.synchronize()
                if i < WARM:
                    continue
                seg = [ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2]), ev[2].elapsed_time(ev[3]),
                       ev[3].elapsed_time(ev[4]), ev[0].elapsed_time(ev[4])]
                for n, v in zip(names, seg):
                    times[key][n].append(v)

    summary = {"run_for_student_weights": rid, "dataset": api.dataset, "batch": B, "warmup": WARM,
               "steps_per_repeat": STEPS, "repeats": REPEATS, "orders": orders,
               "device": torch.cuda.get_device_name(dev), "configs": {}}
    full_ms = float(np.median(times["full_196"]["total"]))
    for key, t in times.items():
        tot = np.asarray(t["total"])
        rep_med = [float(np.median(tot[r * STEPS:(r + 1) * STEPS])) for r in range(REPEATS)]
        summary["configs"][key] = {
            **{f"{n}_ms_median": round(float(np.median(v)), 3) for n, v in t.items()},
            "total_ms_iqr": [round(float(np.percentile(tot, 25)), 3), round(float(np.percentile(tot, 75)), 3)],
            "total_ms_median_per_repeat": [round(v, 3) for v in rep_med],
            "img_per_s": round(B / (float(np.median(tot)) / 1000), 1),
            "time_ratio_to_full": round(float(np.median(tot)) / full_ms, 4),
            "gflops_ratio_to_full": round(GFLOPS[key] / GFLOPS["full_196"], 4),
        }
    c = summary["configs"]
    summary["checks"] = {
        "W1_teacher_ms_mk29_over_full": round(c["maskedkd_29"]["teacher_ms_median"] / c["full_196"]["teacher_ms_median"], 4),
        "W2_step_ratio_mk29": c["maskedkd_29"]["time_ratio_to_full"],
        "W2_gflops_ratio_mk29": c["maskedkd_29"]["gflops_ratio_to_full"],
        "W3_lateroll29_over_maskedkd29": round(c["lateroll_29"]["total_ms_median"] / c["maskedkd_29"]["total_ms_median"], 4),
        "W3_rollout29_over_lateroll29": round(c["rollout_29"]["total_ms_median"] / c["lateroll_29"]["total_ms_median"], 4),
        "W4_maskedkd_total_ms_by_k": [c[f"maskedkd_{k}"]["total_ms_median"] for k in (12, 29, 49, 98)],
        "same_cost_lateroll29_over_maskedkd49": round(c["lateroll_29"]["total_ms_median"] / c["maskedkd_49"]["total_ms_median"], 4),
    }
    api.write_json("summary", summary)
