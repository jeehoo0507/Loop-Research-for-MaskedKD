"""H-W12: does a centre prior on the student's attention rollout keep the masked teacher faithful on the worst
group at every K, where the plain centre disc wins only at K=12 and plain rollout only at K=29/49?

Basis:
  * screen seed 0 (RESULTS_TABLE): WGA center 56.39 > rollout 45.11 at K=12, rollout > center at K=29 (65.41 vs
    62.41) and K=49 (64.66 vs 60.15). Neither rule wins at every K.
  * analysis_proxy_validation_1030_170914_4668: KL(full teacher || masked teacher) on the worst group, measured with
    the run's own student, rank-predicts student WGA (|Spearman| 0.94).
  * analysis_student_sufficiency_1030_173025_6f96: rollout top-12 read from the center@12 student gives kl_worst
    0.603, below the centre disc's 0.694, while rollout@12 read from its own (rollout-trained) student gives 0.700.
    A student trained on centre-like selections may carry a rollout map that is better for the teacher.

Selections (clean val, K in 12, 29, 49):
  center              fixed centre disc
  top_rollout         rollout top-K
  prior_s{2,3,4}      top-K of log rollout - d^2 / (2 sigma^2), d = patch distance from the image centre
  union_half          centre disc of K//2 plus rollout top-(K - K//2) among the remaining patches
Reported per group: masked-teacher accuracy and KL to the full teacher; mean distance of kept patches from the
centre and overlap with the centre disc (fraction of K).
Students (screen seed 0, final): full@196, center@12/29/49, rollout@12/29/49.
"""
import torch

META = {
    "question": "does a Gaussian centre prior on student attention rollout give a masked teacher that is more "
                "faithful on the worst group than both the centre disc and plain rollout, at K 12, 29 and 49?",
    "method": "attention rollout with a spatial prior; keep-only faithfulness of the masked teacher per group",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); DeYoung et al. ACL 2020 (ERASER sufficiency); Sagawa et al. "
             "ICLR 2020 (worst-group)",
}

KS = (12, 29, 49)
SIDE = 14
SIGMAS = (2.0, 3.0, 4.0)
STUDENTS = (("full", 196), ("center", 12), ("center", 29), ("center", 49), ("rollout", 12), ("rollout", 29),
            ("rollout", 49))


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def _rollout(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
    A = [a.float().mean(1) for a in out.attn]
    B, N, _ = A[0].shape
    eye = torch.eye(N, device=x.device).expand(B, N, N)
    R = eye.clone()
    for a in A:
        a = 0.5 * a + 0.5 * eye
        R = (a / a.sum(-1, keepdim=True)) @ R
    return R[:, 0, 1:]


def run(api):
    runs = api.runs()
    pick = []
    for rid, r in runs.items():
        if r["tier"] != "screen" or r["seed"] != 0 or r.get("dataset", "waterbirds") != "waterbirds":
            continue
        if (r["method"], r["keep"]) in STUDENTS:
            pick.append((f"{r['method']}{r['keep']}", rid))
    pick.sort()
    api.write_json("students", {n: rid for n, rid in pick})
    groups = api.val_groups()
    T = api.teacher.eval()
    rr = torch.arange(api.num_patches)
    c = (SIDE - 1) / 2
    d2_c = ((rr // SIDE).float() - c) ** 2 + ((rr % SIDE).float() - c) ** 2
    tie = rr.float() * 1e-6

    batches = []
    ref_n = [0] * 4
    off = 0
    for x, y in api.val_batches(batch=50):
        B = x.shape[0]
        gb = groups[off:off + B]
        off += B
        for gi in range(4):
            ref_n[gi] += int((gb == gi).sum().item())
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        batches.append((x.cpu(), full.cpu(), y.cpu(), gb.cpu()))

    summary = {"group_n": ref_n, "cols": ["t_acc_mean", "t_wga", "kl_mean", "kl_worst", "cdist", "ov_center"]}
    for sname, rid in pick:
        student = api.student(rid, None)
        for p in student.parameters():
            p.requires_grad_(False)
        student.eval()
        acc = {}
        for x, full, y, gb in batches:
            dev = api.device
            x, full, y, gb = x.to(dev), full.to(dev), y.to(dev), gb.to(dev)
            B = x.shape[0]
            d2 = d2_c.to(dev)[None].expand(B, -1)
            tie_d = tie.to(dev)[None]
            R = _rollout(api, student, x)
            logR = R.clamp_min(1e-12).log()
            for k in KS:
                disc = _topk(-(d2 + tie_d), k)
                cands = {"center": disc, "top_rollout": _topk(R, k)}
                for s in SIGMAS:
                    cands[f"prior_s{int(s)}"] = _topk(logR - d2 / (2 * s * s), k)
                half = _topk(-(d2 + tie_d), k // 2)
                rest = _topk(R.masked_fill(half, -1.0), k - k // 2)
                cands["union_half"] = half | rest
                for name, keep in cands.items():
                    with torch.no_grad():
                        lp = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    cd = (d2.sqrt() * keep).sum(1) / k
                    ov = (keep & disc).sum(1).float() / k
                    _acc(acc, f"{name}@{k}", _kl(full, lp), lp.argmax(-1) == y, gb, cd, ov)
        out = _finish(acc, ref_n)
        api.write_json(f"prior_{sname}", out)
        summary[sname] = {key: _row(v) for key, v in out.items()}
        api.write_json("summary", summary)
    api.write_json("summary", summary)


def _acc(store, key, kl, ok, gb, cd, ov):
    row = store.setdefault(key, {"kl": [0.0] * 4, "t_acc": [0.0] * 4, "cd": 0.0, "ov": 0.0, "n": 0})
    kl, ok, gb = kl.float().cpu(), ok.float().cpu(), gb.cpu()
    for gi in range(4):
        m = gb == gi
        row["kl"][gi] += kl[m].sum().item()
        row["t_acc"][gi] += ok[m].sum().item()
    row["cd"] += cd.float().sum().item()
    row["ov"] += ov.float().sum().item()
    row["n"] += kl.shape[0]


def _finish(store, ref_n):
    out = {}
    for key, row in store.items():
        n_img = row["n"]
        t = [round(100 * v / max(b, 1), 2) for v, b in zip(row["t_acc"], ref_n)]
        kl = [round(v / max(b, 1), 4) for v, b in zip(row["kl"], ref_n)]
        out[key] = {"t_acc": t, "t_wga": min(t), "t_acc_mean": round(100 * sum(row["t_acc"]) / n_img, 2),
                    "kl": kl, "kl_mean": round(sum(row["kl"]) / n_img, 4), "kl_worst": max(kl),
                    "cdist": round(row["cd"] / n_img, 3), "ov_center": round(row["ov"] / n_img, 3)}
    return out


def _row(v):
    return [v["t_acc_mean"], v["t_wga"], v["kl_mean"], v["kl_worst"], v["cdist"], v["ov_center"]]
