"""H-W13: does the self worst-group KL proxy (H-W10) hold on runs it was not fitted on?

analysis_proxy_validation_1030_170914_4668 picked self kl_worst (|rho| 0.937 over 16 screen runs). It was then used
(via analysis_prior_rollout_1030_174424_16a6, read on *rollout*-trained students) to predict that priorroll beats
center and rollout at K=12/29/49. Screen seed 0 says otherwise at K=29 (priorroll@29 60.15 vs rollout@29 65.41,
screen_priorroll_k29_s0_1030_175144_018d) and only ties center at K=12 (54.89 vs 56.39).

Held-out runs: priorroll@12, priorroll@29, suffdisc@12, each scored with its own final student (self). Reference
runs at the same K are re-scored identically: center@12/29 (no student needed) and rollout@12/29 (own student).
Also, to separate "the proxy is wrong" from "the proxy was read on the wrong student", priorroll's selection is
scored on the rollout@K student (what H-W12 used) next to priorroll's own student, and the selection geometry is
recorded (mean distance to centre, overlap with the centre disc, overlap with the rollout top-K of the same student).

Pre-stated reading: the proxy holds out of sample if, within each K, the ordering of self kl_worst matches the
ordering of student val WGA (K=29: rollout < priorroll < ... in KL; K=12: center <= priorroll < rollout).
"""
import math

import torch

META = {
    "question": "is the self worst-group KL(full teacher || masked teacher) proxy still rank-correct on held-out "
                "selectors (priorroll@12/29, suffdisc@12), and did H-W12 fail because it was read on the parent's "
                "student instead of the selector's own student?",
    "method": "group-sliced keep-only faithfulness (KL to the full teacher) of the masked teacher under attention "
              "rollout with / without a centre prior and under student-sufficiency disc placement",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); DeYoung et al. ACL 2020 (ERASER sufficiency); Hooker et al. "
             "NeurIPS 2019 (keep-only evaluation); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

SIDE = 14
KS = (12, 29)


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _geom(n, dev):
    rr = torch.arange(n, device=dev)
    c = (SIDE - 1) / 2
    return rr, (rr // SIDE).float(), (rr % SIDE).float(), c


def _center(B, n, k, dev):
    rr, rows, cols, c = _geom(n, dev)
    d = (rows - c) ** 2 + (cols - c) ** 2 + rr.float() * 1e-6
    return _topk((-d).expand(B, n), k)


def _rollout(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
    R = None
    for a in out.attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        R = a if R is None else a @ R
    return R[:, 0, 1:]


def _priorroll(R, k):
    n = R.shape[1]
    rr, rows, cols, c = _geom(n, R.device)
    d2 = (rows - c) ** 2 + (cols - c) ** 2
    s = math.sqrt(k / math.pi)
    return _topk(R.clamp_min(1e-12).log() - d2 / (2 * s * s), k)


def _masked_logits(model, x, idx):
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    cls, patches = h[:, :1], h[:, 1:]
    D = patches.shape[-1]
    patches = torch.gather(patches, 1, idx.unsqueeze(-1).expand(-1, -1, D))
    h = torch.cat([cls, patches], dim=1)
    h = model.norm_pre(h)
    h = model.blocks(h)
    h = model.norm(h)
    return model.forward_head(h)


def _suffdisc(student, x, k, r=2):
    B, dev, n = x.shape[0], x.device, SIDE * SIDE
    rr, rows, cols, c = _geom(n, dev)
    tie = rr.float() * 1e-6
    with torch.no_grad():
        full = student(x).float().log_softmax(-1)
        best_kl = torch.full((B,), float("inf"), device=dev)
        best_idx = None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d = (rows - (c + dy)) ** 2 + (cols - (c + dx)) ** 2 + tie
                idx = (-d).topk(k).indices.sort().values.expand(B, -1)
                lp = _masked_logits(student, x, idx).float().log_softmax(-1)
                kl = (full.exp() * (full - lp)).sum(-1)
                better = kl < best_kl
                best_kl = torch.where(better, kl, best_kl)
                best_idx = idx.clone() if best_idx is None else torch.where(better[:, None], idx, best_idx)
    return torch.zeros(B, n, dtype=torch.bool, device=dev).scatter_(1, best_idx, True)


def run(api):
    runs = api.runs()
    pick = {}
    for rid, r in sorted(runs.items()):
        if r["tier"] != "screen" or r["seed"] != 0 or r.get("dataset", "waterbirds") != "waterbirds":
            continue
        name = f"{r['method'].split('@')[0]}@{int(r['keep'])}"
        if name in ("priorroll@12", "priorroll@29", "suffdisc@12", "rollout@12", "rollout@29",
                    "center@12", "center@29"):
            pick[name] = (rid, r["val_wga_last"], r["val_acc_last"])
    api.write_json("targets", pick)

    groups = api.val_groups()
    T = api.teacher.eval()
    n = api.num_patches
    batches = []
    off = 0
    for x, y in api.val_batches(batch=50):
        B = x.shape[0]
        gb = groups[off:off + B]
        off += B
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        batches.append((x.cpu(), y.cpu(), gb.cpu(), full.cpu()))
    dev = api.device
    gn = [sum(int((b[2] == gi).sum()) for b in batches) for gi in range(4)]

    def load(rid):
        st = api.student(rid)
        for p in st.parameters():
            p.requires_grad_(False)
        return st.eval()

    def score(select_fn, k, ref_fn=None):
        kl = [0.0] * 4
        tacc = [0.0] * 4
        geo = {"cdist": 0.0, "ov_center": 0.0, "ov_ref": 0.0}
        tot = 0
        for x, y, gb, full in batches:
            x, y, gb, full = x.to(dev), y.to(dev), gb.to(dev), full.to(dev)
            B = x.shape[0]
            keep = select_fn(x, k)
            with torch.no_grad():
                lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
            v_kl = (full.exp() * (full - lm)).sum(-1)
            v_acc = (lm.argmax(-1) == y).float()
            for gi in range(4):
                m = gb == gi
                kl[gi] += v_kl[m].sum().item()
                tacc[gi] += v_acc[m].sum().item()
            rr, rows, cols, c = _geom(n, dev)
            dist = ((rows - c) ** 2 + (cols - c) ** 2).sqrt()
            geo["cdist"] += ((keep.float() * dist).sum(1) / k).sum().item()
            geo["ov_center"] += ((keep & _center(B, n, k, dev)).float().sum(1) / k).sum().item()
            if ref_fn is not None:
                geo["ov_ref"] += ((keep & ref_fn(x, k)).float().sum(1) / k).sum().item()
            tot += B
        kl = [round(v / max(c_, 1), 4) for v, c_ in zip(kl, gn)]
        tacc = [round(100 * v / max(c_, 1), 2) for v, c_ in zip(tacc, gn)]
        return {"kl": kl, "kl_worst": max(kl), "t_acc": tacc, "t_wga": min(tacc),
                **{g: round(v / tot, 4) for g, v in geo.items()}}

    rows = {}
    for k in KS:
        if f"center@{k}" in pick:
            rid, wga, acc = pick[f"center@{k}"]
            rows[f"center@{k}"] = {"run": rid, "student_wga": wga, "student_acc": acc,
                                   "self": score(lambda x, kk: _center(x.shape[0], n, kk, dev), k)}
        roll_st = load(pick[f"rollout@{k}"][0]) if f"rollout@{k}" in pick else None
        if roll_st is not None:
            rid, wga, acc = pick[f"rollout@{k}"]
            rows[f"rollout@{k}"] = {"run": rid, "student_wga": wga, "student_acc": acc,
                                    "self": score(lambda x, kk: _topk(_rollout(api, roll_st, x), kk), k)}
        if f"priorroll@{k}" in pick:
            rid, wga, acc = pick[f"priorroll@{k}"]
            st = load(rid)
            entry = {"run": rid, "student_wga": wga, "student_acc": acc,
                     "self": score(lambda x, kk: _priorroll(_rollout(api, st, x), kk), k,
                                   ref_fn=lambda x, kk: _topk(_rollout(api, st, x), kk))}
            # own student's plain rollout top-K: does training with the prior change the student's rollout?
            entry["self_plain_rollout"] = score(lambda x, kk: _topk(_rollout(api, st, x), kk), k)
            if roll_st is not None:
                entry["on_rollout_student"] = score(lambda x, kk: _priorroll(_rollout(api, roll_st, x), kk), k,
                                                    ref_fn=lambda x, kk: _topk(_rollout(api, roll_st, x), kk))
            rows[f"priorroll@{k}"] = entry
            del st
        if k == 12 and "suffdisc@12" in pick:
            rid, wga, acc = pick["suffdisc@12"]
            st = load(rid)
            rows["suffdisc@12"] = {"run": rid, "student_wga": wga, "student_acc": acc,
                                   "self": score(lambda x, kk: _suffdisc(st, x, kk), k)}
            del st
        del roll_st
        api.write_json("rows", rows)

    summary = {}
    for k in KS:
        names = [nm for nm in rows if nm.endswith(f"@{k}")]
        by_kl = sorted(names, key=lambda nm: rows[nm]["self"]["kl_worst"])
        by_wga = sorted(names, key=lambda nm: -rows[nm]["student_wga"])
        summary[f"K{k}"] = {"order_by_self_kl_worst": by_kl, "order_by_student_wga": by_wga,
                            "same_order": by_kl == by_wga,
                            "points": {nm: [rows[nm]["student_wga"], rows[nm]["self"]["kl_worst"]] for nm in names}}
    api.write_json("summary", {"group_n": gn, "per_K": summary})
