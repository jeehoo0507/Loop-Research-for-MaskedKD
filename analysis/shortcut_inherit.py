"""H-W15: does a selector help student WGA by making the masked teacher *less* faithful exactly where the full
teacher follows the background shortcut?

analysis_proxy_holdout_1030_182326_7254 / analysis_prior_failure_1030_182537_611c: at K=29 the scattered rollout
selection is less faithful to the full teacher (self kl_worst 0.443) than priorroll (0.391), yet its student has
the higher val WGA (65.41 vs 60.15, screen seed 0). The full teacher is itself shortcut-prone (val WGA 64.66,
worst group 2). Faithfulness to the full teacher may therefore also mean inheriting its shortcut.

Every finished Waterbirds screen run of maskedkd / rollout / center / grid / random / midattn / priorroll /
suffdisc at K = 12, 29, 49 (any seed) is scored with its own final student on clean val images.
Images are split by whether the FULL teacher is right (FR) or wrong (FW) on the label.
  fix_fw      masked-teacher accuracy on FW images (it corrects the full teacher's mistake)
  inherit_fw  masked teacher repeats the full teacher's wrong class on FW images (binary task: 100 - fix_fw)
  ptrue_fw    mean true-class probability on FW images (what KD pulls the student towards there)
  ptrue_fr_minor  mean true-class probability on FR images of minority groups 1, 2 (cost of unfaithfulness)
  kl_worst    worst-group KL(full || masked), the faithfulness proxy that failed out of sample (reference)
  div_ent     normalised entropy of the per-patch selection frequency over val (1 = every patch equally often)
  div_jac     mean Jaccard between an image's selection and the next image's (low = image-specific choices)
Each is Spearman-correlated with the runs' val WGA over seed-0 runs (all K, and within K).

Pre-stated reading: an axis is a candidate teachability measure only if (a) |Spearman| >= 0.8 over all seed-0
runs and (b) it orders K=29 seed 0 as rollout > center > priorroll (the case kl_worst got wrong). Seed-1 K=29
runs, if finished, are reported as replication points and not used in (a).
"""
import math

import torch

META = {
    "question": "does a teacher-input selector raise student worst-group accuracy by correcting the full "
                "teacher's shortcut errors (unfaithfulness where the full teacher is wrong) or by the spatial "
                "diversity of its choices, rather than by faithfulness to the full teacher?",
    "method": "keep-only evaluation of the masked teacher split by full-teacher correctness, per Waterbirds group, "
              "under CLS attention, attention rollout and spatial priors; selection-frequency entropy",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Hooker et al. NeurIPS 2019 (keep-only evaluation); "
             "Sagawa et al. ICLR 2020 (Waterbirds groups); Kirichenko et al. ICLR 2023 (shortcut features in "
             "the last layer)",
}

KS = (12, 29, 49)
SIDE = 14
METHODS = ("maskedkd", "rollout", "center", "grid", "random", "midattn", "priorroll", "suffdisc")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _geom(n, dev):
    rr = torch.arange(n, device=dev)
    c = (SIDE - 1) / 2
    return rr, (rr // SIDE).float(), (rr % SIDE).float(), c


def _attn(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
    A = [a.float().mean(1) for a in out.attn]
    R = None
    for a in A:
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        R = a if R is None else a @ R
    return A, R[:, 0, 1:]


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


def _select(api, method, student, x, k):
    B, dev, n = x.shape[0], x.device, api.num_patches
    rr, rows, cols, c = _geom(n, dev)
    if method == "center":
        d = (rows - c) ** 2 + (cols - c) ** 2 + rr.float() * 1e-6
        return _topk((-d).expand(B, n), k)
    if method == "grid":
        idx = torch.linspace(0, n - 1, k, device=dev).round().long()
        return torch.zeros(B, n, dtype=torch.bool, device=dev).index_fill_(1, idx, True)
    if method == "random":
        return _topk(torch.rand(B, n, device=dev), k)
    if method == "suffdisc":
        return _suffdisc(student, x, k)
    A, R = _attn(api, student, x)
    if method == "maskedkd":
        return _topk(A[-1][:, 0, 1:], k)
    if method == "midattn":
        return _topk(A[9][:, 0, 1:], k)
    if method == "rollout":
        return _topk(R, k)
    s = math.sqrt(k / math.pi)          # priorroll
    d2 = (rows - c) ** 2 + (cols - c) ** 2
    return _topk(R.clamp_min(1e-12).log() - d2 / (2 * s * s), k)


def _rank(v):
    v = torch.tensor(v, dtype=torch.float64)
    order = v.argsort()
    r = torch.empty_like(v)
    r[order] = torch.arange(len(v), dtype=torch.float64)
    for val in v.unique():
        m = v == val
        r[m] = r[m].mean()
    return r


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = (ra.norm() * rb.norm()).item()
    return None if d == 0 else round((ra @ rb).item() / d, 3)


def run(api):
    runs = api.runs()
    targets = []
    for rid, r in sorted(runs.items()):
        if r["tier"] != "screen" or r.get("dataset", "waterbirds") != "waterbirds":
            continue
        if r["method"] in METHODS and r["keep"] is not None and int(r["keep"]) in KS:
            targets.append((r["method"], int(r["keep"]), int(r["seed"]), rid, r["val_wga_last"]))
    api.write_json("targets", targets)

    groups = api.val_groups()
    T = api.teacher.eval()
    n = api.num_patches
    dev = api.device
    batches = []
    off = 0
    for x, y in api.val_batches(batch=50):
        B = x.shape[0]
        gb = groups[off:off + B]
        off += B
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        batches.append((x.cpu(), y.cpu(), gb.cpu(), full.cpu()))
    G = torch.cat([b[2] for b in batches])
    FR = torch.cat([(b[3].argmax(-1) == b[1]) for b in batches])
    gn = [int((G == gi).sum()) for gi in range(4)]
    fw_n = [int(((G == gi) & ~FR).sum()) for gi in range(4)]

    rows = {}
    for method, k, seed, rid, wga in targets:
        st = api.student(rid)
        for p in st.parameters():
            p.requires_grad_(False)
        ok, pt, ag, kl = [], [], [], []
        freq = torch.zeros(n, device=dev)
        jac = []
        prev = None
        for x, y, gb, full in batches:
            x, y, full = x.to(dev), y.to(dev), full.to(dev)
            keep = _select(api, method, st, x, k)
            with torch.no_grad():
                lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
            ok.append((lm.argmax(-1) == y).cpu())
            pt.append(lm.exp().gather(1, y[:, None])[:, 0].cpu())
            ag.append((lm.argmax(-1) == full.argmax(-1)).cpu())
            kl.append((full.exp() * (full - lm)).sum(-1).cpu())
            freq += keep.float().sum(0)
            seq = keep if prev is None else torch.cat([prev[-1:], keep])
            inter = (seq[1:] & seq[:-1]).sum(1).float()
            union = (seq[1:] | seq[:-1]).sum(1).float()
            jac.append((inter / union.clamp(min=1)).cpu())
            prev = keep
        OK, PT, AG, KL = torch.cat(ok).float(), torch.cat(pt), torch.cat(ag).float(), torch.cat(kl)
        fw, minor_fr = ~FR, FR & ((G == 1) | (G == 2))
        p = freq / freq.sum()
        ent = -(p[p > 0] * p[p > 0].log()).sum().item() / math.log(n)
        kl_g = [KL[G == gi].mean().item() for gi in range(4)]
        rows[f"{method}@{k}/s{seed}"] = {
            "run": rid, "method": method, "K": k, "seed": seed, "student_wga": wga,
            "fix_fw": round(100 * OK[fw].mean().item(), 2),
            "fix_fw_g": [round(100 * OK[fw & (G == gi)].mean().item(), 2) if fw_n[gi] else None for gi in range(4)],
            "inherit_fw": round(100 * AG[fw].mean().item(), 2),
            "ptrue_fw": round(PT[fw].mean().item(), 4),
            "ptrue_fr_minor": round(PT[minor_fr].mean().item(), 4),
            "t_acc_g": [round(100 * OK[G == gi].mean().item(), 2) for gi in range(4)],
            "kl_worst": round(max(kl_g), 4),
            "div_ent": round(ent, 4),
            "div_jac": round(torch.cat(jac).mean().item(), 4),
        }
        api.write_json("rows", rows)
        del st

    stats = {"fix_fw": 1, "inherit_fw": -1, "ptrue_fw": 1, "ptrue_fr_minor": 1, "kl_worst": -1,
             "div_ent": 1, "div_jac": -1}
    corr = {}
    s0 = {nm: r for nm, r in rows.items() if r["seed"] == 0 and r["student_wga"] is not None}
    for s, sign in stats.items():
        c = {"all": _spearman([sign * r[s] for r in s0.values()], [r["student_wga"] for r in s0.values()]),
             "n_all": len(s0)}
        for k in KS:
            rk = [r for r in s0.values() if r["K"] == k]
            c[f"K{k}"] = _spearman([sign * r[s] for r in rk], [r["student_wga"] for r in rk])
        trio = [s0.get(f"{m}@29/s0") for m in ("rollout", "center", "priorroll")]
        c["k29_order_ok"] = None if None in trio else bool(sign * trio[0][s] > sign * trio[1][s] > sign * trio[2][s])
        c["criterion_pass"] = bool(c["all"] is not None and c["all"] >= 0.8 and c["k29_order_ok"])
        corr[s] = c
    api.write_json("summary", {"group_n": gn, "full_wrong_n": fw_n,
                               "full_teacher_val_acc_g": [round(100 * FR[G == gi].float().mean().item(), 2)
                                                          for gi in range(4)],
                               "note": "Spearman of sign-adjusted statistic vs student val WGA over seed-0 runs; "
                                       "criterion: all >= 0.8 and K=29 order rollout > center > priorroll",
                               "corr": corr,
                               "table": {nm: [r["student_wga"], r["fix_fw"], r["ptrue_fw"], r["ptrue_fr_minor"],
                                              r["kl_worst"], r["div_ent"], r["div_jac"]] for nm, r in rows.items()},
                               "table_cols": ["student_wga", "fix_fw", "ptrue_fw", "ptrue_fr_minor", "kl_worst",
                                              "div_ent", "div_jac"]})
