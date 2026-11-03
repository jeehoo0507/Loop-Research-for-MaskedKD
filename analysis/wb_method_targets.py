"""H-A41 (round 288, analysis only, no training): on Waterbirds K=29, does the quality of each method's KD target per
group explain the gate-1 WGA order, including the unexpected v0 (trustfilt) result?

Gate 1 records (val WGA, seeds 1-3, K=29): trustfilt (v0) 72.68 > teacher_topk 64.41 > trustmask_v1 62.40 ~
rollout 62.16 > maskedkd 38.59. H-A40 showed maskedkd's masked teacher is less correct than v1's on the minority
groups. This looks at all five methods with one measurement.

Students: Waterbirds final maskedkd, rollout, teacher_topk, trustmask_v1, trustfilt @29, seeds 1, 2, 3, at epoch 25
and final. Mask: each run's own token rule on the un-augmented val image, K = 29:
  maskedkd      last-block head-mean CLS attention top-K
  rollout       student rollout over all blocks top-K
  teacher_topk  teacher rollout over all blocks (= the cached map on an un-augmented image) top-K
  trustmask_v1  student rollout over blocks 6..11 top-K (its KD weight is not a mask and is not measured here)
  trustfilt     student rollout over blocks 6..11 top-K inside the teacher map's top ceil(1.5 K)
Per image:
  t_acc   masked teacher (keep-only) top-1 == label
  agree   masked teacher top-1 == full teacher top-1
  fid     KL(teacher full || teacher masked)
  cov     teacher rollout mass inside the kept set
  salign  fraction of the kept set inside the student's own late-block (6..11) rollout top-K
  s_acc   student top-1 == label
Per image the 3 seeds are averaged; per group the mean and a within-group image bootstrap 95% CI (2000 draws,
numpy default_rng 20261105). Paired differences use the same images.
No labels, groups or positions enter any mask; groups are used only to slice the results.
"""
import math

import numpy
import torch

META = {
    "question": "on Waterbirds K=29, does the per-group quality of each method's masked-teacher KD target (accuracy, "
                "KL, coverage) rank maskedkd, rollout, teacher_topk, trustmask_v1 and trustfilt in the same order "
                "as their student WGA, and is trustfilt's WGA lead over teacher_topk a more correct target or a "
                "target closer to what the student attends to?",
    "method": "attention rollout (teacher all blocks, student all / 6..11 blocks); last-block CLS attention; keep-only "
              "masked-teacher faithfulness sliced by Waterbirds group; within-group image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 "
             "(keep-only); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

K = 29
START = 6
WIDEN = 1.5
NBOOT = 2000
BATCH = 50
EPOCHS = (25, "final")
METHODS = ("maskedkd", "rollout", "teacher_topk", "trustmask_v1", "trustfilt")
WGA = {"trustfilt": 72.68, "teacher_topk": 64.41, "trustmask_v1": 62.40, "rollout": 62.16, "maskedkd": 38.59}
GROUPS = (0, 1, 2, 3)
MINOR = (1, 2)
STATS = ("t_acc", "agree", "fid", "cov", "salign", "s_acc")


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


def _ci(v, gen):
    n = len(v)
    bs = numpy.array([v[gen.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return [round(float(v.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4), int(n)]


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2.0
        i = j + 1
    return numpy.array(r)


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum()))


def _mask(api, m, S, x, tm):
    depth = len(S.blocks)
    with torch.no_grad():
        so = api.student_forward(S, x, attn_layers=list(range(depth)))
    late = _rollout([so.attn[i] for i in range(START, depth)])
    if m == "maskedkd":
        keep = _topk(so.attn[depth - 1].float().mean(1)[:, 0, 1:], K)
    elif m == "rollout":
        keep = _topk(_rollout([so.attn[i] for i in range(depth)]), K)
    elif m == "teacher_topk":
        keep = _topk(tm, K)
    elif m == "trustmask_v1":
        keep = _topk(late, K)
    else:
        allowed = _topk(tm, min(int(math.ceil(WIDEN * K)), tm.shape[1]))
        keep = _topk(late.masked_fill(~allowed, -1.0), K)
    return keep, _topk(late, K), so.logits


def run(api):
    runs = api.runs()
    studs = {m: {} for m in METHODS}
    seen = {m: {} for m in METHODS}
    for rid, r in sorted(runs.items()):
        if (r.get("dataset") == "waterbirds" and r["tier"] == "final" and r["method"] in METHODS
                and r.get("keep") == K and r["seed"] in (1, 2, 3)):
            studs[r["method"]][r["seed"]] = rid
            seen[r["method"]].setdefault(str(r["seed"]), []).append(rid)
    api.write_json("students", {"used": {m: {str(s): [rid, runs[rid]["val_wga_last"], runs[rid]["val_acc_last"],
                                                      runs[rid]["snapshots"]] for s, rid in sorted(d.items())}
                                         for m, d in studs.items()},
                                "all_matching": seen})
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        tm = _rollout(out.attn)
        full = out.logits.float().log_softmax(-1)
        batches.append((x, y, tm, full))
    groups = api.val_groups().numpy()

    per = {m: {str(e): {s: [] for s in STATS} for e in EPOCHS} for m in METHODS}
    for m in METHODS:
        for seed, rid in sorted(studs[m].items()):
            for ep in EPOCHS:
                S = api.student(rid, epoch=None if ep == "final" else ep)
                acc = {s: [] for s in STATS}
                for x, y, tm, full in batches:
                    keep, own, slog = _mask(api, m, S, x, tm)
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    acc["t_acc"].append((lm.argmax(-1) == y).float().cpu())
                    acc["agree"].append((lm.argmax(-1) == full.argmax(-1)).float().cpu())
                    acc["fid"].append((full.exp() * (full - lm)).sum(-1).float().cpu())
                    acc["cov"].append((tm * keep).sum(1).float().cpu())
                    acc["salign"].append(((keep & own).sum(1).float() / K).cpu())
                    acc["s_acc"].append((slog.float().argmax(-1) == y).float().cpu())
                for s in STATS:
                    per[m][str(ep)][s].append(torch.cat(acc[s]).numpy())

    gen = numpy.random.default_rng(20261105)
    d = {m: {str(e): {s: numpy.mean(numpy.stack(per[m][str(e)][s]), 0) for s in STATS} for e in EPOCHS}
         for m in METHODS if studs[m]}
    have = [m for m in METHODS if m in d]
    out = {"k": K, "n_val": int(len(groups)), "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": {m: {str(s): r for s, r in v.items()} for m, v in studs.items()},
           "seeds_per_method": {m: len(studs[m]) for m in METHODS}}
    for ep in EPOCHS:
        e = str(ep)
        res = {"by_method": {m: {s: {str(g): _ci(d[m][e][s][groups == g], gen) for g in GROUPS} for s in STATS}
                             for m in have},
               "seedwise_group_mean": {m: {s: {str(g): [round(float(v[groups == g].mean()), 4)
                                                        for v in per[m][e][s]] for g in GROUPS} for s in STATS}
                                       for m in have}}
        res["masked_teacher_wga"] = {m: round(min(res["by_method"][m]["t_acc"][str(g)][0] for g in GROUPS), 4)
                                     for m in have}
        if "trustfilt" in d:
            res["trustfilt_minus"] = {o: {s: {str(g): _ci((d["trustfilt"][e][s] - d[o][e][s])[groups == g], gen)
                                              for g in GROUPS} for s in STATS}
                                      for o in have if o != "trustfilt"}
        out[e] = res
        api.write_json(f"epoch_{e}", res)

    fin = out["final"]
    rho = _spearman([fin["masked_teacher_wga"][m] for m in have], [WGA[m] for m in have])
    rho_fid = _spearman([-max(fin["by_method"][m]["fid"][str(g)][0] for g in GROUPS) for m in have],
                        [WGA[m] for m in have])
    tf = fin.get("trustfilt_minus", {}).get("teacher_topk", {})
    out["verdict"] = {
        "methods": have,
        "P1_spearman_masked_teacher_wga_vs_student_wga": round(rho, 4),
        "P1_holds_rho_ge_0.9": bool(len(have) == len(METHODS) and rho >= 0.9),
        "P2_trustfilt_minus_topk_tacc_minor": (
            {str(g): tf["t_acc"][str(g)] for g in MINOR} if tf else None),
        "P2_holds_not_more_correct_both_minor": (
            bool(all(tf["t_acc"][str(g)][1] <= 0 for g in MINOR)) if tf else None),
        "R_trustfilt_minus_topk_salign_minor": ({str(g): tf["salign"][str(g)] for g in MINOR} if tf else None),
        "R_spearman_neg_worst_group_kl_vs_student_wga": round(rho_fid, 4),
    }
    api.write_json("summary", out)
