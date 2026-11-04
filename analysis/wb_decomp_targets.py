"""H-A61 (round 336, analysis only, no training): on Waterbirds seeds 4-6, do the parts of v0 (trustfilt) change the
per-group quality of the masked teacher's KD target the way they change WGA?

H-A60 (seeds 4-6, K=29): trustfilt_attn - v0 WGA -3.51 [-5.67, -1.35] ("lateroll is needed"), trustfilt_pure - v0
-4.76 [-7.61, -1.91]; H-A59 v0 - teacher_topk +6.01 [-0.46, +12.48]. H-A41 (seeds 1-3) found v0's target more
correct than teacher_topk's on both minority groups, but a method-level Spearman of only 0.30 between target
accuracy and WGA. This measures the same target statistics on the 15 seed 4-6 runs.

Students (pinned run ids, final weights): Waterbirds screen seeds 4, 5, 6 of
  trustfilt@29 (v0)    student late-block (6..11) rollout top-K inside the teacher map's top ceil(1.5 K)
  teacher_topk@29      teacher map top-K
  trustfilt_attn@29    last-block head-mean CLS attention top-K inside the teacher map's top ceil(1.5 K)
  trustfilt_pure@29    late rollout top-K AND teacher map top-K (empty -> best late token inside teacher top-K)
  trustfilt@20         as v0 with K = 20
The teacher map is the teacher's rollout over all blocks on the un-augmented val image (= the cached map's rule).
Per image:
  t_acc   masked teacher (keep-only) top-1 == label
  agree   masked teacher top-1 == full teacher top-1
  fid     KL(teacher full || teacher masked)
  ntok    kept tokens
  salign  fraction of the kept set inside the student's own late-block rollout top-K
  s_acc   student top-1 == label
Per image the 3 seeds are averaged; per group the mean and a within-group image bootstrap 95% CI (2000 draws,
numpy default_rng 20261106). Paired differences use the same images.
No labels, groups or positions enter any mask; groups are used only to slice the results.
"""
import math

import numpy
import torch

META = {
    "question": "on Waterbirds seeds 4-6, is v0's masked-teacher KD target more correct on the minority groups than "
                "teacher_topk's and than the one picked by MaskedKD's rule inside the same teacher region "
                "(trustfilt_attn), and does run-level worst-group target accuracy track run WGA across the 15 runs?",
    "method": "attention rollout (teacher all blocks, student blocks 6..11); last-block CLS attention; keep-only "
              "masked-teacher faithfulness sliced by Waterbirds group; within-group image bootstrap; Spearman",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 "
             "(keep-only); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

START = 6
WIDEN = 1.5
NBOOT = 2000
BATCH = 50
ARMS = {
    "v0_29": ("trustfilt", 29, {4: "waterbirds_screen_trustfilt_k29_s4_1104_041245_3797",
                                5: "waterbirds_screen_trustfilt_k29_s5_1104_041250_8224",
                                6: "waterbirds_screen_trustfilt_k29_s6_1104_041254_004f"}),
    "topk_29": ("teacher_topk", 29, {4: "waterbirds_screen_teacher_topk_k29_s4_1104_041258_c453",
                                     5: "waterbirds_screen_teacher_topk_k29_s5_1104_041303_2db2",
                                     6: "waterbirds_screen_teacher_topk_k29_s6_1104_041307_d108"}),
    "attn_29": ("trustfilt_attn", 29, {4: "waterbirds_screen_trustfilt_attn_k29_s4_1104_050310_43e0",
                                       5: "waterbirds_screen_trustfilt_attn_k29_s5_1104_050317_860f",
                                       6: "waterbirds_screen_trustfilt_attn_k29_s6_1104_050321_808a"}),
    "pure_29": ("trustfilt_pure", 29, {4: "waterbirds_screen_trustfilt_pure_k29_s4_1104_050326_25d1",
                                       5: "waterbirds_screen_trustfilt_pure_k29_s5_1104_050330_15c1",
                                       6: "waterbirds_screen_trustfilt_pure_k29_s6_1104_050334_db07"}),
    "v0_20": ("trustfilt", 20, {4: "waterbirds_screen_trustfilt_k20_s4_1104_050339_b2dd",
                                5: "waterbirds_screen_trustfilt_k20_s5_1104_050343_9a6d",
                                6: "waterbirds_screen_trustfilt_k20_s6_1104_050347_3cfa"}),
}
GROUPS = (0, 1, 2, 3)
MINOR = (1, 2)
STATS = ("t_acc", "agree", "fid", "ntok", "salign", "s_acc")


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


def _mask(api, method, k, S, x, tm):
    depth = len(S.blocks)
    with torch.no_grad():
        so = api.student_forward(S, x, attn_layers=list(range(depth)))
    late = _rollout([so.attn[i] for i in range(START, depth)])
    n = tm.shape[1]
    if method == "teacher_topk":
        keep = _topk(tm, k)
    elif method == "trustfilt_pure":
        t_in = _topk(tm, k)
        keep = _topk(late, k) & t_in
        empty = ~keep.any(1)
        if empty.any():
            best = late.masked_fill(~t_in, -1.0).argmax(1)
            keep[empty, best[empty]] = True
    else:
        allowed = _topk(tm, min(int(math.ceil(WIDEN * k)), n))
        s = so.attn[depth - 1].float().mean(1)[:, 0, 1:] if method == "trustfilt_attn" else late
        keep = _topk(s.masked_fill(~allowed, -1.0), k)
    return keep, _topk(late, k), so.logits


def run(api):
    runs = api.runs()
    used = {a: {str(s): [rid, runs[rid]["method"], runs[rid]["keep"], runs[rid]["seed"], runs[rid]["tier"],
                         runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]]
                for s, rid in sorted(ids.items())} for a, (_, _, ids) in ARMS.items()}
    api.write_json("students", used)
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

    per = {a: {s: [] for s in STATS} for a in ARMS}
    for a, (method, k, ids) in ARMS.items():
        for seed, rid in sorted(ids.items()):
            S = api.student(rid)
            acc = {s: [] for s in STATS}
            for x, y, tm, full in batches:
                keep, own, slog = _mask(api, method, k, S, x, tm)
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                nk = keep.sum(1).float()
                acc["t_acc"].append((lm.argmax(-1) == y).float().cpu())
                acc["agree"].append((lm.argmax(-1) == full.argmax(-1)).float().cpu())
                acc["fid"].append((full.exp() * (full - lm)).sum(-1).float().cpu())
                acc["ntok"].append(nk.cpu())
                acc["salign"].append(((keep & own).sum(1).float() / nk).cpu())
                acc["s_acc"].append((slog.float().argmax(-1) == y).float().cpu())
            for s in STATS:
                per[a][s].append(torch.cat(acc[s]).numpy())

    gen = numpy.random.default_rng(20261106)
    d = {a: {s: numpy.mean(numpy.stack(per[a][s]), 0) for s in STATS} for a in ARMS}
    out = {"n_val": int(len(groups)), "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": used}
    out["by_arm"] = {a: {s: {str(g): _ci(d[a][s][groups == g], gen) for g in GROUPS} for s in STATS} for a in ARMS}
    out["seedwise_group_mean"] = {a: {s: {str(g): [round(float(v[groups == g].mean()), 4) for v in per[a][s]]
                                          for g in GROUPS} for s in STATS} for a in ARMS}
    pairs = (("v0_29", "topk_29"), ("v0_29", "attn_29"), ("pure_29", "v0_29"), ("v0_20", "v0_29"),
             ("v0_20", "pure_29"), ("attn_29", "topk_29"))
    out["diff"] = {f"{p}-{q}": {s: {str(g): _ci((d[p][s] - d[q][s])[groups == g], gen) for g in GROUPS}
                                for s in STATS} for p, q in pairs}

    tw, sw, names = [], [], []
    for a, (_, _, ids) in ARMS.items():
        for i, (seed, rid) in enumerate(sorted(ids.items())):
            tw.append(min(float(per[a]["t_acc"][i][groups == g].mean()) for g in GROUPS))
            sw.append(float(runs[rid]["val_wga_last"]))
            names.append(rid)
    rho = _spearman(tw, sw)
    out["run_level"] = {"runs": names, "masked_teacher_min_group_t_acc": [round(v, 4) for v in tw],
                        "student_val_wga": sw, "spearman": round(rho, 4)}
    arm_t = [min(out["by_arm"][a]["t_acc"][str(g)][0] for g in GROUPS) for a in ARMS]
    arm_w = [float(numpy.mean([runs[rid]["val_wga_last"] for rid in ids.values()])) for _, _, ids in ARMS.values()]
    out["arm_level"] = {"arms": list(ARMS), "masked_teacher_min_group_t_acc": [round(v, 4) for v in arm_t],
                        "student_val_wga_mean": [round(v, 4) for v in arm_w],
                        "spearman": round(_spearman(arm_t, arm_w), 4)}

    dt, da = out["diff"]["v0_29-topk_29"]["t_acc"], out["diff"]["v0_29-attn_29"]["t_acc"]
    out["verdict"] = {
        "P1_v0_minus_topk_tacc_minor": {str(g): dt[str(g)] for g in MINOR},
        "P1_holds_ci_low_gt0_both_minor": bool(all(dt[str(g)][1] > 0 for g in MINOR)),
        "P2_v0_minus_attn_tacc_group2": da["2"],
        "P2_holds_ci_low_gt0": bool(da["2"][1] > 0),
        "P3_run_level_spearman": round(rho, 4),
        "P3_holds_ge_0.5": bool(rho >= 0.5),
    }
    api.write_json("summary", out)
