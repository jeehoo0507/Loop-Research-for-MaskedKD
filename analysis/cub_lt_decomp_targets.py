"""H-A71 (round 360, analysis only, no training): does H-A62's run-level link (masked-teacher tail target accuracy
vs run tail accuracy, Spearman 0.52 over the 15 gate-2 runs) hold out of sample on the 12 cub_lt v0-decomposition
runs that were trained after it (lateroll, trustfilt_attn, trustfilt_pure, trustfilt@20; final, seeds 1-3)?

Records: H-A67 v0 - lateroll tail -2.44 [-4.96, +0.07]; H-A69 attn - v0 -0.11 [-1.83, +1.60]; H-A70 pure - v0
-5.11 [-10.63, +0.41] (mean_kept 12.80), v0@20 - v0@29 -4.44 [-9.44, +0.55]. This measures the same per-image
statistics as analysis/cub_lt_tail_targets.py (H-A62) on these arms plus v0@29 and maskedkd@29 as references.

Students (pinned run ids, final weights), cub_lt final seeds 1, 2, 3:
  maskedkd@29        student last-block head-mean CLS attention top-K
  trustfilt@29 (v0)  student late-block (6..11) rollout top-K inside the teacher map's top ceil(1.5 K)
  lateroll@29        student late-block rollout top-K
  trustfilt_attn@29  last-block CLS attention top-K inside the teacher map's top ceil(1.5 K)
  trustfilt_pure@29  teacher top-K AND late-block rollout top-K (variable count, >= 1; fallback = student's best
                     token inside the teacher top-K)
  trustfilt@20       v0 rule with K = 20
The teacher map is the teacher's rollout over all blocks on the un-augmented val image (= the cached map's rule).
Per image: t_acc, agree, fid, pt, s_acc as in H-A62, plus n_kept. Per image the 3 seeds are averaged; per group the
mean and a within-group image bootstrap 95% CI (2000 draws, numpy default_rng 20261108). Paired differences use the
same images. No labels, groups or positions enter any mask; groups are used only to slice the results.
"""
import math

import numpy
import torch

META = {
    "question": "out of sample: on the 12 cub_lt v0-decomposition runs (lateroll, trustfilt_attn, trustfilt_pure, "
                "trustfilt@20), does the masked teacher's tail target accuracy rank run tail accuracy as in H-A62, "
                "and does it explain pure < v0?",
    "method": "attention rollout (teacher all blocks, student blocks 6..11); last-block CLS attention; keep-only "
              "masked-teacher faithfulness sliced by head/tail class; within-group image bootstrap; Spearman",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 "
             "(keep-only); Wah et al. 2011 (CUB-200-2011)",
}

START = 6
WIDEN = 1.5
NBOOT = 2000
BATCH = 50
ARMS = {
    "maskedkd": (29, {1: "cub_lt_final_maskedkd_k29_s1_1103_234609_76d1",
                      2: "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                      3: "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"}),
    "trustfilt": (29, {1: "cub_lt_final_trustfilt_k29_s1_1103_234609_dbe3",
                       2: "cub_lt_final_trustfilt_k29_s2_1103_234647_43e9",
                       3: "cub_lt_final_trustfilt_k29_s3_1103_234704_8732"}),
    "lateroll": (29, {1: "cub_lt_final_lateroll_k29_s1_1104_084157_2b15",
                      2: "cub_lt_final_lateroll_k29_s2_1104_084204_9f09",
                      3: "cub_lt_final_lateroll_k29_s3_1104_084209_a3f8"}),
    "trustfilt_attn": (29, {1: "cub_lt_final_trustfilt_attn_k29_s1_1104_103757_81f3",
                            2: "cub_lt_final_trustfilt_attn_k29_s2_1104_103804_0d6e",
                            3: "cub_lt_final_trustfilt_attn_k29_s3_1104_103809_befb"}),
    "trustfilt_pure": (29, {1: "cub_lt_final_trustfilt_pure_k29_s1_1104_113513_79a8",
                            2: "cub_lt_final_trustfilt_pure_k29_s2_1104_113520_6b99",
                            3: "cub_lt_final_trustfilt_pure_k29_s3_1104_113525_690f"}),
    "trustfilt_k20": (20, {1: "cub_lt_final_trustfilt_k20_s1_1104_113537_fc0c",
                           2: "cub_lt_final_trustfilt_k20_s2_1104_113550_3620",
                           3: "cub_lt_final_trustfilt_k20_s3_1104_113600_d7c8"}),
}
NEW = ("lateroll", "trustfilt_attn", "trustfilt_pure", "trustfilt_k20")
GROUPS = (0, 1)
TAIL = 1
STATS = ("t_acc", "agree", "fid", "pt", "s_acc", "n_kept")


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


def _mask(api, arm, k, S, x, tm):
    depth = len(S.blocks)
    with torch.no_grad():
        so = api.student_forward(S, x, attn_layers=list(range(depth)))
    last = so.attn[depth - 1].float().mean(1)[:, 0, 1:]
    if arm == "maskedkd":
        return _topk(last, k), so.logits
    late = _rollout([so.attn[i] for i in range(START, depth)])
    if arm == "lateroll":
        return _topk(late, k), so.logits
    if arm == "trustfilt_pure":
        t_in = _topk(tm, k)
        keep = _topk(late, k) & t_in
        empty = ~keep.any(1)
        if empty.any():
            best = late.masked_fill(~t_in, -1.0).argmax(1)
            keep[empty, best[empty]] = True
        return keep, so.logits
    allowed = _topk(tm, min(int(math.ceil(WIDEN * k)), tm.shape[1]))
    s = last if arm == "trustfilt_attn" else late
    return _topk(s.masked_fill(~allowed, -1.0), k), so.logits


def run(api):
    runs = api.runs()
    used = {a: {str(s): [rid, runs[rid]["method"], runs[rid]["keep"], runs[rid]["seed"], runs[rid]["tier"],
                         runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]]
                for s, rid in sorted(ids.items())} for a, (k, ids) in ARMS.items()}
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
    for a, (k, ids) in ARMS.items():
        for seed, rid in sorted(ids.items()):
            S = api.student(rid)
            acc = {s: [] for s in STATS}
            for x, y, tm, full in batches:
                keep, slog = _mask(api, a, k, S, x, tm)
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                acc["t_acc"].append((lm.argmax(-1) == y).float().cpu())
                acc["agree"].append((lm.argmax(-1) == full.argmax(-1)).float().cpu())
                acc["fid"].append((full.exp() * (full - lm)).sum(-1).float().cpu())
                acc["pt"].append(lm.gather(1, y.view(-1, 1)).squeeze(1).exp().float().cpu())
                acc["s_acc"].append((slog.float().argmax(-1) == y).float().cpu())
                acc["n_kept"].append(keep.sum(1).float().cpu())
            for s in STATS:
                per[a][s].append(torch.cat(acc[s]).numpy())

    gen = numpy.random.default_rng(20261108)
    d = {a: {s: numpy.mean(numpy.stack(per[a][s]), 0) for s in STATS} for a in ARMS}
    out = {"n_val": int(len(groups)), "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": used}
    out["by_arm"] = {a: {s: {str(g): _ci(d[a][s][groups == g], gen) for g in GROUPS} for s in STATS} for a in ARMS}
    out["seedwise_group_mean"] = {a: {s: {str(g): [round(float(v[groups == g].mean()), 4) for v in per[a][s]]
                                          for g in GROUPS} for s in STATS} for a in ARMS}
    pairs = (("trustfilt_pure", "trustfilt"), ("trustfilt", "lateroll"), ("trustfilt_attn", "trustfilt"),
             ("trustfilt_k20", "trustfilt"), ("trustfilt_k20", "trustfilt_pure"), ("lateroll", "maskedkd"),
             ("trustfilt_attn", "maskedkd"))
    out["diff"] = {f"{p}-{q}": {s: {str(g): _ci((d[p][s] - d[q][s])[groups == g], gen) for g in GROUPS}
                                for s in STATS} for p, q in pairs}

    def _link(arms):
        tw, sw, names = [], [], []
        for a in arms:
            for i, (seed, rid) in enumerate(sorted(ARMS[a][1].items())):
                tw.append(float(per[a]["t_acc"][i][groups == TAIL].mean()))
                sw.append(float(runs[rid]["val_wga_last"]))
                names.append(rid)
        return {"runs": names, "masked_teacher_tail_t_acc": [round(v, 4) for v in tw],
                "student_val_tail_acc": sw, "spearman": round(_spearman(tw, sw), 4)}

    out["run_level_new12"] = _link(NEW)
    out["run_level_all18"] = _link(list(ARMS))
    arm_t = [out["by_arm"][a]["t_acc"][str(TAIL)][0] for a in ARMS]
    arm_w = [float(numpy.mean([runs[rid]["val_wga_last"] for rid in ids.values()])) for k, ids in ARMS.values()]
    out["arm_level"] = {"arms": list(ARMS), "masked_teacher_tail_t_acc": [round(v, 4) for v in arm_t],
                        "student_val_tail_acc_mean": [round(v, 4) for v in arm_w],
                        "spearman": round(_spearman(arm_t, arm_w), 4)}

    rho = out["run_level_new12"]["spearman"]
    dp = out["diff"]["trustfilt_pure-trustfilt"]["t_acc"][str(TAIL)]
    out["verdict"] = {
        "P1_run_level_spearman_new12": rho,
        "P1_holds_ge_0.5": bool(rho >= 0.5),
        "P2_pure_minus_v0_tail_tacc": dp,
        "P2_holds_ci_high_lt0": bool(dp[2] < 0),
    }
    api.write_json("summary", out)
