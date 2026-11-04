"""H-A62 (round 337, analysis only, no training): on cub_lt (gate 2), is the masked teacher's KD target on the
tail classes less correct under MaskedKD's rule than under the other K=29 rules, and does run-level tail target
accuracy track run tail accuracy?

Gate 2 (H-A39, final K=29, seeds 1-3, tail accuracy): v1 - maskedkd +6.33 [+1.72, +10.94], v1 - teacher_topk +4.22
[-0.63, +9.08], v0 - maskedkd +3.33 [-3.44, +10.11]; Full - CE tail +8.11. On Waterbirds (H-A41, H-A61) the masked
teacher's minority-group target accuracy separated the rules. This measures the same target statistics on the 15
cub_lt K=29 runs, with group 0 = head classes, group 1 = tail classes.

Students (pinned run ids, final weights), cub_lt final seeds 1, 2, 3, K = 29:
  maskedkd       student last-block head-mean CLS attention top-K
  rollout        student all-block rollout top-K
  teacher_topk   teacher map top-K
  trustfilt (v0) student late-block (6..11) rollout top-K inside the teacher map's top ceil(1.5 K)
  trustmask_v1   student late-block rollout top-K (the teacher map only weights the KD term, not the kept set)
The teacher map is the teacher's rollout over all blocks on the un-augmented val image (= the cached map's rule).
Per image:
  t_acc   masked teacher (keep-only) top-1 == label
  agree   masked teacher top-1 == full teacher top-1
  fid     KL(teacher full || teacher masked)
  pt      masked teacher p(label)
  s_acc   student top-1 == label
Per image the 3 seeds are averaged; per group the mean and a within-group image bootstrap 95% CI (2000 draws,
numpy default_rng 20261107). Paired differences use the same images.
No labels, groups or positions enter any mask; groups are used only to slice the results.
"""
import math

import numpy
import torch

META = {
    "question": "on cub_lt K=29, is MaskedKD's masked-teacher KD target less correct on the tail classes than v1's "
                "(and the other rules'), and does run-level tail target accuracy track run tail accuracy across "
                "the 15 gate-2 runs?",
    "method": "attention rollout (teacher all blocks, student all blocks or 6..11); last-block CLS attention; "
              "keep-only masked-teacher faithfulness sliced by head/tail class; within-group image bootstrap; "
              "Spearman",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 "
             "(keep-only); Wah et al. 2011 (CUB-200-2011)",
}

START = 6
WIDEN = 1.5
K = 29
NBOOT = 2000
BATCH = 50
ARMS = {
    "maskedkd": {1: "cub_lt_final_maskedkd_k29_s1_1103_234609_76d1",
                 2: "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 3: "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"},
    "rollout": {1: "cub_lt_final_rollout_k29_s1_1103_234609_8382",
                2: "cub_lt_final_rollout_k29_s2_1103_234647_9b4c",
                3: "cub_lt_final_rollout_k29_s3_1103_234704_0edf"},
    "teacher_topk": {1: "cub_lt_final_teacher_topk_k29_s1_1103_234627_9ef0",
                     2: "cub_lt_final_teacher_topk_k29_s2_1103_234655_cfa0",
                     3: "cub_lt_final_teacher_topk_k29_s3_1103_234716_6d91"},
    "trustfilt": {1: "cub_lt_final_trustfilt_k29_s1_1103_234609_dbe3",
                  2: "cub_lt_final_trustfilt_k29_s2_1103_234647_43e9",
                  3: "cub_lt_final_trustfilt_k29_s3_1103_234704_8732"},
    "trustmask_v1": {1: "cub_lt_final_trustmask_v1_k29_s1_1103_234609_10df",
                     2: "cub_lt_final_trustmask_v1_k29_s2_1103_234647_4e79",
                     3: "cub_lt_final_trustmask_v1_k29_s3_1103_234704_d748"},
}
GROUPS = (0, 1)
TAIL = 1
STATS = ("t_acc", "agree", "fid", "pt", "s_acc")


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


def _mask(api, method, S, x, tm):
    depth = len(S.blocks)
    if method == "teacher_topk":
        with torch.no_grad():
            so = api.student_forward(S, x, attn_layers=[depth - 1])
        return _topk(tm, K), so.logits
    with torch.no_grad():
        so = api.student_forward(S, x, attn_layers=list(range(depth)))
    if method == "maskedkd":
        return _topk(so.attn[depth - 1].float().mean(1)[:, 0, 1:], K), so.logits
    if method == "rollout":
        return _topk(_rollout([so.attn[i] for i in range(depth)]), K), so.logits
    late = _rollout([so.attn[i] for i in range(START, depth)])
    if method == "trustmask_v1":
        return _topk(late, K), so.logits
    allowed = _topk(tm, min(int(math.ceil(WIDEN * K)), tm.shape[1]))
    return _topk(late.masked_fill(~allowed, -1.0), K), so.logits


def run(api):
    runs = api.runs()
    used = {a: {str(s): [rid, runs[rid]["method"], runs[rid]["keep"], runs[rid]["seed"], runs[rid]["tier"],
                         runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]]
                for s, rid in sorted(ids.items())} for a, ids in ARMS.items()}
    api.write_json("students", used)
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    batches = []
    full_acc = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        tm = _rollout(out.attn)
        full = out.logits.float().log_softmax(-1)
        batches.append((x, y, tm, full))
        full_acc.append((full.argmax(-1) == y).float().cpu())
    groups = api.val_groups().numpy()
    full_acc = torch.cat(full_acc).numpy()

    per = {a: {s: [] for s in STATS} for a in ARMS}
    for a, ids in ARMS.items():
        for seed, rid in sorted(ids.items()):
            S = api.student(rid)
            acc = {s: [] for s in STATS}
            for x, y, tm, full in batches:
                keep, slog = _mask(api, a, S, x, tm)
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                acc["t_acc"].append((lm.argmax(-1) == y).float().cpu())
                acc["agree"].append((lm.argmax(-1) == full.argmax(-1)).float().cpu())
                acc["fid"].append((full.exp() * (full - lm)).sum(-1).float().cpu())
                acc["pt"].append(lm.gather(1, y.view(-1, 1)).squeeze(1).exp().float().cpu())
                acc["s_acc"].append((slog.float().argmax(-1) == y).float().cpu())
            for s in STATS:
                per[a][s].append(torch.cat(acc[s]).numpy())

    gen = numpy.random.default_rng(20261107)
    d = {a: {s: numpy.mean(numpy.stack(per[a][s]), 0) for s in STATS} for a in ARMS}
    out = {"n_val": int(len(groups)), "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": used, "full_teacher_acc": {str(g): _ci(full_acc[groups == g], gen) for g in GROUPS}}
    out["by_arm"] = {a: {s: {str(g): _ci(d[a][s][groups == g], gen) for g in GROUPS} for s in STATS} for a in ARMS}
    out["seedwise_group_mean"] = {a: {s: {str(g): [round(float(v[groups == g].mean()), 4) for v in per[a][s]]
                                          for g in GROUPS} for s in STATS} for a in ARMS}
    pairs = (("trustmask_v1", "maskedkd"), ("trustmask_v1", "teacher_topk"), ("trustfilt", "maskedkd"),
             ("rollout", "maskedkd"), ("teacher_topk", "maskedkd"), ("trustfilt", "trustmask_v1"))
    out["diff"] = {f"{p}-{q}": {s: {str(g): _ci((d[p][s] - d[q][s])[groups == g], gen) for g in GROUPS}
                                for s in STATS} for p, q in pairs}

    tw, sw, names = [], [], []
    for a, ids in ARMS.items():
        for i, (seed, rid) in enumerate(sorted(ids.items())):
            tw.append(float(per[a]["t_acc"][i][groups == TAIL].mean()))
            sw.append(float(runs[rid]["val_wga_last"]))
            names.append(rid)
    rho = _spearman(tw, sw)
    out["run_level"] = {"runs": names, "masked_teacher_tail_t_acc": [round(v, 4) for v in tw],
                        "student_val_tail_acc": sw, "spearman": round(rho, 4)}
    arm_t = [out["by_arm"][a]["t_acc"][str(TAIL)][0] for a in ARMS]
    arm_w = [float(numpy.mean([runs[rid]["val_wga_last"] for rid in ids.values()])) for ids in ARMS.values()]
    out["arm_level"] = {"arms": list(ARMS), "masked_teacher_tail_t_acc": [round(v, 4) for v in arm_t],
                        "student_val_tail_acc_mean": [round(v, 4) for v in arm_w],
                        "spearman": round(_spearman(arm_t, arm_w), 4)}

    dv = out["diff"]["trustmask_v1-maskedkd"]["t_acc"][str(TAIL)]
    out["verdict"] = {
        "P1_v1_minus_maskedkd_tail_tacc": dv,
        "P1_holds_ci_low_gt0": bool(dv[1] > 0),
        "P2_run_level_spearman": round(rho, 4),
        "P2_holds_ge_0.5": bool(rho >= 0.5),
    }
    api.write_json("summary", out)
