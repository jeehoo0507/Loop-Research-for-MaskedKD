"""H-A82 (round 371, analysis only, no training): the representation-level loop question (H-A81) on cub_lt, where
gate 3 was run (H-A66: online - frozen tail +2.22 [-1.22, +5.66], "inherited"). Does training under its own last-block
selection (maskedkd@29) pull the student's last-block CLS attention away from the teacher's evidence, compared with
students whose own attention never chose the teacher input (maskedkd_frozen@29 = same budget, selection from the fixed
initial student; full@196; ce)?

Same token rule for every student (last-block head-mean CLS attention, top-K, K = 29, harness MaskedKD), at epoch 25,
50 and final, on the un-augmented val image:
  cov    teacher rollout mass (all blocks, full image = the cached map) inside the student's top-K
  q_y    masked teacher (keep-only, teacher_forward_keepmask) true-class probability
  t_acc  masked teacher top-1 == label
  s_acc  student top-1 == label
Per run: mean over val images. Paired same-seed t 95% CI (df 2) for maskedkd - {frozen, full, ce}, per epoch, and for
the growth (maskedkd - ref) final - e25. No labels or positions enter any mask; labels enter only the statistics.

Students (pinned run ids), cub_lt final seeds 1, 2, 3.
"""
import numpy
import torch

META = {
    "question": "on cub_lt (final, seeds 1-3), along training (e25, e50, final), does the maskedkd@29 student's "
                "last-block CLS top-29 cover less teacher rollout mass than the same rule applied to "
                "maskedkd_frozen@29, full@196 and ce students, and does that gap grow with training?",
    "method": "attention rollout (teacher all blocks); last-block CLS attention top-K; keep-only masked teacher; "
              "paired t CI over seeds",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 "
             "(keep-only); Wah et al. 2011 (CUB-200-2011); Liu et al. CVPR 2019 (long-tail split)",
}

K = 29
BATCH = 50
EPOCHS = (25, 50, "final")
STATS = ("cov", "q_y", "t_acc", "s_acc")
REFS = ("frozen", "full", "ce")
ARMS = {
    "maskedkd": {1: "cub_lt_final_maskedkd_k29_s1_1103_234609_76d1",
                 2: "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 3: "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"},
    "frozen": {1: "cub_lt_final_maskedkd_frozen_k29_s1_1104_074618_8245",
               2: "cub_lt_final_maskedkd_frozen_k29_s2_1104_074622_647c",
               3: "cub_lt_final_maskedkd_frozen_k29_s3_1104_074627_dd2d"},
    "full": {1: "cub_lt_final_full_k196_s1_1103_234609_88d5",
             2: "cub_lt_final_full_k196_s2_1103_234647_7b36",
             3: "cub_lt_final_full_k196_s3_1103_234704_3d84"},
    "ce": {1: "cub_lt_final_ce_k0_s1_1103_234642_ee32",
           2: "cub_lt_final_ce_k0_s2_1103_234700_71de",
           3: "cub_lt_final_ce_k0_s3_1103_234720_e74b"},
}


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


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid]["method"], runs[rid]["keep"], runs[rid]["seed"],
                                             runs[rid]["val_acc_last"], runs[rid]["snapshots"]]
                                    for s, rid in sorted(ids.items())} for a, ids in ARMS.items()})
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        batches.append((x, y, _rollout(out.attn)))

    res = {a: {str(e): {s: {} for s in STATS} for e in EPOCHS} for a in ARMS}
    for a, ids in ARMS.items():
        for seed, rid in sorted(ids.items()):
            for ep in EPOCHS:
                S = api.student(rid, epoch=None if ep == "final" else ep)
                depth = len(S.blocks)
                acc = {s: [] for s in STATS}
                for x, y, tm in batches:
                    with torch.no_grad():
                        so = api.student_forward(S, x, attn_layers=[depth - 1])
                        keep = _topk(so.attn[depth - 1].float().mean(1)[:, 0, 1:], K)
                        lq = api.teacher_forward_keepmask(T, x, keep).float()
                    acc["cov"].append((tm * keep).sum(1).float())
                    acc["q_y"].append(lq.softmax(-1).gather(1, y[:, None])[:, 0])
                    acc["t_acc"].append((lq.argmax(-1) == y).float())
                    acc["s_acc"].append((so.logits.float().argmax(-1) == y).float())
                for s in STATS:
                    res[a][str(ep)][s][seed] = round(torch.cat(acc[s]).mean().item(), 4)

    by_arm = {a: {e: {s: round(float(numpy.mean(list(res[a][e][s].values()))), 4) for s in STATS}
                  for e in res[a]} for a in ARMS}
    pair = {}
    for ref in REFS:
        for e in map(str, EPOCHS):
            for s in STATS:
                pair[f"maskedkd-{ref}:{s}_{e}"] = _t95([res["maskedkd"][e][s][k] - res[ref][e][s][k]
                                                        for k in (1, 2, 3)])
        for s in STATS:
            pair[f"maskedkd-{ref}:{s}_final_minus_e25"] = _t95(
                [(res["maskedkd"]["final"][s][k] - res[ref]["final"][s][k])
                 - (res["maskedkd"]["25"][s][k] - res[ref]["25"][s][k]) for k in (1, 2, 3)])
    for s in STATS:
        pair[f"full-ce:{s}_final"] = _t95([res["full"]["final"][s][k] - res["ce"]["final"][s][k] for k in (1, 2, 3)])
        for a in ARMS:
            pair[f"{a}:{s}_final_minus_e25"] = _t95([res[a]["final"][s][k] - res[a]["25"][s][k] for k in (1, 2, 3)])
    p1 = pair["maskedkd-frozen:cov_final"]
    p2 = pair["maskedkd-frozen:cov_final_minus_e25"]
    out = {"k": K, "n_val": int(sum(len(b[1]) for b in batches)), "by_arm": by_arm, "per_seed": res,
           "paired_t95": pair,
           "verdict": {"P1_maskedkd_minus_frozen_cov_final_ci_high_lt0": bool(p1[2] < 0),
                       "P2_maskedkd_minus_frozen_cov_growth_final_minus_e25_ci_high_lt0": bool(p2[2] < 0)}}
    api.write_json("summary", out)
