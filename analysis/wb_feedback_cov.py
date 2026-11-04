"""H-A83 (round 372, analysis only, no training): positive control for the representation-level loop measure of
H-A81 (Aircraft) and H-A82 (cub_lt). Both found no gap in teacher-evidence coverage between the self-selecting
maskedkd@29 student and students whose attention never chose the teacher input, and read it as "inherited". On
Waterbirds gate 3 found a behavioural loop (H-A38: online - frozen WGA -22.81 [-26.69, -18.93]). If the same coverage
measure shows no gap here either, it does not see the loop that behaviour shows, and the H-A81/82 readings weaken.

Same token rule for every student (last-block head-mean CLS attention, top-K, K = 29, harness MaskedKD), at epoch 25,
50 and final, on the un-augmented val image:
  cov    teacher rollout mass (all blocks, full image = the cached map) inside the student's top-K
  q_y    masked teacher (keep-only, teacher_forward_keepmask) true-class probability
  t_acc  masked teacher top-1 == label
  s_acc  student top-1 == label
Per run: mean over val images, pooled and per Waterbirds group (2*y + place). Paired same-seed t 95% CI (df 2) for
maskedkd - {frozen, full, ce}, per epoch, and for the growth (maskedkd - ref) final - e25. No labels, groups or
positions enter any mask; labels and groups enter only the statistics.

Students (pinned run ids), Waterbirds final seeds 1, 2, 3 (gate 3 pairs and the gate 1 full / ce runs).
"""
import numpy
import torch

META = {
    "question": "on Waterbirds (final, seeds 1-3), where gate 3 showed a behavioural loop, does the maskedkd@29 "
                "student's last-block CLS top-29 cover less teacher rollout mass than the same rule applied to "
                "maskedkd_frozen@29, full@196 and ce students (pooled and per group), and does that gap grow?",
    "method": "attention rollout (teacher all blocks); last-block CLS attention top-K; keep-only masked teacher; "
              "paired t CI over seeds",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 "
             "(keep-only); Sagawa et al. ICLR 2020 (Waterbirds)",
}

K = 29
BATCH = 50
EPOCHS = (25, 50, "final")
STATS = ("cov", "q_y", "t_acc", "s_acc")
GROUPS = ("all", "0", "1", "2", "3")
REFS = ("frozen", "full", "ce")
ARMS = {
    "maskedkd": {1: "waterbirds_final_maskedkd_k29_s1_1031_201502_4be8",
                 2: "waterbirds_final_maskedkd_k29_s2_1103_170931_92bd",
                 3: "waterbirds_final_maskedkd_k29_s3_1103_170935_907a"},
    "frozen": {1: "waterbirds_final_maskedkd_frozen_k29_s1_1103_171021_f885",
               2: "waterbirds_final_maskedkd_frozen_k29_s2_1103_171026_2b13",
               3: "waterbirds_final_maskedkd_frozen_k29_s3_1103_171031_b65e"},
    "full": {1: "final_full_k196_s1_1030_201811_b764",
             2: "final_full_k196_s2_1030_212126_a1dc",
             3: "final_full_k196_s3_1030_220958_8d7e"},
    "ce": {1: "waterbirds_final_ce_k0_s1_1031_130203_4b89",
           2: "waterbirds_final_ce_k0_s2_1031_201229_603c",
           3: "waterbirds_final_ce_k0_s3_1031_201502_f7a9"},
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
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"),runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid]["snapshots"]]
                                    for s, rid in sorted(ids.items())} for a, ids in ARMS.items()})
    T = api.teacher.eval()
    depth_t = len(T.blocks)
    groups = api.val_groups().numpy()
    masks = {"all": numpy.ones(len(groups), dtype=bool)}
    masks.update({str(g): groups == g for g in range(4)})

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        batches.append((x, y, _rollout(out.attn)))

    res = {a: {str(e): {s: {g: {} for g in GROUPS} for s in STATS} for e in EPOCHS} for a in ARMS}
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
                    v = torch.cat(acc[s]).float().cpu().numpy()
                    for g in GROUPS:
                        res[a][str(ep)][s][g][seed] = round(float(v[masks[g]].mean()), 4)

    by_arm = {a: {e: {s: {g: round(float(numpy.mean(list(res[a][e][s][g].values()))), 4) for g in GROUPS}
                      for s in STATS} for e in res[a]} for a in ARMS}
    pair = {}
    for ref in REFS:
        for g in GROUPS:
            for s in STATS:
                for e in map(str, EPOCHS):
                    pair[f"maskedkd-{ref}:{s}_g{g}_{e}"] = _t95(
                        [res["maskedkd"][e][s][g][k] - res[ref][e][s][g][k] for k in (1, 2, 3)])
                pair[f"maskedkd-{ref}:{s}_g{g}_final_minus_e25"] = _t95(
                    [(res["maskedkd"]["final"][s][g][k] - res[ref]["final"][s][g][k])
                     - (res["maskedkd"]["25"][s][g][k] - res[ref]["25"][s][g][k]) for k in (1, 2, 3)])
    for g in GROUPS:
        for s in STATS:
            pair[f"full-ce:{s}_g{g}_final"] = _t95(
                [res["full"]["final"][s][g][k] - res["ce"]["final"][s][g][k] for k in (1, 2, 3)])
            for a in ARMS:
                pair[f"{a}:{s}_g{g}_final_minus_e25"] = _t95(
                    [res[a]["final"][s][g][k] - res[a]["25"][s][g][k] for k in (1, 2, 3)])
    p1 = pair["maskedkd-frozen:cov_gall_final"]
    p2 = pair["maskedkd-frozen:cov_gall_final_minus_e25"]
    p3 = pair["maskedkd-frozen:cov_g2_final"]
    out = {"k": K, "n_val": int(sum(len(b[1]) for b in batches)),
           "n_group": {g: int(masks[g].sum()) for g in GROUPS}, "by_arm": by_arm, "per_seed": res,
           "paired_t95": pair,
           "verdict": {"P1_maskedkd_minus_frozen_cov_all_final_ci_high_lt0": bool(p1[2] < 0),
                       "P2_maskedkd_minus_frozen_cov_all_growth_final_minus_e25_ci_high_lt0": bool(p2[2] < 0),
                       "P3_maskedkd_minus_frozen_cov_group2_final_ci_high_lt0": bool(p3[2] < 0),
                       "measure_sees_loop_P1_or_P3": bool(p1[2] < 0 or p3[2] < 0)}}
    api.write_json("summary", out)
