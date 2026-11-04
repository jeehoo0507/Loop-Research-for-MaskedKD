"""H-A81 (round 370, analysis only, no training): the representation-level loop question on unbiased Aircraft. Does
training under its own last-block selection (maskedkd@29) pull the student's last-block CLS attention away from the
teacher's evidence, compared with students whose attention never chose the teacher input (full@196, ce)?

Gate 3 (frozen vs online) was not run on Aircraft. Here the same token rule (last-block head-mean CLS attention,
top-K, K = 29, harness MaskedKD) is applied to every student, at epoch 25, 50 and final, on the un-augmented val image:
  cov    teacher rollout mass (all blocks, full image = the cached map) inside the student's top-K
  q_y    masked teacher (keep-only, teacher_forward_keepmask) true-class probability
  t_acc  masked teacher top-1 == label
  s_acc  student top-1 == label
Per run: mean over val images. Paired same-seed t 95% CI (df 2) for maskedkd - full and maskedkd - ce, per epoch, and
for the growth (maskedkd - full) final - e25. No labels or positions enter any mask; labels enter only the statistics.

Students (pinned run ids), Aircraft final seeds 1, 2, 3.
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3), along training (e25, e50, final), does the maskedkd@29 student's "
                "last-block CLS top-29 cover less teacher rollout mass than the same rule applied to full@196 and "
                "ce students, and does that gap grow with training?",
    "method": "attention rollout (teacher all blocks); last-block CLS attention top-K; keep-only masked teacher; "
              "paired t CI over seeds",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 "
             "(keep-only); Maji et al. 2013 (FGVC-Aircraft)",
}

K = 29
BATCH = 50
EPOCHS = (25, 50, "final")
STATS = ("cov", "q_y", "t_acc", "s_acc")
ARMS = {
    "maskedkd": {1: "aircraft_final_maskedkd_k29_s1_1102_132227_6542",
                 2: "aircraft_final_maskedkd_k29_s2_1102_180321_52e2",
                 3: "aircraft_final_maskedkd_k29_s3_1102_180336_347e"},
    "full": {1: "aircraft_final_full_k196_s1_1102_123044_ed5d",
             2: "aircraft_final_full_k196_s2_1102_132227_123b",
             3: "aircraft_final_full_k196_s3_1102_132227_69cc"},
    "ce": {1: "aircraft_final_ce_k0_s1_1102_123044_6d5f",
           2: "aircraft_final_ce_k0_s2_1102_132227_80b7",
           3: "aircraft_final_ce_k0_s3_1102_132227_55cb"},
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
    for ref in ("full", "ce"):
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
    p1 = pair["maskedkd-full:cov_final"]
    p2 = pair["maskedkd-full:cov_final_minus_e25"]
    out = {"k": K, "n_val": int(sum(len(b[1]) for b in batches)), "by_arm": by_arm, "per_seed": res,
           "paired_t95": pair,
           "verdict": {"P1_maskedkd_minus_full_cov_final_ci_high_lt0": bool(p1[2] < 0),
                       "P2_maskedkd_minus_full_cov_growth_final_minus_e25_ci_high_lt0": bool(p2[2] < 0)}}
    api.write_json("summary", out)
