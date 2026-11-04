"""H-A64 (round 339, analysis only, no training): on cub_lt (gate 2), is the v1 - maskedkd gap in masked-teacher
target accuracy already present at the e25 snapshot, and does it grow by the final epoch?

H-A62 (final weights): v1 - maskedkd t_acc tail +0.111, head +0.128. On Waterbirds (H-A54) the maskedkd rule's
majority pull was already present at e25 and did not grow. This measures the same target statistics as H-A62 at
the e25, e50 and final snapshots of the 6 maskedkd and trustmask_v1 runs.

Students (pinned run ids), cub_lt final seeds 1, 2, 3, K = 29:
  maskedkd       student last-block head-mean CLS attention top-K
  trustmask_v1   student late-block (6..11) rollout top-K (the teacher map only weights the KD term)
Per image, per epoch:
  t_acc   masked teacher (keep-only) top-1 == label
  pt      masked teacher p(label)
  s_acc   student top-1 == label
Per image the 3 seeds are averaged; per group the mean and a within-group image bootstrap 95% CI (2000 draws,
numpy default_rng 20261109). Growth = (v1 - maskedkd) at final minus the same at e25, on the same images.
No labels, groups or positions enter any mask; groups are used only to slice the results.
"""
import numpy
import torch

META = {
    "question": "on cub_lt K=29, is the v1 - maskedkd masked-teacher target accuracy gap already present at e25, "
                "and does it grow by the final epoch?",
    "method": "attention rollout (student blocks 6..11); last-block CLS attention; keep-only masked-teacher "
              "accuracy at epoch snapshots sliced by head/tail class; within-group image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 "
             "(keep-only); Wah et al. 2011 (CUB-200-2011)",
}

START = 6
K = 29
NBOOT = 2000
BATCH = 50
EPOCHS = (25, 50, "final")
ARMS = {
    "maskedkd": {1: "cub_lt_final_maskedkd_k29_s1_1103_234609_76d1",
                 2: "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 3: "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"},
    "trustmask_v1": {1: "cub_lt_final_trustmask_v1_k29_s1_1103_234609_10df",
                     2: "cub_lt_final_trustmask_v1_k29_s2_1103_234647_4e79",
                     3: "cub_lt_final_trustmask_v1_k29_s3_1103_234704_d748"},
}
GROUPS = (0, 1)
TAIL = 1
STATS = ("t_acc", "pt", "s_acc")


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


def _mask(api, method, S, x):
    depth = len(S.blocks)
    with torch.no_grad():
        so = api.student_forward(S, x, attn_layers=list(range(depth)))
    if method == "maskedkd":
        return _topk(so.attn[depth - 1].float().mean(1)[:, 0, 1:], K), so.logits
    return _topk(_rollout([so.attn[i] for i in range(START, depth)]), K), so.logits


def run(api):
    runs = api.runs()
    used = {a: {str(s): [rid, runs[rid]["method"], runs[rid]["keep"], runs[rid]["seed"], runs[rid]["tier"],
                         runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]]
                for s, rid in sorted(ids.items())} for a, ids in ARMS.items()}
    api.write_json("students", used)
    T = api.teacher.eval()

    batches = [(x, y) for x, y in api.val_batches(batch=BATCH)]
    groups = api.val_groups().numpy()

    per = {e: {a: {s: [] for s in STATS} for a in ARMS} for e in EPOCHS}
    for e in EPOCHS:
        ep = None if e == "final" else e
        for a, ids in ARMS.items():
            for seed, rid in sorted(ids.items()):
                S = api.student(rid, epoch=ep)
                acc = {s: [] for s in STATS}
                for x, y in batches:
                    keep, slog = _mask(api, a, S, x)
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    acc["t_acc"].append((lm.argmax(-1) == y).float().cpu())
                    acc["pt"].append(lm.gather(1, y.view(-1, 1)).squeeze(1).exp().float().cpu())
                    acc["s_acc"].append((slog.float().argmax(-1) == y).float().cpu())
                for s in STATS:
                    per[e][a][s].append(torch.cat(acc[s]).numpy())

    gen = numpy.random.default_rng(20261109)
    d = {e: {a: {s: numpy.mean(numpy.stack(per[e][a][s]), 0) for s in STATS} for a in ARMS} for e in EPOCHS}
    gap = {e: {s: d[e]["trustmask_v1"][s] - d[e]["maskedkd"][s] for s in STATS} for e in EPOCHS}
    out = {"n_val": int(len(groups)), "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": used}
    out["by_epoch_arm"] = {str(e): {a: {s: {str(g): _ci(d[e][a][s][groups == g], gen) for g in GROUPS}
                                        for s in STATS} for a in ARMS} for e in EPOCHS}
    out["seedwise_group_mean"] = {str(e): {a: {s: {str(g): [round(float(v[groups == g].mean()), 4)
                                                           for v in per[e][a][s]] for g in GROUPS}
                                               for s in STATS} for a in ARMS} for e in EPOCHS}
    out["v1_minus_maskedkd"] = {str(e): {s: {str(g): _ci(gap[e][s][groups == g], gen) for g in GROUPS}
                                         for s in STATS} for e in EPOCHS}
    out["growth_final_minus_e25"] = {s: {str(g): _ci((gap["final"][s] - gap[25][s])[groups == g], gen)
                                         for g in GROUPS} for s in STATS}
    out["growth_final_minus_e50"] = {s: {str(g): _ci((gap["final"][s] - gap[50][s])[groups == g], gen)
                                         for g in GROUPS} for s in STATS}

    p1 = out["v1_minus_maskedkd"]["25"]["t_acc"][str(TAIL)]
    p2 = out["growth_final_minus_e25"]["t_acc"][str(TAIL)]
    out["verdict"] = {
        "P1_e25_v1_minus_maskedkd_tail_tacc": p1,
        "P1_holds_ci_low_gt0": bool(p1[1] > 0),
        "P2_tail_growth_final_minus_e25": p2,
        "P2_holds_ci_low_gt0": bool(p2[1] > 0),
    }
    api.write_json("summary", out)
