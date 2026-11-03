"""H-A42 (round 289, analysis only, no training): on Waterbirds K=29, does online MaskedKD training make the
student's own last-block attention a worse teacher input on the minority groups than training with the frozen
step-0 selector does?

Gate 3 (H-A38) found online maskedkd@29 - frozen@29 WGA = -22.81 [-26.69, -18.93] ("loop"). The frozen run picks
teacher tokens with the step-0 student, so its trained student's attention never feeds the teacher. Here both
trained students' attention is put through the same rule (last-block head-mean CLS attention top-K, = what an online
MaskedKD selector would pick from that student). If the online students' attention is worse on the minority groups,
the loop changed the student's attention itself; if not, the frozen gain comes only from which attention picked.

Students: Waterbirds final maskedkd@29 and maskedkd_frozen@29, seeds 1, 2, 3, at epoch 25, 50 and final.
Per image (un-augmented val):
  t_acc   masked teacher (keep-only) top-1 == label
  agree   masked teacher top-1 == full teacher top-1
  fid     KL(teacher full || teacher masked)
  cov     teacher rollout mass (all blocks, = the cached map on an un-augmented image) inside the kept set
  s_acc   student top-1 == label
Per image the 3 seeds are averaged; per group the mean and a within-group image bootstrap 95% CI (2000 draws,
numpy default_rng 20261106). Paired differences use the same images.
No labels, groups or positions enter any mask; groups are used only to slice the results.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29, is the masked teacher fed the top-K of a student's own last-block CLS attention "
                "less correct on the minority groups when that student was trained by online MaskedKD than when it "
                "was trained with the frozen step-0 selector, and is the gap already there at epoch 25?",
    "method": "last-block CLS attention top-K; attention rollout (teacher all blocks) for coverage; keep-only "
              "masked-teacher faithfulness sliced by Waterbirds group; within-group image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Hooker et al. NeurIPS 2019 "
             "(keep-only); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

K = 29
NBOOT = 2000
BATCH = 50
EPOCHS = (25, 50, "final")
METHODS = ("maskedkd", "maskedkd_frozen")
GROUPS = (0, 1, 2, 3)
MINOR = (1, 2)
STATS = ("t_acc", "agree", "fid", "cov", "s_acc")


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


def run(api):
    runs = api.runs()
    studs = {m: {} for m in METHODS}
    for rid, r in sorted(runs.items()):
        if (r.get("dataset") == "waterbirds" and r["tier"] == "final" and r["method"] in METHODS
                and r.get("keep") == K and r["seed"] in (1, 2, 3)):
            studs[r["method"]][r["seed"]] = rid
    api.write_json("students", {m: {str(s): [rid, runs[rid]["val_wga_last"], runs[rid]["val_acc_last"],
                                             runs[rid]["snapshots"]] for s, rid in sorted(d.items())}
                                for m, d in studs.items()})
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
                last = len(S.blocks) - 1
                acc = {s: [] for s in STATS}
                for x, y, tm, full in batches:
                    with torch.no_grad():
                        so = api.student_forward(S, x, attn_layers=[last])
                    keep = _topk(so.attn[last].float().mean(1)[:, 0, 1:], K)
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    acc["t_acc"].append((lm.argmax(-1) == y).float().cpu())
                    acc["agree"].append((lm.argmax(-1) == full.argmax(-1)).float().cpu())
                    acc["fid"].append((full.exp() * (full - lm)).sum(-1).float().cpu())
                    acc["cov"].append((tm * keep).sum(1).float().cpu())
                    acc["s_acc"].append((so.logits.float().argmax(-1) == y).float().cpu())
                for s in STATS:
                    per[m][str(ep)][s].append(torch.cat(acc[s]).numpy())

    gen = numpy.random.default_rng(20261106)
    d = {m: {str(e): {s: numpy.mean(numpy.stack(per[m][str(e)][s]), 0) for s in STATS} for e in EPOCHS}
         for m in METHODS}
    seedwise = {m: {str(e): {s: {str(g): [round(float(v[groups == g].mean()), 4) for v in per[m][str(e)][s]]
                                 for g in GROUPS} for s in STATS} for e in EPOCHS} for m in METHODS}
    out = {"k": K, "n_val": int(len(groups)), "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": {m: {str(s): r for s, r in v.items()} for m, v in studs.items()}}
    on, fr = METHODS
    for ep in EPOCHS:
        e = str(ep)
        res = {"by_method": {m: {s: {str(g): _ci(d[m][e][s][groups == g], gen) for g in GROUPS} for s in STATS}
                             for m in METHODS},
               "online_minus_frozen": {s: {str(g): _ci((d[on][e][s] - d[fr][e][s])[groups == g], gen)
                                           for g in GROUPS} for s in STATS},
               "seedwise_group_mean": {m: seedwise[m][e] for m in METHODS}}
        res["masked_teacher_wga"] = {m: round(min(res["by_method"][m]["t_acc"][str(g)][0] for g in GROUPS), 4)
                                     for m in METHODS}
        out[e] = res
        api.write_json(f"epoch_{e}", res)

    fin, e25 = out["final"]["online_minus_frozen"], out["25"]["online_minus_frozen"]
    out["verdict"] = {
        "P1_final_tacc_lower_both_minor_groups": bool(all(fin["t_acc"][str(g)][2] < 0 for g in MINOR)),
        "P2_e25_tacc_lower_both_minor_groups": bool(all(e25["t_acc"][str(g)][2] < 0 for g in MINOR)),
        "R_final_tacc_lower_majority_groups": {str(g): bool(fin["t_acc"][str(g)][2] < 0) for g in (0, 3)},
        "R_final_cov_lower_both_minor_groups": bool(all(fin["cov"][str(g)][2] < 0 for g in MINOR)),
    }
    api.write_json("summary", out)
