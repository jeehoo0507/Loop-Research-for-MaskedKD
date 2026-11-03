"""H-A40 (QUEUE 2026-11-04 01:40 KST, analysis after the gates): on Waterbirds, does the masked teacher the student
picks for get worse on the minority groups as training goes on, for MaskedKD but not for v1?

Gate 3 (H-A38) found online maskedkd@29 - frozen@29 WGA = -22.81 [-26.69, -18.93] ("loop"). This analysis looks
at the KD target itself along training, per val group (2*y + place, analysis only; group 1 = landbird on water,
group 2 = waterbird on land are the minority groups).

Students: Waterbirds final maskedkd@29 and trustmask_v1@29, seeds 1, 2, 3, at epoch 25, 50 and final.
Mask: each run's own token rule on the un-augmented val image, K = 29:
  maskedkd  last-block head-mean CLS attention top-K (harness builtin)
  v1        student rollout over blocks 6..11 top-K (selectors/trustmask_v1.py)
Per image:
  t_acc   masked teacher (keep-only, teacher_forward_keepmask) top-1 == label
  agree   masked teacher top-1 == full teacher top-1
  fid     KL(teacher full || teacher masked)
  cov     teacher rollout mass (all blocks, full image = the cached map on an un-augmented image) inside the kept set
  s_acc   student top-1 == label
Per image the 3 seeds are averaged; per group the mean and a within-group image bootstrap 95% CI (2000 draws,
numpy default_rng 20261104). Paired differences use the same images.
No labels, groups or positions enter any mask; groups are used only to slice the results.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29, along training (epoch 25, 50, final), is the masked teacher fed MaskedKD's "
                "student-picked tokens less correct and less covered by the teacher's own evidence on the minority "
                "groups than the one fed v1's tokens, and does that gap grow with training?",
    "method": "attention rollout (teacher all blocks, student blocks 6..11); last-block CLS attention; keep-only "
              "masked-teacher faithfulness (accuracy, top-1 agreement, KL) sliced by Waterbirds group; "
              "within-group image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 "
             "(keep-only); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

K = 29
START = 6
NBOOT = 2000
BATCH = 50
EPOCHS = (25, 50, "final")
METHODS = ("maskedkd", "trustmask_v1")
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
                depth = len(S.blocks)
                layers = [depth - 1] if m == "maskedkd" else list(range(START, depth))
                acc = {s: [] for s in STATS}
                for x, y, tm, full in batches:
                    with torch.no_grad():
                        so = api.student_forward(S, x, attn_layers=layers)
                    if m == "maskedkd":
                        score = so.attn[depth - 1].float().mean(1)[:, 0, 1:]
                    else:
                        score = _rollout([so.attn[i] for i in layers])
                    keep = _topk(score, K)
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    acc["t_acc"].append((lm.argmax(-1) == y).float().cpu())
                    acc["agree"].append((lm.argmax(-1) == full.argmax(-1)).float().cpu())
                    acc["fid"].append((full.exp() * (full - lm)).sum(-1).float().cpu())
                    acc["cov"].append((tm * keep).sum(1).float().cpu())
                    acc["s_acc"].append((so.logits.float().argmax(-1) == y).float().cpu())
                for s in STATS:
                    per[m][str(ep)][s].append(torch.cat(acc[s]).numpy())

    gen = numpy.random.default_rng(20261104)
    d = {m: {str(e): {s: numpy.mean(numpy.stack(per[m][str(e)][s]), 0) for s in STATS} for e in EPOCHS}
         for m in METHODS}
    seedwise = {m: {str(e): {s: {str(g): [round(float(v[groups == g].mean()), 4) for v in per[m][str(e)][s]]
                                 for g in GROUPS} for s in STATS} for e in EPOCHS} for m in METHODS}
    out = {"k": K, "n_val": int(len(groups)), "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": {m: {str(s): r for s, r in v.items()} for m, v in studs.items()}}
    mk, v1 = "maskedkd", "trustmask_v1"
    for ep in EPOCHS:
        e = str(ep)
        res = {"by_method": {m: {s: {str(g): _ci(d[m][e][s][groups == g], gen) for g in GROUPS} for s in STATS}
                             for m in METHODS},
               "mkd_minus_v1": {s: {str(g): _ci((d[mk][e][s] - d[v1][e][s])[groups == g], gen) for g in GROUPS}
                                for s in STATS},
               "seedwise_group_mean": {m: seedwise[m][e] for m in METHODS}}
        res["masked_teacher_wga"] = {m: round(min(res["by_method"][m]["t_acc"][str(g)][0] for g in GROUPS), 4)
                                     for m in METHODS}
        out[e] = res
        api.write_json(f"epoch_{e}", res)

    minor = numpy.isin(groups, MINOR)
    dyn = {}
    for s in ("t_acc", "cov", "fid"):
        dm = d[mk]["final"][s] - d[mk]["25"][s]
        dv = d[v1]["final"][s] - d[v1]["25"][s]
        dyn[s] = {"mkd_final_minus_e25": {str(g): _ci(dm[groups == g], gen) for g in GROUPS},
                  "v1_final_minus_e25": {str(g): _ci(dv[groups == g], gen) for g in GROUPS},
                  "did_mkd_minus_v1": {str(g): _ci((dm - dv)[groups == g], gen) for g in GROUPS},
                  "did_mkd_minus_v1_minor": _ci((dm - dv)[minor], gen)}
    out["dynamics"] = dyn
    api.write_json("dynamics", dyn)

    fin = out["final"]["mkd_minus_v1"]
    out["verdict"] = {
        "P1_tacc_lower_both_minor_groups": bool(all(fin["t_acc"][str(g)][2] < 0 for g in MINOR)),
        "P2_cov_lower_both_minor_groups": bool(all(fin["cov"][str(g)][2] < 0 for g in MINOR)),
        "P3_tacc_gap_grows_minor": bool(dyn["t_acc"]["did_mkd_minus_v1_minor"][2] < 0),
        "R_tacc_lower_majority_groups": {str(g): bool(fin["t_acc"][str(g)][2] < 0) for g in (0, 3)},
    }
    api.write_json("summary", out)
