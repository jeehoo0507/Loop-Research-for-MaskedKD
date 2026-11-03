"""H-A43 (round 290, analysis only, no training): on Waterbirds K=29, does the quality of a trained student's OWN
last-block attention as a teacher input (one common rule for every method) track that student's WGA across methods?

H-A41 put each run's own token rule through the masked teacher and found rank correlation 0.30 with student WGA.
H-A42 found the online-MaskedKD student's own attention is a worse teacher input on group 2 than the frozen-trained
student's. Here every final student (maskedkd, maskedkd_frozen, rollout, teacher_topk, trustmask_v1, trustfilt @29
and ce @0; seeds 1, 2, 3) is read with the same rule: last-block head-mean CLS attention top-K (= what an online
MaskedKD selector would pick from that student). If the student bias left by training shows in its attention, the
per-run attention quality on the worst group should rank the runs like their WGA.

Per image (un-augmented val):
  t_acc   masked teacher (keep-only) top-1 == label
  cov     teacher rollout mass (all blocks, = the cached map on an un-augmented image) inside the kept set
  s_acc   student top-1 == label
Per run: group means; worst-group masked-teacher accuracy (min over groups of t_acc) and minority coverage (mean of
groups 1, 2 cov). Spearman rho over runs against run val WGA, with a run bootstrap 95% CI (2000 draws, numpy
default_rng 20261107). Per method: seed means.
No labels, groups or positions enter any mask; groups are used only to slice the results.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29, across methods, does the masked-teacher accuracy on the worst group when the "
                "teacher is fed the top-K of a trained student's own last-block CLS attention rank the runs like "
                "their student WGA?",
    "method": "last-block CLS attention top-K; attention rollout (teacher all blocks) for coverage; keep-only "
              "masked-teacher faithfulness sliced by Waterbirds group; Spearman rank correlation, run bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Hooker et al. NeurIPS 2019 "
             "(keep-only); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

K = 29
NBOOT = 2000
BATCH = 50
METHODS = {"maskedkd": 29, "maskedkd_frozen": 29, "rollout": 29, "teacher_topk": 29, "trustmask_v1": 29,
           "trustfilt": 29, "ce": 0}
GROUPS = (0, 1, 2, 3)
MINOR = (1, 2)
STATS = ("t_acc", "cov", "s_acc")


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


def _rank(v):
    r = numpy.empty(len(v))
    r[numpy.argsort(v, kind="stable")] = numpy.arange(len(v))
    return r


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(numpy.corrcoef(ra, rb)[0, 1])


def _rho_ci(a, b, gen):
    n = len(a)
    bs = []
    for _ in range(NBOOT):
        i = gen.integers(0, n, n)
        r = _spearman(a[i], b[i])
        if r == r:
            bs.append(r)
    bs = numpy.array(bs)
    return [round(_spearman(a, b), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4), int(n)]


def run(api):
    runs = api.runs()
    studs = []
    for rid, r in sorted(runs.items()):
        if (r.get("dataset") == "waterbirds" and r["tier"] == "final" and r["method"] in METHODS
                and r.get("keep") == METHODS[r["method"]] and r["seed"] in (1, 2, 3)):
            studs.append((r["method"], r["seed"], rid))
    api.write_json("students", [[m, s, rid, runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]]
                                for m, s, rid in studs])
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        batches.append((x, y, _rollout(out.attn)))
    groups = api.val_groups().numpy()

    rows = []
    for m, seed, rid in studs:
        S = api.student(rid)
        last = len(S.blocks) - 1
        acc = {s: [] for s in STATS}
        for x, y, tm in batches:
            with torch.no_grad():
                so = api.student_forward(S, x, attn_layers=[last])
            keep = _topk(so.attn[last].float().mean(1)[:, 0, 1:], K)
            with torch.no_grad():
                lm = api.teacher_forward_keepmask(T, x, keep).float()
            acc["t_acc"].append((lm.argmax(-1) == y).float().cpu())
            acc["cov"].append((tm * keep).sum(1).float().cpu())
            acc["s_acc"].append((so.logits.float().argmax(-1) == y).float().cpu())
        v = {s: torch.cat(acc[s]).numpy() for s in STATS}
        gm = {s: {str(g): round(float(v[s][groups == g].mean()), 4) for g in GROUPS} for s in STATS}
        rows.append({"method": m, "seed": seed, "run": rid, "val_wga": runs[rid]["val_wga_last"],
                     "val_acc": runs[rid]["val_acc_last"], "group_mean": gm,
                     "t_acc_worst_group": min(gm["t_acc"].values()),
                     "cov_minor": round(float(numpy.mean([gm["cov"][str(g)] for g in MINOR])), 4),
                     "s_acc_worst_group": min(gm["s_acc"].values())})

    gen = numpy.random.default_rng(20261107)
    wga = numpy.array([r["val_wga"] for r in rows], dtype=float)
    tw = numpy.array([r["t_acc_worst_group"] for r in rows], dtype=float)
    cm = numpy.array([r["cov_minor"] for r in rows], dtype=float)
    nomk = numpy.array([r["method"] != "maskedkd" for r in rows])
    rho = {"t_acc_worst_vs_wga": _rho_ci(tw, wga, gen), "cov_minor_vs_wga": _rho_ci(cm, wga, gen),
           "t_acc_worst_vs_wga_without_maskedkd": _rho_ci(tw[nomk], wga[nomk], gen),
           "cov_minor_vs_wga_without_maskedkd": _rho_ci(cm[nomk], wga[nomk], gen)}
    by_method = {}
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        if rs:
            by_method[m] = {k: round(float(numpy.mean([r[k] for r in rs])), 4)
                            for k in ("val_wga", "val_acc", "t_acc_worst_group", "cov_minor", "s_acc_worst_group")}
            by_method[m]["n_seeds"] = len(rs)
    low_tw = min(by_method, key=lambda m: by_method[m]["t_acc_worst_group"])
    out = {"k": K, "n_val": int(len(groups)), "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "n_runs": len(rows), "rho": rho, "by_method": by_method, "runs": rows,
           "verdict": {"P1_rho_t_acc_worst_vs_wga_ge_0.5": bool(rho["t_acc_worst_vs_wga"][0] >= 0.5),
                       "P2_maskedkd_lowest_t_acc_worst": bool(low_tw == "maskedkd"),
                       "R_lowest_t_acc_worst_method": low_tw}}
    api.write_json("summary", out)
