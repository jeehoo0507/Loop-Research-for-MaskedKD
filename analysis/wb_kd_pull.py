"""H-A44 (round 291, analysis only, no training): on Waterbirds K=29, does the direction of each run's KD signal on
the minority groups (does the masked-teacher soft label pull the student toward or away from the true class)
rank the runs like their final WGA?

H-A41 found the accuracy of each method's masked-teacher target ranks the five methods only weakly like their WGA
(rho 0.30; teacher_topk has a poor target but the second-best WGA). Target accuracy ignores the student. With
TAU = 1 and binary labels, the KD gradient on the student's true-class logit is p_T^mask(y) - p_S(y): positive
means the KD term pushes the student toward the true class, negative means away. This measures that pull with each
run's own token rule on its own mid-training (epoch 25) student, and at the final student for the record.

Students: Waterbirds final maskedkd, rollout, teacher_topk, trustmask_v1, trustfilt @29, seeds 1, 2, 3.
(maskedkd_frozen is left out: its step-0 student is not available to the analysis API.)
Mask: each run's own rule on the un-augmented val image (same as H-A41):
  maskedkd      last-block head-mean CLS attention top-K
  rollout       student rollout over all blocks top-K
  teacher_topk  teacher rollout over all blocks (= the cached map on an un-augmented image) top-K
  trustmask_v1  student rollout over blocks 6..11 top-K (its KD weight is not a mask; not applied here)
  trustfilt     student rollout over blocks 6..11 top-K inside the teacher map's top ceil(1.5 K)
Per image:
  pull    p_T^mask(y) - p_S(y)       (TAU = 1)
  harm    1[pull < 0]
  pt      p_T^mask(y)                (soft masked-teacher target on the true class)
  ps      p_S(y)
Per run: group means; minority value = mean over images of groups 1 and 2. Spearman rho over the 15 runs against run
val WGA, run bootstrap 95% CI (2000 draws, numpy default_rng 20261108). Per method: seed means.
No labels, groups or positions enter any mask; labels enter only the measured quantities; groups only slice results.
"""
import math

import numpy
import torch

META = {
    "question": "on Waterbirds K=29, does the minority-group KD pull p_T^mask(y) - p_S(y) of each run's own token "
                "rule on its epoch-25 student rank the 15 runs (5 methods x 3 seeds) like their final student WGA?",
    "method": "attention rollout (teacher all blocks, student all / 6..11 blocks); last-block CLS attention; keep-only "
              "masked teacher; KD gradient sign at TAU 1 sliced by Waterbirds group; Spearman, run bootstrap",
    "paper": "Hinton et al. 2015 (KD gradient); Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); "
             "Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

K = 29
START = 6
WIDEN = 1.5
NBOOT = 2000
BATCH = 50
EPOCHS = (25, "final")
METHODS = ("maskedkd", "rollout", "teacher_topk", "trustmask_v1", "trustfilt")
GROUPS = (0, 1, 2, 3)
MINOR = (1, 2)
STATS = ("pull", "harm", "pt", "ps")


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


def _mask(api, m, S, x, tm):
    depth = len(S.blocks)
    with torch.no_grad():
        so = api.student_forward(S, x, attn_layers=list(range(depth)))
    late = _rollout([so.attn[i] for i in range(START, depth)])
    if m == "maskedkd":
        keep = _topk(so.attn[depth - 1].float().mean(1)[:, 0, 1:], K)
    elif m == "rollout":
        keep = _topk(_rollout([so.attn[i] for i in range(depth)]), K)
    elif m == "teacher_topk":
        keep = _topk(tm, K)
    elif m == "trustmask_v1":
        keep = _topk(late, K)
    else:
        allowed = _topk(tm, min(int(math.ceil(WIDEN * K)), tm.shape[1]))
        keep = _topk(late.masked_fill(~allowed, -1.0), K)
    return keep, so.logits


def run(api):
    runs = api.runs()
    studs = []
    for rid, r in sorted(runs.items()):
        if (r.get("dataset") == "waterbirds" and r["tier"] == "final" and r["method"] in METHODS
                and r.get("keep") == K and r["seed"] in (1, 2, 3)):
            studs.append((r["method"], r["seed"], rid))
    api.write_json("students", [[m, s, rid, runs[rid]["val_wga_last"], runs[rid]["val_acc_last"],
                                 runs[rid]["snapshots"]] for m, s, rid in studs])
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        batches.append((x, y, _rollout(out.attn)))
    groups = api.val_groups().numpy()
    minor = numpy.isin(groups, MINOR)

    rows = []
    for m, seed, rid in studs:
        row = {"method": m, "seed": seed, "run": rid, "val_wga": runs[rid]["val_wga_last"],
               "val_acc": runs[rid]["val_acc_last"]}
        for ep in EPOCHS:
            S = api.student(rid, epoch=None if ep == "final" else ep)
            acc = {s: [] for s in STATS}
            for x, y, tm in batches:
                keep, slog = _mask(api, m, S, x, tm)
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float()
                yi = y.to(lm.device).long().unsqueeze(1)
                pt = lm.softmax(-1).gather(1, yi).squeeze(1)
                ps = slog.float().softmax(-1).gather(1, yi.to(slog.device)).squeeze(1).to(pt.device)
                acc["pull"].append((pt - ps).cpu())
                acc["harm"].append((pt < ps).float().cpu())
                acc["pt"].append(pt.cpu())
                acc["ps"].append(ps.cpu())
            v = {s: torch.cat(acc[s]).numpy() for s in STATS}
            row[str(ep)] = {"group_mean": {s: {str(g): round(float(v[s][groups == g].mean()), 4) for g in GROUPS}
                                           for s in STATS},
                            "minor": {s: round(float(v[s][minor].mean()), 4) for s in STATS}}
        rows.append(row)

    gen = numpy.random.default_rng(20261108)
    wga = numpy.array([r["val_wga"] for r in rows], dtype=float)
    rho = {}
    for ep in EPOCHS:
        for s in STATS:
            rho[f"{ep}_minor_{s}_vs_wga"] = _rho_ci(numpy.array([r[str(ep)]["minor"][s] for r in rows]), wga, gen)
    nomk = numpy.array([r["method"] != "maskedkd" for r in rows])
    rho["25_minor_pull_vs_wga_without_maskedkd"] = _rho_ci(
        numpy.array([r["25"]["minor"]["pull"] for r in rows])[nomk], wga[nomk], gen)
    by_method = {}
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        if rs:
            by_method[m] = {"val_wga": round(float(numpy.mean([r["val_wga"] for r in rs])), 4),
                            "n_seeds": len(rs)}
            for ep in EPOCHS:
                by_method[m][str(ep)] = {s: round(float(numpy.mean([r[str(ep)]["minor"][s] for r in rs])), 4)
                                         for s in STATS}
    top = max(by_method, key=lambda m: by_method[m]["25"]["pull"])
    out = {"k": K, "n_val": int(len(groups)), "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "n_runs": len(rows), "rho": rho, "by_method": by_method, "runs": rows,
           "verdict": {"P1_rho_25_minor_pull_vs_wga_ge_0.5": bool(rho["25_minor_pull_vs_wga"][0] >= 0.5),
                       "P2_trustfilt_highest_25_minor_pull": bool(top == "trustfilt"),
                       "R_highest_25_minor_pull_method": top}}
    api.write_json("summary", out)
