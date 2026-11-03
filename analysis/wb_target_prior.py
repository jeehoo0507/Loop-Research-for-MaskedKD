"""H-A48 (round 295, analysis only, no training): on Waterbirds K=29, does the class prior of each run's masked-teacher
KD target explain the logit offset of its student head?

H-A47: shifting only the threshold of the original head's logit difference removed most of the WGA gap between
methods (sd 10.86 -> 1.64); every run's best threshold was on the waterbird side and maskedkd's moved most. If the
offset comes from the KD target, runs whose masked teacher (own token rule, own student) gives less waterbird mass
should have students that give less waterbird mass and need a larger threshold shift.

Students: Waterbirds final maskedkd, rollout, teacher_topk, trustmask_v1, trustfilt @29, seeds 1, 2, 3 (15 runs).
(maskedkd_frozen is left out: its step-0 student is not available to the analysis API. ce has no KD target.)
Mask: each run's own rule on the un-augmented val image (same as H-A41/H-A44), applied with the run's epoch-25
student and with the final student.
Per run, over all val images:
  qT_ep   mean p_T^mask(waterbird)  with the epoch-ep student's mask (TAU = 1)
  qS      mean p_S(waterbird)        final student, full image
  dS      mean logit difference z[1] - z[0] of the final student, in units of its sd (offset; more negative = more
          landbird-leaning)
Reference: the full (unmasked) teacher's mean p_T(waterbird); the val label prior.
Spearman rho over the 15 runs: qT_25 vs qS, qT_25 vs dS, qT_25 vs run val WGA; run bootstrap 95% CI (2000 draws,
numpy default_rng 20261112). Per method: seed means.
No labels, groups or positions enter any mask; labels enter only the label prior; groups are not used.
"""
import math

import numpy
import torch

META = {
    "question": "on Waterbirds K=29, does the mean waterbird probability of each run's masked-teacher KD target (own "
                "token rule, epoch-25 student) rank the 15 runs like the waterbird probability and logit offset of "
                "their final student?",
    "method": "attention rollout (teacher all blocks, student all / 6..11 blocks); last-block CLS attention; keep-only "
              "masked teacher; class marginal of soft targets; Spearman, run bootstrap",
    "paper": "Hinton et al. 2015 (KD soft targets); Menon et al. ICLR 2021 (logit adjustment); Son et al. ECCV 2024 "
             "(MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Sagawa et al. ICLR 2020 (Waterbirds)",
}

K = 29
START = 6
WIDEN = 1.5
NBOOT = 2000
BATCH = 50
EPOCHS = (25, "final")
METHODS = ("maskedkd", "rollout", "teacher_topk", "trustmask_v1", "trustfilt")


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
    api.write_json("students", [[m, s, rid, runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]]
                                for m, s, rid in studs])
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    batches, ys, pfull = [], [], []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        batches.append((x, _rollout(out.attn)))
        ys.append(y)
        pfull.append(out.logits.float().softmax(-1)[:, 1].cpu())
    y = torch.cat(ys).float()
    ref = {"label_prior_waterbird": round(float(y.mean()), 4),
           "full_teacher_q": round(float(torch.cat(pfull).mean()), 4)}

    rows = []
    for m, seed, rid in studs:
        row = {"method": m, "seed": seed, "run": rid, "val_wga": runs[rid]["val_wga_last"],
               "val_acc": runs[rid]["val_acc_last"]}
        for ep in EPOCHS:
            S = api.student(rid, epoch=None if ep == "final" else ep)
            qt, qs, ds = [], [], []
            for x, tm in batches:
                keep, slog = _mask(api, m, S, x, tm)
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float()
                qt.append(lm.softmax(-1)[:, 1].cpu())
                sl = slog.float()
                qs.append(sl.softmax(-1)[:, 1].cpu())
                ds.append((sl[:, 1] - sl[:, 0]).cpu())
            d = torch.cat(ds)
            row[f"qT_{ep}"] = round(float(torch.cat(qt).mean()), 4)
            if ep == "final":
                row["qS"] = round(float(torch.cat(qs).mean()), 4)
                row["dS"] = round(float(d.mean() / d.std().clamp(min=1e-6)), 4)
        rows.append(row)

    gen = numpy.random.default_rng(20261112)
    col = lambda k: numpy.array([r[k] for r in rows], dtype=float)
    rho = {"qT_25_vs_qS": _rho_ci(col("qT_25"), col("qS"), gen),
           "qT_25_vs_dS": _rho_ci(col("qT_25"), col("dS"), gen),
           "qT_25_vs_wga": _rho_ci(col("qT_25"), col("val_wga"), gen),
           "qT_final_vs_qS": _rho_ci(col("qT_final"), col("qS"), gen),
           "qS_vs_wga": _rho_ci(col("qS"), col("val_wga"), gen)}
    nomk = numpy.array([r["method"] != "maskedkd" for r in rows])
    rho["qT_25_vs_qS_without_maskedkd"] = _rho_ci(col("qT_25")[nomk], col("qS")[nomk], gen)
    keys = ("val_wga", "qT_25", "qT_final", "qS", "dS")
    by_method = {}
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        if rs:
            by_method[m] = {k: round(float(numpy.mean([r[k] for r in rs])), 4) for k in keys}
            by_method[m]["n_seeds"] = len(rs)
    low = min(by_method, key=lambda m: by_method[m]["qT_25"])
    out = {"k": K, "n_val": int(len(y)), "n_runs": len(rows), "reference": ref, "rho": rho,
           "by_method": by_method, "runs": rows,
           "verdict": {"P1_rho_qT25_vs_qS_ge_0.5": bool(rho["qT_25_vs_qS"][0] >= 0.5),
                       "P2_maskedkd_lowest_qT25": bool(low == "maskedkd"),
                       "R_lowest_qT25_method": low}}
    api.write_json("summary", out)
