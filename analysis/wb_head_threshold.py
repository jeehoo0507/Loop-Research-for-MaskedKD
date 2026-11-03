"""H-A47 (round 294, analysis only, no training): on Waterbirds K=29, is the WGA gap between methods (which H-A46
removed by refitting the whole last layer) only a logit offset of the original head?

H-A46: refitting the last linear layer on val (group- or class-balanced) shrank the sd of method-mean WGA from 10.86
to about 1, so the gap lives in the head. A full refit changes both the decision direction and the offset. If only
the offset (class prior / calibration) differs, shifting the threshold of the original head's logit difference,
without groups and without new weights, should already remove most of the gap.

Every final student (maskedkd, maskedkd_frozen, rollout, teacher_topk, trustmask_v1, trustfilt @29 and ce @0;
seeds 1, 2, 3): original head logit difference d = logit[1] - logit[0] on the un-augmented val image (full image).
orig: predict d > 0 (recomputed val WGA).  thr: 2-fold CV by index parity; on the train fold pick the threshold t
(among the fold's own d values and 0) that maximises class-balanced accuracy (no group labels); predict d > t on the
held-out fold; pool the two folds. Per run: orig WGA, thr WGA, thr mean acc, thr gain = thr WGA - orig WGA, t in units
of the run's sd(d). Per method: seed means. Spread = sd over the 7 method means. Spearman rho over runs of thr WGA vs
orig WGA, run bootstrap 95% CI (2000 draws, numpy default_rng 20261111).
Group labels enter only the WGA read-out; no selection uses them.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29, does shifting only the threshold of the original head's logit difference "
                "(class-balanced, no groups) remove the WGA gap between methods that a full head refit removed?",
    "method": "original head logit difference, class-balanced threshold chosen by 2-fold CV; spread of method means; "
              "Spearman rank correlation with run bootstrap",
    "paper": "Kirichenko et al. ICLR 2023 (DFR, last layer re-training); Menon et al. ICLR 2021 (logit adjustment); "
             "Sagawa et al. ICLR 2020 (Waterbirds groups); Son et al. ECCV 2024 (MaskedKD)",
}

NBOOT = 2000
BATCH = 50
METHODS = {"maskedkd": 29, "maskedkd_frozen": 29, "rollout": 29, "teacher_topk": 29, "trustmask_v1": 29,
           "trustfilt": 29, "ce": 0}


def _logit_diff(M, xs):
    out = []
    for x in xs:
        with torch.no_grad():
            z = M(x).float()
        out.append(z[:, 1] - z[:, 0])
    return torch.cat(out)


def _accs(pred, y, g):
    ok = (pred == y).float()
    accs = [ok[g == k].mean().item() for k in range(4) if (g == k).any()]
    return 100.0 * min(accs), 100.0 * ok.mean().item(), [round(100.0 * a, 2) for a in accs]


def _best_t(d, y):
    """Threshold maximising class-balanced accuracy of (d > t) among the fold's own d values and 0."""
    cand = torch.cat([d, torch.zeros(1, device=d.device)]).unique()
    pos = (d[None, :] > cand[:, None])
    tpr = (pos & (y[None, :] == 1)).float().sum(1) / (y == 1).sum().clamp(min=1)
    tnr = (~pos & (y[None, :] == 0)).float().sum(1) / (y == 0).sum().clamp(min=1)
    bal = (tpr + tnr) / 2
    best = bal.max()
    idx = torch.nonzero(bal == best).flatten()
    ts = cand[idx]
    return ts[ts.abs().argmin()]


def _thr(d, y, g):
    n = d.shape[0]
    folds = torch.arange(n, device=d.device) % 2
    pred = torch.zeros(n, dtype=torch.long, device=d.device)
    ts = []
    for f in (0, 1):
        tr, te = folds != f, folds == f
        t = _best_t(d[tr], y[tr])
        ts.append(t.item())
        pred[te] = (d[te] > t).long()
    w, a, per = _accs(pred, y, g)
    return w, a, per, ts


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
    xs, ys = [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
    y = torch.cat(ys)
    groups = api.val_groups().to(y.device).long()

    rows = []
    for m, seed, rid in studs:
        d = _logit_diff(api.student(rid), xs)
        ow, oa, oper = _accs((d > 0).long(), y, groups)
        tw, ta, tper, ts = _thr(d, y, groups)
        sd = d.std().item()
        rows.append({"method": m, "seed": seed, "run": rid, "val_wga": runs[rid]["val_wga_last"],
                     "orig_wga": round(ow, 2), "orig_acc": round(oa, 2), "orig_group_acc": oper,
                     "thr_wga": round(tw, 2), "thr_acc": round(ta, 2), "thr_group_acc": tper,
                     "thr_gain": round(tw - ow, 2), "t_over_sd": round(float(numpy.mean(ts)) / max(sd, 1e-6), 4),
                     "t_folds": [round(t, 4) for t in ts], "sd_d": round(sd, 4)})

    gen = numpy.random.default_rng(20261111)
    ow = numpy.array([r["orig_wga"] for r in rows], dtype=float)
    tw = numpy.array([r["thr_wga"] for r in rows], dtype=float)
    nomk = numpy.array([r["method"] != "maskedkd" for r in rows])
    rho = {"thr_wga_vs_orig_wga": _rho_ci(tw, ow, gen),
           "thr_wga_vs_orig_wga_without_maskedkd": _rho_ci(tw[nomk], ow[nomk], gen)}
    keys = ("val_wga", "orig_wga", "orig_acc", "thr_wga", "thr_acc", "thr_gain", "t_over_sd")
    by_method = {}
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        if rs:
            by_method[m] = {k: round(float(numpy.mean([r[k] for r in rs])), 3) for k in keys}
            by_method[m]["n_seeds"] = len(rs)
    spread = {k: round(float(numpy.std([by_method[m][k] for m in by_method], ddof=1)), 3)
              for k in ("val_wga", "orig_wga", "thr_wga")}
    top_gain = max(by_method, key=lambda m: by_method[m]["thr_gain"])
    out = {"n_val": int(len(y)), "n_runs": len(rows), "group_counts": torch.bincount(groups, minlength=4).tolist(),
           "spread_sd_over_methods": spread, "rho": rho, "by_method": by_method, "runs": rows,
           "verdict": {"P1_thr_spread_le_half_orig_spread": bool(spread["thr_wga"] <= 0.5 * spread["orig_wga"]),
                       "P2_maskedkd_largest_thr_gain": bool(top_gain == "maskedkd"),
                       "R_largest_thr_gain_method": top_gain}}
    api.write_json("summary", out)
