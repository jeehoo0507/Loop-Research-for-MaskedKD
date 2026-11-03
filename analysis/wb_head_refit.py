"""H-A46 (round 293, analysis only, no training): on Waterbirds K=29, is the WGA gap between methods in the
student's classifier head rather than in its feature?

H-A45 found every final student's pre-logit feature carries about the same within-class place information
(place_bacc 0.92-0.95, rho with WGA -0.18, uncertain), so the feature does not rank runs like WGA. If the gap is in
how the head uses the feature, refitting only the last linear layer with group-balanced weights (DFR, Kirichenko et
al. ICLR 2023) should shrink the spread between methods and recover maskedkd the most.

Every final student (maskedkd, maskedkd_frozen, rollout, teacher_topk, trustmask_v1, trustfilt @29 and ce @0;
seeds 1, 2, 3): pre-logit feature on the un-augmented val image (full image, same as H-A45). Closed-form ridge
(lam 100, targets +-1 by class, standardised features, intercept), 2-fold CV by index parity; held-out predictions
pooled over the two folds. Two weightings of the training fold:
  grp: each of the 4 groups weighs the same (DFR);  cls: each class weighs the same (control: refit without groups).
Per run: refit WGA (%, min over 4 groups of pooled held-out accuracy) and mean accuracy; gain = refit WGA - run val
WGA. Per method: seed means. Spread = sd over the 7 method means. Spearman rho over runs of grp WGA vs run val WGA,
run bootstrap 95% CI (2000 draws, numpy default_rng 20261110).
Group labels enter only this offline head refit; no selection uses them.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29, does refitting only the last linear layer with group-balanced weights (DFR) "
                "on the final students' features remove the WGA gap between methods?",
    "method": "group-balanced closed-form ridge head on the pre-logit feature, 2-fold CV; spread of method means; "
              "Spearman rank correlation with run bootstrap",
    "paper": "Kirichenko et al. ICLR 2023 (DFR, last layer re-training); Sagawa et al. ICLR 2020 (Waterbirds "
             "groups); Son et al. ECCV 2024 (MaskedKD)",
}

NBOOT = 2000
BATCH = 50
LAM = 100.0
METHODS = {"maskedkd": 29, "maskedkd_frozen": 29, "rollout": 29, "teacher_topk": 29, "trustmask_v1": 29,
           "trustfilt": 29, "ce": 0}


def _feat(M, x):
    """Pre-logit feature on the full image (same path as the timm forward)."""
    h = M.patch_embed(x)
    h = M._pos_embed(h)
    h = M.patch_drop(h)
    h = M.norm_pre(h)
    h = M.blocks(h)
    h = M.norm(h)
    return M.forward_head(h, pre_logits=True)


def _refit(F, y, g, weighting, lam=LAM):
    """2-fold CV (index parity) weighted ridge head (closed form, targets +-1 by class).
    weighting 'grp': every group sums to the same weight; 'cls': every class does.
    Returns pooled held-out (WGA %, mean acc %)."""
    n = F.shape[0]
    folds = torch.arange(n, device=F.device) % 2
    pred = torch.zeros(n, dtype=torch.long, device=F.device)
    for f in (0, 1):
        tr, te = folds != f, folds == f
        mu, sd = F[tr].mean(0), F[tr].std(0).clamp(min=1e-6)
        Xtr, Xte = (F[tr] - mu) / sd, (F[te] - mu) / sd
        key = g[tr] if weighting == "grp" else y[tr]
        cnt = torch.bincount(key, minlength=4).double().clamp(min=1)
        sw = 1.0 / cnt[key]
        sw = sw / sw.sum()
        ttr = y[tr].double() * 2 - 1
        X = torch.cat([Xtr.double(), torch.ones(Xtr.shape[0], 1, device=F.device, dtype=torch.float64)], 1)
        A = X.T @ (X * sw[:, None]) + lam / Xtr.shape[0] * torch.eye(X.shape[1], device=F.device,
                                                                     dtype=torch.float64)
        wb = torch.linalg.solve(A, X.T @ (sw * ttr))
        w, b = wb[:-1].float(), wb[-1].float()
        pred[te] = ((Xte @ w + b) > 0).long()
    ok = (pred == y).float()
    accs = [ok[g == k].mean().item() for k in range(4) if (g == k).any()]
    return 100.0 * min(accs), 100.0 * ok.mean().item()


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


def _features(M, xs):
    out = []
    for x in xs:
        with torch.no_grad():
            out.append(_feat(M, x).float())
    return torch.cat(out)


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

    Ft = _features(api.teacher.eval(), xs)
    teacher = {w: [round(v, 2) for v in _refit(Ft, y, groups, w)] for w in ("grp", "cls")}

    rows = []
    for m, seed, rid in studs:
        F = _features(api.student(rid), xs)
        gw, ga = _refit(F, y, groups, "grp")
        cw, ca = _refit(F, y, groups, "cls")
        wga = runs[rid]["val_wga_last"]
        rows.append({"method": m, "seed": seed, "run": rid, "val_wga": wga, "val_acc": runs[rid]["val_acc_last"],
                     "grp_wga": round(gw, 2), "grp_acc": round(ga, 2), "cls_wga": round(cw, 2),
                     "cls_acc": round(ca, 2), "grp_gain": round(gw - wga, 2), "cls_gain": round(cw - wga, 2)})

    gen = numpy.random.default_rng(20261110)
    wga = numpy.array([r["val_wga"] for r in rows], dtype=float)
    gwa = numpy.array([r["grp_wga"] for r in rows], dtype=float)
    nomk = numpy.array([r["method"] != "maskedkd" for r in rows])
    rho = {"grp_wga_vs_val_wga": _rho_ci(gwa, wga, gen),
           "grp_wga_vs_val_wga_without_maskedkd": _rho_ci(gwa[nomk], wga[nomk], gen)}
    keys = ("val_wga", "val_acc", "grp_wga", "grp_acc", "cls_wga", "cls_acc", "grp_gain", "cls_gain")
    by_method = {}
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        if rs:
            by_method[m] = {k: round(float(numpy.mean([r[k] for r in rs])), 2) for k in keys}
            by_method[m]["n_seeds"] = len(rs)
    spread = {k: round(float(numpy.std([by_method[m][k] for m in by_method], ddof=1)), 3)
              for k in ("val_wga", "grp_wga", "cls_wga")}
    top_gain = max(by_method, key=lambda m: by_method[m]["grp_gain"])
    out = {"lam": LAM, "n_val": int(len(y)), "n_runs": len(rows), "group_counts":
           torch.bincount(groups, minlength=4).tolist(), "teacher_refit_wga_acc": teacher,
           "spread_sd_over_methods": spread, "rho": rho, "by_method": by_method, "runs": rows,
           "verdict": {"P1_grp_spread_le_half_val_spread": bool(spread["grp_wga"] <= 0.5 * spread["val_wga"]),
                       "P2_maskedkd_largest_grp_gain": bool(top_gain == "maskedkd"),
                       "R_largest_grp_gain_method": top_gain,
                       "R_cls_spread_le_half_val_spread": bool(spread["cls_wga"] <= 0.5 * spread["val_wga"])}}
    api.write_json("summary", out)
