"""H-A45 (round 292, analysis only, no training): on Waterbirds K=29, is the WGA gap between methods carried by the
trained student's own feature (how much background/place it encodes inside a class), rather than by its attention?

H-A43 found that the quality of a student's own last-block attention as a teacher input ranks runs like WGA only
because of maskedkd (rho without maskedkd 0.23, uncertain). H-A44 found the KD pull on minority groups ranks runs in
the opposite direction. The remaining place is the student's representation. Here every final student (maskedkd,
maskedkd_frozen, rollout, teacher_topk, trustmask_v1, trustfilt @29 and ce @0; seeds 1, 2, 3) gets the same linear
probe as H-W7 (analysis/shortcut_probe.py): inside each class, a class-balanced closed-form ridge probe (lam 100,
2-fold CV by index parity) predicts place from the student's pre-logit feature (un-augmented val, full image).
place_bacc = balanced accuracy averaged over the two classes. The full teacher's feature is the reference.

Per run: place_bacc. Spearman rho over runs against run val WGA, run bootstrap 95% CI (2000 draws, numpy
default_rng 20261109); the same without maskedkd. Per method: seed means.
No labels, groups or positions enter any selection; place labels are used only by the offline probe.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29, across methods, does the place (background) information a trained student's "
                "feature carries inside each class rank the runs inversely to their student WGA?",
    "method": "linear probe (class-balanced closed-form ridge, 2-fold CV) on the pre-logit feature, per class; "
              "Spearman rank correlation with run bootstrap",
    "paper": "Alain & Bengio ICLR-W 2017 (linear probes); Kirichenko et al. ICLR 2023 (DFR, spurious feature "
             "decodability); Sagawa et al. ICLR 2020 (Waterbirds groups)",
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


def _probe_bacc(F, t, lam=LAM):
    """2-fold CV (index parity) class-balanced ridge probe (closed form, targets +-1);
    balanced accuracy over the two values of t. Same as analysis/shortcut_probe.py."""
    n = F.shape[0]
    folds = torch.arange(n, device=F.device) % 2
    correct = torch.zeros(2, device=F.device)
    count = torch.zeros(2, device=F.device)
    for f in (0, 1):
        tr, te = folds != f, folds == f
        mu, sd = F[tr].mean(0), F[tr].std(0).clamp(min=1e-6)
        Xtr, Xte = (F[tr] - mu) / sd, (F[te] - mu) / sd
        ttr = t[tr].double() * 2 - 1
        sw = torch.where(ttr > 0, 1.0 / (ttr > 0).sum().clamp(min=1), 1.0 / (ttr < 0).sum().clamp(min=1))
        X = torch.cat([Xtr.double(), torch.ones(Xtr.shape[0], 1, device=F.device, dtype=torch.float64)], 1)
        A = X.T @ (X * sw[:, None]) + lam / Xtr.shape[0] * torch.eye(X.shape[1], device=F.device,
                                                                     dtype=torch.float64)
        wb = torch.linalg.solve(A, X.T @ (sw * ttr))
        w, b = wb[:-1].float(), wb[-1].float()
        pred = ((Xte @ w + b) > 0).long()
        for v in (0, 1):
            m = t[te] == v
            correct[v] += (pred[m] == v).sum()
            count[v] += m.sum()
    return (correct / count.clamp(min=1)).mean().item()


def _place_bacc(F, y, place):
    return float(numpy.mean([_probe_bacc(F[y == c], place[y == c]) for c in (0, 1)]))


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
    groups = api.val_groups().to(y.device)
    place = groups - 2 * y

    T = api.teacher.eval()
    teacher_bacc = round(_place_bacc(_features(T, xs), y, place), 4)

    rows = []
    for m, seed, rid in studs:
        S = api.student(rid)
        rows.append({"method": m, "seed": seed, "run": rid, "val_wga": runs[rid]["val_wga_last"],
                     "val_acc": runs[rid]["val_acc_last"],
                     "place_bacc": round(_place_bacc(_features(S, xs), y, place), 4)})

    gen = numpy.random.default_rng(20261109)
    wga = numpy.array([r["val_wga"] for r in rows], dtype=float)
    pb = numpy.array([r["place_bacc"] for r in rows], dtype=float)
    nomk = numpy.array([r["method"] != "maskedkd" for r in rows])
    rho = {"place_bacc_vs_wga": _rho_ci(pb, wga, gen),
           "place_bacc_vs_wga_without_maskedkd": _rho_ci(pb[nomk], wga[nomk], gen)}
    by_method = {}
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        if rs:
            by_method[m] = {k: round(float(numpy.mean([r[k] for r in rs])), 4)
                            for k in ("val_wga", "val_acc", "place_bacc")}
            by_method[m]["n_seeds"] = len(rs)
    top = max(by_method, key=lambda m: by_method[m]["place_bacc"])
    out = {"lam": LAM, "n_val": int(len(y)), "n_runs": len(rows), "teacher_place_bacc": teacher_bacc,
           "rho": rho, "by_method": by_method, "runs": rows,
           "verdict": {"P1_rho_place_bacc_vs_wga_le_-0.5": bool(rho["place_bacc_vs_wga"][0] <= -0.5),
                       "P2_maskedkd_highest_place_bacc": bool(top == "maskedkd"),
                       "R_highest_place_bacc_method": top,
                       "R_kd_methods_below_ce": {m: bool(by_method[m]["place_bacc"] < by_method["ce"]["place_bacc"])
                                                 for m in by_method if m != "ce"} if "ce" in by_method else {}}}
    api.write_json("summary", out)
