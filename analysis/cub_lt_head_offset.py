"""H-A72 (round 361, analysis only, no training): on cub_lt K=29 (gate 2), is the tail-accuracy gap between methods
only a tail-vs-head logit offset of the original head? (cub_lt analogue of Waterbirds H-A47.)

H-A47 (Waterbirds): shifting only the threshold of the original head's logit difference (class-balanced, 2-fold CV)
shrank the sd of method-mean WGA from 10.86 to 1.64; the gap was a logit offset. Gate 2 (H-A39) on cub_lt: v1 -
maskedkd tail +6.33 [+1.72, +10.94]. If the cub_lt gap is also an offset (the student's prior on head classes), adding
one scalar b to the 100 tail-class logits of the original head should remove most of it.

Every final student (maskedkd, maskedkd_frozen, rollout, teacher_topk, trustmask_v1, trustfilt, tome, lateroll @29;
full @196; ce @0; seeds 1, 2, 3): original head logits z on the un-augmented val image (full image).
orig: argmax z (recomputed tail / head / mean accuracy).  adj: 2-fold CV by index parity; on the train fold pick b
from the grid B_GRID that maximises head/tail-balanced accuracy (mean of head and tail accuracy; tail membership is a
function of the label); ties -> smallest |b|; predict argmax(z + b * tail) on the held-out fold; pool the two folds.
Per run: orig tail, adj tail, adj gain = adj tail - orig tail, b (mean of folds), adj mean accuracy. Per method:
seed means. Spread = sd over the 10 method means. Spearman rho over runs of adj tail vs orig tail, run bootstrap 95%
CI (2000 draws, numpy default_rng 20261112). No selection uses anything but labels.
"""
import numpy
import torch

META = {
    "question": "on cub_lt K=29, does one scalar offset on the tail-class logits of the original head (head/tail "
                "balanced, 2-fold CV) remove the tail-accuracy gap between methods, as a threshold did on Waterbirds?",
    "method": "post-hoc logit adjustment of the original head with one tail offset chosen by 2-fold CV; spread of "
              "method means; Spearman rank correlation with run bootstrap",
    "paper": "Menon et al. ICLR 2021 (logit adjustment); Kang et al. ICLR 2020 (decoupling, tau-normalisation); "
             "Kirichenko et al. ICLR 2023 (DFR); Son et al. ECCV 2024 (MaskedKD)",
}

NBOOT = 2000
BATCH = 50
TAIL0 = 100
B_GRID = [round(-5.0 + 0.05 * i, 2) for i in range(201)]
METHODS = {"maskedkd": 29, "maskedkd_frozen": 29, "rollout": 29, "teacher_topk": 29, "trustmask_v1": 29,
           "trustfilt": 29, "tome": 29, "lateroll": 29, "full": 196, "ce": 0}


def _logits(M, xs):
    out = []
    for x in xs:
        with torch.no_grad():
            out.append(M(x).float())
    return torch.cat(out)


def _accs(pred, y):
    ok = (pred == y).float()
    tail = y >= TAIL0
    return 100.0 * ok[tail].mean().item(), 100.0 * ok[~tail].mean().item(), 100.0 * ok.mean().item()


def _best_b(z, y):
    tail = y >= TAIL0
    best, bb = -1.0, 0.0
    for b in sorted(B_GRID, key=abs):
        zz = z.clone()
        zz[:, TAIL0:] += b
        ok = (zz.argmax(1) == y).float()
        bal = 0.5 * (ok[tail].mean().item() + ok[~tail].mean().item())
        if bal > best + 1e-12:
            best, bb = bal, b
    return bb


def _adj(z, y):
    n = z.shape[0]
    folds = torch.arange(n, device=z.device) % 2
    pred = torch.zeros(n, dtype=torch.long, device=z.device)
    bs = []
    for f in (0, 1):
        tr, te = folds != f, folds == f
        b = _best_b(z[tr], y[tr])
        bs.append(b)
        zz = z[te].clone()
        zz[:, TAIL0:] += b
        pred[te] = zz.argmax(1)
    return _accs(pred, y), bs


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
        if (r.get("dataset") == "cub_lt" and r["tier"] == "final" and r["method"] in METHODS
                and r.get("keep") == METHODS[r["method"]] and r["seed"] in (1, 2, 3)):
            studs.append((r["method"], r["seed"], rid))
    api.write_json("students", [[m, s, rid, runs[rid].get("val_wga_last"), runs[rid].get("val_acc_last")]
                                for m, s, rid in studs])
    xs, ys = [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
    y = torch.cat(ys)

    rows = []
    for m, seed, rid in studs:
        z = _logits(api.student(rid), xs).to(y.device)
        ot, oh, oa = _accs(z.argmax(1), y)
        (at, ah, aa), bs = _adj(z, y)
        rows.append({"method": m, "seed": seed, "run": rid, "val_wga": runs[rid].get("val_wga_last"),
                     "orig_tail": round(ot, 2), "orig_head": round(oh, 2), "orig_acc": round(oa, 2),
                     "adj_tail": round(at, 2), "adj_head": round(ah, 2), "adj_acc": round(aa, 2),
                     "adj_gain": round(at - ot, 2), "b": round(float(numpy.mean(bs)), 3), "b_folds": bs})

    gen = numpy.random.default_rng(20261112)
    ot = numpy.array([r["orig_tail"] for r in rows], dtype=float)
    at = numpy.array([r["adj_tail"] for r in rows], dtype=float)
    nomk = numpy.array([r["method"] != "maskedkd" for r in rows])
    rho = {"adj_tail_vs_orig_tail": _rho_ci(at, ot, gen),
           "adj_tail_vs_orig_tail_without_maskedkd": _rho_ci(at[nomk], ot[nomk], gen)}
    keys = ("orig_tail", "orig_head", "orig_acc", "adj_tail", "adj_head", "adj_acc", "adj_gain", "b")
    by_method = {}
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        if rs:
            by_method[m] = {k: round(float(numpy.mean([r[k] for r in rs])), 3) for k in keys}
            by_method[m]["n_seeds"] = len(rs)
    spread = {k: round(float(numpy.std([by_method[m][k] for m in by_method], ddof=1)), 3)
              for k in ("orig_tail", "adj_tail", "orig_acc", "adj_acc")}
    top_gain = max(by_method, key=lambda m: by_method[m]["adj_gain"])
    pair = {}
    for a, c in (("trustmask_v1", "maskedkd"), ("trustfilt", "maskedkd"), ("full", "ce")):
        for k in ("orig_tail", "adj_tail"):
            d = [next(r[k] for r in rows if r["method"] == a and r["seed"] == s)
                 - next(r[k] for r in rows if r["method"] == c and r["seed"] == s)
                 for s in (1, 2, 3)
                 if any(r["method"] == a and r["seed"] == s for r in rows)
                 and any(r["method"] == c and r["seed"] == s for r in rows)]
            if len(d) >= 2:
                mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
                tq = {2: 12.706, 3: 4.303}[len(d)]
                pair[f"{a}-{c}:{k}"] = [round(mu, 2), round(mu - tq * se, 2), round(mu + tq * se, 2), len(d)]
    out = {"n_val": int(len(y)), "n_tail_val": int((y >= TAIL0).sum().item()), "n_runs": len(rows),
           "spread_sd_over_methods": spread, "rho": rho, "paired_t95": pair, "by_method": by_method, "runs": rows,
           "verdict": {"P1_adj_spread_le_half_orig_spread": bool(spread["adj_tail"] <= 0.5 * spread["orig_tail"]),
                       "P2_maskedkd_largest_adj_gain": bool(top_gain == "maskedkd"),
                       "R_largest_adj_gain_method": top_gain}}
    api.write_json("summary", out)
