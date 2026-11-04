"""H-A73 (round 362, analysis only, no training): on cub_lt K=29 (gate 2), is the tail-accuracy gap between methods
in the student's feature rather than in its head?

H-A72 found one scalar tail offset on the original head shrinks the sd of method-mean tail accuracy only from 6.42 to
4.07 (ranking kept, rho 0.91): the gap is not a head/tail offset. Two places remain: the head's per-class directions,
or the feature itself. A head-free, parameter-free classifier on the pre-logit feature separates the two: if the
method gap is in the feature, it should survive; if it is in the head, it should shrink.

Every final student (maskedkd, maskedkd_frozen, rollout, teacher_topk, trustmask_v1, trustfilt, tome, lateroll @29;
full @196; ce @0; seeds 1, 2, 3; same 30 runs as H-A72): pre-logit feature on the un-augmented val image (full image).
Leave-one-out mean-cosine classifier on val: L2-normalise features; score of image i for class c = mean cosine of i
to the val images of class c, excluding i itself; predict argmax. Images whose class has no other val image are
skipped (count recorded). orig = argmax of the original head on the same evaluable images.
Per run: ncm tail / head / mean accuracy, orig tail / head / mean on the same images. Per method: seed means.
Spread = sd over the 10 method means. Spearman rho over runs of ncm tail vs orig tail, run bootstrap 95% CI (2000
draws, numpy default_rng 20261113). Paired same-seed t 95% CI (df 2): v1 - maskedkd, v0 - maskedkd, full - ce.
Only labels enter the classifier; no selection uses anything else.
"""
import numpy
import torch

META = {
    "question": "on cub_lt K=29, does the tail-accuracy gap between methods survive a head-free leave-one-out "
                "mean-cosine classifier on the students' pre-logit features (feature) or shrink (head)?",
    "method": "leave-one-out nearest-class-mean (mean cosine) classifier on val pre-logit features; spread of method "
              "means; Spearman rank correlation with run bootstrap; paired t CI",
    "paper": "Mensink et al. TPAMI 2013 (nearest class mean); Kang et al. ICLR 2020 (decoupling representation and "
             "classifier); Son et al. ECCV 2024 (MaskedKD)",
}

NBOOT = 2000
BATCH = 50
TAIL0 = 100
METHODS = {"maskedkd": 29, "maskedkd_frozen": 29, "rollout": 29, "teacher_topk": 29, "trustmask_v1": 29,
           "trustfilt": 29, "tome": 29, "lateroll": 29, "full": 196, "ce": 0}


def _feat_logits(M, xs):
    fs, zs = [], []
    for x in xs:
        with torch.no_grad():
            h = M.patch_embed(x)
            h = M._pos_embed(h)
            h = M.patch_drop(h)
            h = M.norm_pre(h)
            h = M.blocks(h)
            h = M.norm(h)
            fs.append(M.forward_head(h, pre_logits=True).float())
            zs.append(M.forward_head(h).float())
    return torch.cat(fs), torch.cat(zs)


def _ncm_loo(F, y, ncls):
    """Leave-one-out mean-cosine classifier. Returns predictions and a mask of evaluable images."""
    Fn = torch.nn.functional.normalize(F, dim=1)
    S = Fn @ Fn.T
    Y = torch.nn.functional.one_hot(y, ncls).float()
    sums = S @ Y
    cnt = Y.sum(0).unsqueeze(0).repeat(len(y), 1)
    idx = torch.arange(len(y), device=y.device)
    sums[idx, y] -= S[idx, idx]
    cnt[idx, y] -= 1
    score = torch.where(cnt > 0, sums / cnt.clamp(min=1), torch.full_like(sums, -1e9))
    ok_mask = cnt[idx, y] > 0
    return score.argmax(1), ok_mask


def _accs(pred, y, m):
    ok = (pred == y).float()
    tail = y >= TAIL0
    return (100.0 * ok[m & tail].mean().item(), 100.0 * ok[m & ~tail].mean().item(), 100.0 * ok[m].mean().item())


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
    ncls = int(y.max().item()) + 1

    rows = []
    n_eval = None
    for m, seed, rid in studs:
        F, z = _feat_logits(api.student(rid), xs)
        F, z = F.to(y.device), z.to(y.device)
        pred, mask = _ncm_loo(F, y, ncls)
        n_eval = int(mask.sum().item())
        nt, nh, na = _accs(pred, y, mask)
        ot, oh, oa = _accs(z.argmax(1), y, mask)
        rows.append({"method": m, "seed": seed, "run": rid, "val_wga": runs[rid].get("val_wga_last"),
                     "ncm_tail": round(nt, 2), "ncm_head": round(nh, 2), "ncm_acc": round(na, 2),
                     "orig_tail": round(ot, 2), "orig_head": round(oh, 2), "orig_acc": round(oa, 2)})

    gen = numpy.random.default_rng(20261113)
    ot = numpy.array([r["orig_tail"] for r in rows], dtype=float)
    nt = numpy.array([r["ncm_tail"] for r in rows], dtype=float)
    rho = {"ncm_tail_vs_orig_tail": _rho_ci(nt, ot, gen)}
    keys = ("ncm_tail", "ncm_head", "ncm_acc", "orig_tail", "orig_head", "orig_acc")
    by_method = {}
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        if rs:
            by_method[m] = {k: round(float(numpy.mean([r[k] for r in rs])), 3) for k in keys}
            by_method[m]["n_seeds"] = len(rs)
    spread = {k: round(float(numpy.std([by_method[m][k] for m in by_method], ddof=1)), 3)
              for k in ("orig_tail", "ncm_tail", "orig_acc", "ncm_acc")}
    pair = {}
    for a, c in (("trustmask_v1", "maskedkd"), ("trustfilt", "maskedkd"), ("full", "ce")):
        for k in ("orig_tail", "ncm_tail"):
            d = [next(r[k] for r in rows if r["method"] == a and r["seed"] == s)
                 - next(r[k] for r in rows if r["method"] == c and r["seed"] == s)
                 for s in (1, 2, 3)
                 if any(r["method"] == a and r["seed"] == s for r in rows)
                 and any(r["method"] == c and r["seed"] == s for r in rows)]
            if len(d) >= 2:
                mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
                tq = {2: 12.706, 3: 4.303}[len(d)]
                pair[f"{a}-{c}:{k}"] = [round(mu, 2), round(mu - tq * se, 2), round(mu + tq * se, 2), len(d)]
    p2 = pair.get("trustmask_v1-maskedkd:ncm_tail")
    out = {"n_val": int(len(y)), "n_eval": n_eval, "n_tail_val": int((y >= TAIL0).sum().item()), "n_runs": len(rows),
           "spread_sd_over_methods": spread, "rho": rho, "paired_t95": pair, "by_method": by_method, "runs": rows,
           "verdict": {"P1_ncm_spread_ge_half_orig_spread": bool(spread["ncm_tail"] >= 0.5 * spread["orig_tail"]),
                       "P2_v1_minus_maskedkd_ncm_tail_ci_low_gt_0": bool(p2 is not None and p2[1] > 0)}}
    api.write_json("summary", out)
