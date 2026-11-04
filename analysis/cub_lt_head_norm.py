"""H-A74 (round 363, analysis only, no training): on cub_lt K=29 (gate 2), is the head-direction gap between methods
(H-A73) a per-class weight-norm gap?

H-A73 found a head-free leave-one-out mean-cosine classifier on the feature removes most of the method spread in tail
accuracy (sd 6.42 -> 3.04): the gap lives in the head. Long-tail work (Kang et al. ICLR 2020) shows the head of a
model trained on long-tailed data has larger weight norms for head classes, and that rescaling each class row to
unit norm (tau-normalisation, tau = 1, no fitting) recovers tail accuracy. If the method gap is that norm gap, the
same parameter-free rescaling should remove it.

Same 30 runs as H-A72/H-A73 (cub_lt final seeds 1, 2, 3 of maskedkd, maskedkd_frozen, rollout, teacher_topk,
trustmask_v1, trustfilt, tome, lateroll @29; full @196; ce @0). Un-augmented val (600 images).
Per run, from the student's own head W (200 x d), b:
  orig = argmax(W f + b); taun = argmax((W / ||W_c||) f) (bias dropped, nothing fitted);
  norm_ratio = mean ||W_c|| over tail classes (100-199) / mean over head classes (0-99);
  bias_gap = mean b over tail - mean over head;
  align_tail / align_head = mean over classes of cos(W_c, mu_c - mu), mu_c = val mean feature of class c,
  mu = val mean feature (labels only).
Spread = sd over the 10 method means. Spearman rho over runs (norm_ratio vs orig tail), run bootstrap 95% CI (2000
draws, numpy default_rng 20261114). Paired same-seed t 95% CI (df 2): v1 - maskedkd, v0 - maskedkd, full - ce.
"""
import numpy
import torch

META = {
    "question": "on cub_lt K=29, does unit-norm rescaling of each class row of the student head (tau-normalisation, "
                "no fitting) remove the tail-accuracy gap between methods, i.e. is the head gap a weight-norm gap?",
    "method": "tau-normalised classifier (tau=1, bias dropped); per-class weight norm ratio tail/head; cosine of "
              "class weight to centred val class-mean feature; spread of method means; Spearman with run "
              "bootstrap; paired t CI",
    "paper": "Kang et al. ICLR 2020 (decoupling representation and classifier, tau-normalisation); "
             "Son et al. ECCV 2024 (MaskedKD)",
}

NBOOT = 2000
BATCH = 50
TAIL0 = 100
METHODS = {"maskedkd": 29, "maskedkd_frozen": 29, "rollout": 29, "teacher_topk": 29, "trustmask_v1": 29,
           "trustfilt": 29, "tome": 29, "lateroll": 29, "full": 196, "ce": 0}


def _feats(M, xs):
    fs = []
    for x in xs:
        with torch.no_grad():
            h = M.patch_embed(x)
            h = M._pos_embed(h)
            h = M.patch_drop(h)
            h = M.norm_pre(h)
            h = M.blocks(h)
            h = M.norm(h)
            fs.append(M.forward_head(h, pre_logits=True).float())
    return torch.cat(fs)


def _accs(pred, y):
    ok = (pred == y).float()
    tail = y >= TAIL0
    return 100.0 * ok[tail].mean().item(), 100.0 * ok[~tail].mean().item(), 100.0 * ok.mean().item()


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
    Y = torch.nn.functional.one_hot(y, ncls).float()
    tail_c = torch.arange(ncls, device=y.device) >= TAIL0

    rows = []
    for m, seed, rid in studs:
        M = api.student(rid)
        F = _feats(M, xs).to(y.device)
        W = M.head.weight.detach().float().to(y.device)
        b = M.head.bias.detach().float().to(y.device)
        z = F @ W.T + b
        wn = W.norm(dim=1)
        zt = F @ (W / wn.unsqueeze(1)).T
        ot, oh, oa = _accs(z.argmax(1), y)
        tt, th, ta = _accs(zt.argmax(1), y)
        mu_c = (Y.T @ F) / Y.sum(0).clamp(min=1).unsqueeze(1)
        cs = torch.nn.functional.cosine_similarity(W, mu_c - F.mean(0, keepdim=True), dim=1)
        rows.append({"method": m, "seed": seed, "run": rid, "val_wga": runs[rid].get("val_wga_last"),
                     "orig_tail": round(ot, 2), "orig_head": round(oh, 2), "orig_acc": round(oa, 2),
                     "taun_tail": round(tt, 2), "taun_head": round(th, 2), "taun_acc": round(ta, 2),
                     "norm_ratio": round((wn[tail_c].mean() / wn[~tail_c].mean()).item(), 4),
                     "bias_gap": round((b[tail_c].mean() - b[~tail_c].mean()).item(), 4),
                     "align_tail": round(cs[tail_c].mean().item(), 4),
                     "align_head": round(cs[~tail_c].mean().item(), 4)})

    gen = numpy.random.default_rng(20261114)
    ot = numpy.array([r["orig_tail"] for r in rows], dtype=float)
    nr = numpy.array([r["norm_ratio"] for r in rows], dtype=float)
    al = numpy.array([r["align_tail"] for r in rows], dtype=float)
    rho = {"norm_ratio_vs_orig_tail": _rho_ci(nr, ot, gen), "align_tail_vs_orig_tail": _rho_ci(al, ot, gen)}
    keys = ("orig_tail", "orig_head", "orig_acc", "taun_tail", "taun_head", "taun_acc",
            "norm_ratio", "bias_gap", "align_tail", "align_head")
    by_method = {}
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        if rs:
            by_method[m] = {k: round(float(numpy.mean([r[k] for r in rs])), 4) for k in keys}
            by_method[m]["n_seeds"] = len(rs)
    spread = {k: round(float(numpy.std([by_method[m][k] for m in by_method], ddof=1)), 3)
              for k in ("orig_tail", "taun_tail", "orig_acc", "taun_acc")}
    pair = {}
    for a, c in (("trustmask_v1", "maskedkd"), ("trustfilt", "maskedkd"), ("full", "ce")):
        for k in ("orig_tail", "taun_tail", "norm_ratio", "align_tail"):
            d = [next(r[k] for r in rows if r["method"] == a and r["seed"] == s)
                 - next(r[k] for r in rows if r["method"] == c and r["seed"] == s)
                 for s in (1, 2, 3)
                 if any(r["method"] == a and r["seed"] == s for r in rows)
                 and any(r["method"] == c and r["seed"] == s for r in rows)]
            if len(d) >= 2:
                mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
                tq = {2: 12.706, 3: 4.303}[len(d)]
                pair[f"{a}-{c}:{k}"] = [round(mu, 4), round(mu - tq * se, 4), round(mu + tq * se, 4), len(d)]
    out = {"n_val": int(len(y)), "n_tail_val": int((y >= TAIL0).sum().item()), "n_runs": len(rows),
           "spread_sd_over_methods": spread, "rho": rho, "paired_t95": pair, "by_method": by_method, "runs": rows,
           "verdict": {"P1_taun_spread_le_half_orig_spread": bool(spread["taun_tail"] <= 0.5 * spread["orig_tail"]),
                       "P2_rho_norm_ratio_vs_orig_tail_ge_0.5": bool(rho["norm_ratio_vs_orig_tail"][0] >= 0.5)}}
    api.write_json("summary", out)
