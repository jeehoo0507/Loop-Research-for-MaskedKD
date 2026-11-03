"""H-A56 (round 303, analysis only, no training): on Waterbirds K=29, is the landbird (majority-class) pull of
single-block CLS attention picks specific to the LAST block, or does it grow steadily with depth?

H-A55: inside the same student, last-block CLS attention top-K (L) pulls the masked teacher toward landbird relative
to blocks 6..11 rollout top-K (R), in maskedkd- and trustmask_v1-trained students alike (group 2). Here, inside the
same student, for every block b in 6..11:
  B_b = block-b head-mean CLS attention top-K
  R   = blocks 6..11 rollout CLS row top-K   (trustmask_v1 rule: 0.5A + 0.5I, row-normalised, product)
  land_b = p_T(B_b \\ R)(landbird) - p_T(R \\ B_b)(landbird)   (no label)
  pt_b   = p_T(B_b)(y) - p_T(R)(y)                              (record)
Verdict on group 2 at epoch 25 (final recorded):
  last_vs_rest = land_11 - mean(land_6..10)   (per image)
keep-only, T 1. Empty sets -> missing for that image, block and seed.

Students: Waterbirds final maskedkd@29 and trustmask_v1@29, seeds 1, 2, 3, snapshots e25 and final. Un-augmented
val. Per image: seed mean (missing skipped), then image bootstrap, 2000 draws, numpy default_rng 20261120.
No labels, groups or positions enter any mask; labels enter only pt, groups only slice results.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29, inside the same student, is the landbird pull of single-block CLS attention "
                "picks (vs blocks 6..11 rollout picks) specific to the last block or graded with depth?",
    "method": "same-student depth profile of single-block CLS attention top-K vs rollout top-K; set differences; "
              "keep-only masked-teacher class probabilities; image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (attention rollout); Hooker et al. NeurIPS "
             "2019 (keep-only); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

K = 29
START = 6
NBOOT = 2000
BATCH = 50
METHODS = ("maskedkd", "trustmask_v1")
GROUPS = (0, 1, 2, 3)
FOCUS = 2
LAND = 0
EPOCHS = (25, "final")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _ci(v, gen):
    v = v[~numpy.isnan(v)]
    n = len(v)
    if n == 0:
        return [None, None, None, 0]
    bs = numpy.array([v[gen.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return [round(float(v.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4), int(n)]


def _probs(api, T, x, keep):
    """keep-only masked-teacher softmax; rows with an empty set get one dummy token and are returned as nan."""
    empty = keep.sum(1) == 0
    k = keep.clone()
    k[empty, 0] = True
    with torch.no_grad():
        p = api.teacher_forward_keepmask(T, x, k).float().softmax(-1)
    p = p.float().cpu()
    p[empty.cpu()] = float("nan")
    return p


def _rules(api, S, x):
    layers = list(range(START, len(S.blocks)))
    with torch.no_grad():
        so = api.student_forward(S, x, attn_layers=layers)
    blk = {i: _topk(so.attn[i].float().mean(1)[:, 0, 1:], K) for i in layers}
    r = None
    for i in layers:
        a = so.attn[i].float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        r = a if r is None else a @ r
    rk = _topk(r[:, 0, 1:], K)
    return layers, blk, rk


def run(api):
    runs = api.runs()
    studs = {m: {} for m in METHODS}
    for rid, r in sorted(runs.items()):
        if (r.get("dataset") == "waterbirds" and r["tier"] == "final" and r["method"] in studs
                and r.get("keep") == K and r["seed"] in (1, 2, 3)):
            studs[r["method"]][r["seed"]] = rid
    T = api.teacher.eval()
    batches = [(x, y) for x, y in api.val_batches(batch=BATCH)]
    groups = api.val_groups().numpy()

    layers = None
    # land[m][e][b] -> list over seeds of per-image arrays
    land, pt, dsz = {}, {}, {}
    for m in METHODS:
        land[m], pt[m], dsz[m] = {}, {}, {}
        for e in EPOCHS:
            ep = None if e == "final" else e
            land[m][e], pt[m][e], dsz[m][e] = {}, {}, {}
            for s in sorted(studs[m]):
                S = api.student(studs[m][s], epoch=ep)
                la, pp, sz = {}, {}, {}
                for x, y in batches:
                    layers, blk, rk = _rules(api, S, x)
                    yc = y.cpu()[:, None]
                    fr = _probs(api, T, x, rk).gather(1, yc)[:, 0]
                    for b in layers:
                        bk = blk[b]
                        pb = _probs(api, T, x, bk & ~rk)
                        pr = _probs(api, T, x, rk & ~bk)
                        la.setdefault(b, []).append(pb[:, LAND] - pr[:, LAND])
                        fb = _probs(api, T, x, bk).gather(1, yc)[:, 0]
                        pp.setdefault(b, []).append(fb - fr)
                        sz.setdefault(b, []).append((bk & ~rk).sum(1).float().cpu())
                for b in layers:
                    land[m][e].setdefault(b, []).append(torch.cat(la[b]).numpy())
                    pt[m][e].setdefault(b, []).append(torch.cat(pp[b]).numpy())
                    dsz[m][e].setdefault(b, []).append(torch.cat(sz[b]).numpy())

    gen = numpy.random.default_rng(20261120)
    with numpy.errstate(all="ignore"):
        mland = {m: {e: {b: numpy.nanmean(numpy.stack(land[m][e][b]), 0) for b in layers} for e in EPOCHS}
                 for m in METHODS}
        mpt = {m: {e: {b: numpy.nanmean(numpy.stack(pt[m][e][b]), 0) for b in layers} for e in EPOCHS}
               for m in METHODS}
        last = layers[-1]
        rest = layers[:-1]
        lvr = {m: {e: mland[m][e][last] - numpy.nanmean(numpy.stack([mland[m][e][b] for b in rest]), 0)
                   for e in EPOCHS} for m in METHODS}

    def _slice(v):
        return {"all": _ci(v, gen), **{str(g): _ci(v[groups == g], gen) for g in GROUPS}}

    out = {"k": K, "start": START, "blocks": layers, "n_val": int(len(groups)),
           "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": {m: {str(s): r for s, r in v.items()} for m, v in studs.items()}}
    out["land"] = {m: {str(e): {str(b): _slice(mland[m][e][b]) for b in layers} for e in EPOCHS} for m in METHODS}
    out["pt_B_minus_R"] = {m: {str(e): {str(b): _slice(mpt[m][e][b]) for b in layers} for e in EPOCHS}
                           for m in METHODS}
    out["last_vs_rest"] = {m: {str(e): _slice(lvr[m][e]) for e in EPOCHS} for m in METHODS}
    out["mean_diff_size"] = {m: {str(e): {str(b): {str(g): round(float(numpy.mean(numpy.stack(dsz[m][e][b]), 0)
                                                                       [groups == g].mean()), 3) for g in GROUPS}
                                          for b in layers} for e in EPOCHS} for m in METHODS}
    g2 = {m: [float(numpy.nanmean(mland[m][25][b][groups == FOCUS])) for b in layers] for m in METHODS}
    out["group2_e25_land_by_block"] = {m: [round(v, 4) for v in g2[m]] for m in METHODS}
    out["group2_e25_argmax_block"] = {m: int(layers[int(numpy.argmax(g2[m]))]) for m in METHODS}

    a = out["last_vs_rest"]["maskedkd"]["25"][str(FOCUS)]
    b = out["last_vs_rest"]["trustmask_v1"]["25"][str(FOCUS)]
    out["verdict"] = {
        "P1_maskedkd_e25_group2_last_vs_rest_lower_gt0": bool(a[1] is not None and a[1] > 0),
        "P2_maskedkd_e25_group2_argmax_is_last": bool(out["group2_e25_argmax_block"]["maskedkd"] == last),
        "R_maskedkd_e25_group2_last_vs_rest": a,
        "R_v1_e25_group2_last_vs_rest": b,
    }
    api.write_json("summary", out)
