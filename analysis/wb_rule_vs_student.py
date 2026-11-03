"""H-A55 (round 302, analysis only, no training): on Waterbirds K=29, is the landbird (majority-class) pull of
MaskedKD-picked tokens a property of the token rule (last-block CLS attention) or of the MaskedKD-trained student?

H-A52..54: online MaskedKD-only tokens pull the masked teacher toward landbird, already at epoch 25. H-A50: gate 1's
trustmask_v1 > maskedkd looked like the token rule's share (blocks 6..11 rollout), not the KD weight. 2x2 split,
inside the SAME student:
  L = last-block head-mean CLS attention top-K   (maskedkd rule)
  R = blocks 6..11 rollout CLS row top-K          (trustmask_v1 rule: 0.5A + 0.5I, row-normalised, product)
  D_L = L \\ R, D_R = R \\ L   (|D_L| = |D_R|)
  land = p_T(D_L)(landbird) - p_T(D_R)(landbird)   (no label)
  pt   = p_T(L)(y) - p_T(R)(y)                      (record)
repeated on maskedkd-trained and trustmask_v1-trained students.
Verdict on group 2 at epoch 25 (final recorded). keep-only, T 1. Empty sets -> missing for that image and seed.

Students: Waterbirds final maskedkd@29 and trustmask_v1@29, seeds 1, 2, 3, snapshots e25 and final. Un-augmented
val. Per image: seed mean (missing skipped), then image bootstrap, 2000 draws, numpy default_rng 20261119.
No labels, groups or positions enter any mask; labels enter only pt, groups only slice results.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29, inside the same student, do last-block CLS attention picks (vs blocks 6..11 "
                "rollout picks) pull the masked teacher toward landbird, in maskedkd- and trustmask_v1-trained "
                "students alike?",
    "method": "same-student 2x2 of token rule x training method; set differences; keep-only masked-teacher class "
              "probabilities; image bootstrap",
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
    last = layers[-1]
    lk = _topk(so.attn[last].float().mean(1)[:, 0, 1:], K)
    r = None
    for i in layers:
        a = so.attn[i].float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        r = a if r is None else a @ r
    rk = _topk(r[:, 0, 1:], K)
    return lk, rk


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

    land = {m: {e: [] for e in EPOCHS} for m in METHODS}
    pt = {m: {e: [] for e in EPOCHS} for m in METHODS}
    dsz = {m: {e: [] for e in EPOCHS} for m in METHODS}
    for m in METHODS:
        for e in EPOCHS:
            ep = None if e == "final" else e
            for s in sorted(studs[m]):
                S = api.student(studs[m][s], epoch=ep)
                la, pp, sz = [], [], []
                for x, y in batches:
                    lk, rk = _rules(api, S, x)
                    pl = _probs(api, T, x, lk & ~rk)
                    pr = _probs(api, T, x, rk & ~lk)
                    la.append(pl[:, LAND] - pr[:, LAND])
                    yc = y.cpu()[:, None]
                    fl = _probs(api, T, x, lk).gather(1, yc)[:, 0]
                    fr = _probs(api, T, x, rk).gather(1, yc)[:, 0]
                    pp.append(fl - fr)
                    sz.append((lk & ~rk).sum(1).float().cpu())
                land[m][e].append(torch.cat(la).numpy())
                pt[m][e].append(torch.cat(pp).numpy())
                dsz[m][e].append(torch.cat(sz).numpy())

    gen = numpy.random.default_rng(20261119)
    with numpy.errstate(all="ignore"):
        mland = {m: {e: numpy.nanmean(numpy.stack(land[m][e]), 0) for e in EPOCHS} for m in METHODS}
        mpt = {m: {e: numpy.nanmean(numpy.stack(pt[m][e]), 0) for e in EPOCHS} for m in METHODS}

    def _slice(v):
        return {"all": _ci(v, gen), **{str(g): _ci(v[groups == g], gen) for g in GROUPS}}

    out = {"k": K, "start": START, "n_val": int(len(groups)),
           "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": {m: {str(s): r for s, r in v.items()} for m, v in studs.items()}}
    out["land"] = {m: {str(e): _slice(mland[m][e]) for e in EPOCHS} for m in METHODS}
    out["pt_L_minus_R"] = {m: {str(e): _slice(mpt[m][e]) for e in EPOCHS} for m in METHODS}
    out["land_maskedkd_minus_v1"] = {str(e): _slice(mland["maskedkd"][e] - mland["trustmask_v1"][e])
                                     for e in EPOCHS}
    out["mean_diff_size"] = {m: {str(e): {str(g): round(float(numpy.mean(numpy.stack(dsz[m][e]), 0)[groups == g]
                                                             .mean()), 3) for g in GROUPS}
                                      for e in EPOCHS} for m in METHODS}
    out["seedwise_group2_land_e25"] = {m: [round(float(numpy.nanmean(v[groups == FOCUS])), 4)
                                           for v in land[m][25]] for m in METHODS}

    a = out["land"]["maskedkd"]["25"][str(FOCUS)]
    b = out["land"]["trustmask_v1"]["25"][str(FOCUS)]
    out["verdict"] = {
        "P1_maskedkd_e25_group2_land_lower_gt0": bool(a[1] is not None and a[1] > 0),
        "P2_v1_e25_group2_land_lower_gt0": bool(b[1] is not None and b[1] > 0),
        "R_maskedkd_e25_group2": a,
        "R_v1_e25_group2": b,
    }
    api.write_json("summary", out)
