"""H-A54 (round 301, analysis only, no training): on Waterbirds K=29, does the majority-class (landbird) pull of the
tokens only the online MaskedKD student picks grow along training (epoch 25 -> 50 -> final)?

H-A52/53: at the final epoch the online-only tokens D_O = O \\ F pull the masked teacher toward landbird in every
group (not toward the student's own prediction). Gate 3 calls online - frozen a feedback loop; a loop predicts the
pull is built up by training, not present from early on. Per seed pair (online s, frozen-trained s) and epoch e,
both students at the same epoch e:
  O = online set, F = frozen-trained set, D_O = O \\ F, D_F = F \\ O   (|D_O| = |D_F|)
  land = p_T(D_O)(landbird) - p_T(D_F)(landbird)   (no label)
  dpt  = p_T(D_O)(y) - p_T(D_F)(y)   (2 classes: in waterbird groups dpt = -land, recorded for H-A52 continuity)
Verdicts are per group (val is 78% landbird; pooled values are records only).
keep-only, T 1. Empty sets -> missing for that image, seed and epoch.

Students: Waterbirds final maskedkd@29 and maskedkd_frozen@29, seeds 1, 2, 3, snapshots e25, e50 and final; mask rule
for every student: its own last-block head-mean CLS attention top-K. Un-augmented val. Per image: seed mean (missing
skipped), then image bootstrap, 2000 draws, numpy default_rng 20261118. Epoch contrasts are per image (same image).
No labels, groups or positions enter any mask; labels enter only dpt, groups only slice results.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29, does the landbird pull of online-only MaskedKD tokens (vs frozen-trained-only "
                "tokens) grow from epoch 25 to the final epoch?",
    "method": "last-block CLS attention top-K at epoch snapshots; set differences; keep-only masked-teacher class "
              "probabilities; per-image epoch contrasts; image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 (keep-only); Sagawa et al. ICLR 2020 "
             "(Waterbirds groups)",
}

K = 29
NBOOT = 2000
BATCH = 50
ON, FR = "maskedkd", "maskedkd_frozen"
GROUPS = (0, 1, 2, 3)
FOCUS = 2
LAND = 0
EPOCHS = (25, 50, "final")


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


def _keeps(api, S, batches):
    last = len(S.blocks) - 1
    out = []
    for x, _ in batches:
        with torch.no_grad():
            so = api.student_forward(S, x, attn_layers=[last])
        out.append(_topk(so.attn[last].float().mean(1)[:, 0, 1:], K))
    return out


def run(api):
    runs = api.runs()
    studs = {ON: {}, FR: {}}
    for rid, r in sorted(runs.items()):
        if (r.get("dataset") == "waterbirds" and r["tier"] == "final" and r["method"] in studs
                and r.get("keep") == K and r["seed"] in (1, 2, 3)):
            studs[r["method"]][r["seed"]] = rid
    seeds = sorted(set(studs[ON]) & set(studs[FR]))
    T = api.teacher.eval()
    batches = [(x, y) for x, y in api.val_batches(batch=BATCH)]
    groups = api.val_groups().numpy()

    land = {e: [] for e in EPOCHS}
    dpt = {e: [] for e in EPOCHS}
    dsz = {e: [] for e in EPOCHS}
    for e in EPOCHS:
        ep = None if e == "final" else e
        for s in seeds:
            ko = _keeps(api, api.student(studs[ON][s], epoch=ep), batches)
            kf = _keeps(api, api.student(studs[FR][s], epoch=ep), batches)
            la, dp, sz = [], [], []
            for (x, y), o, f in zip(batches, ko, kf):
                po = _probs(api, T, x, o & ~f)
                pf = _probs(api, T, x, f & ~o)
                yc = y.cpu()[:, None]
                la.append(po[:, LAND] - pf[:, LAND])
                dp.append(po.gather(1, yc)[:, 0] - pf.gather(1, yc)[:, 0])
                sz.append((o & ~f).sum(1).float().cpu())
            land[e].append(torch.cat(la).numpy())
            dpt[e].append(torch.cat(dp).numpy())
            dsz[e].append(torch.cat(sz).numpy())

    gen = numpy.random.default_rng(20261118)
    st = {"land": {e: numpy.stack(land[e]) for e in EPOCHS}, "dpt": {e: numpy.stack(dpt[e]) for e in EPOCHS}}
    with numpy.errstate(all="ignore"):
        m = {q: {e: numpy.nanmean(st[q][e], 0) for e in EPOCHS} for q in st}
        growth = {q: {f"final_minus_e{a}": numpy.nanmean(st[q]["final"] - st[q][a], 0) for a in (25, 50)}
                  for q in st}

    out = {"k": K, "n_val": int(len(groups)), "seeds": seeds,
           "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": {mm: {str(s): r for s, r in v.items()} for mm, v in studs.items()}}
    out["by_epoch"] = {q: {str(e): {"all": _ci(m[q][e], gen),
                                    **{str(g): _ci(m[q][e][groups == g], gen) for g in GROUPS}}
                           for e in EPOCHS} for q in m}
    out["growth"] = {q: {c: {"all": _ci(v, gen), **{str(g): _ci(v[groups == g], gen) for g in GROUPS}}
                         for c, v in d.items()} for q, d in growth.items()}
    out["mean_diff_size"] = {str(e): {str(g): round(float(numpy.mean(numpy.stack(dsz[e]), 0)[groups == g].mean()), 3)
                                      for g in GROUPS} for e in EPOCHS}
    out["seedwise_all_land"] = {str(e): [round(float(numpy.nanmean(v)), 4) for v in land[e]] for e in EPOCHS}

    g = out["growth"]
    grown = [gg for gg in GROUPS if g["land"]["final_minus_e25"][str(gg)][1] is not None
             and g["land"]["final_minus_e25"][str(gg)][1] > 0]
    e25 = out["by_epoch"]["land"]["25"][str(FOCUS)]
    out["verdict"] = {
        "P1_groups_land_growth_lower_gt0": grown,
        "P1_at_least_3_of_4": bool(len(grown) >= 3),
        "P2_group2_land_e25_lower_gt0": bool(e25[1] is not None and e25[1] > 0),
        "R_pooled_land_growth": g["land"]["final_minus_e25"]["all"],
        "R_group2_dpt_growth": g["dpt"]["final_minus_e25"][str(FOCUS)],
    }
    api.write_json("summary", out)
