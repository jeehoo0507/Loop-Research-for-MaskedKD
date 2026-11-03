"""H-A52 (round 299, analysis only, no training): on Waterbirds K=29 group 2, is the masked-teacher accuracy drop of
the online MaskedKD kept set caused by the tokens only the online student picks (harmful addition), or by the tokens
only the frozen-trained student picks being missing (lost evidence)?

H-A51: the group-2 online - frozen gap in pt = p_T^mask(y) (-0.16) is not explained by overlap with the teacher map
(mass or set). Remaining candidate: the content of the non-shared tokens. Per seed pair (online s, frozen s):
  O = online set, F = frozen set, I = O & F, D_O = O \\ F, D_F = F \\ O   (|D_O| = |D_F|)
and pt is measured keep-only on each of the five sets. Empty sets -> missing for that image and seed.

Students: Waterbirds final maskedkd@29 and maskedkd_frozen@29, seeds 1, 2, 3, final epoch; mask rule for every student:
its own last-block head-mean CLS attention top-K. Un-augmented val. Per image: seed mean (missing skipped), then image
bootstrap inside a group, 2000 draws, numpy default_rng 20261116.
No labels, groups or positions enter any mask; labels enter only pt, groups only slice results.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29 group 2, is the masked-teacher p(y) drop of online vs frozen-trained MaskedKD kept "
                "sets due to the online-only tokens (harmful addition) or the missing frozen-only tokens?",
    "method": "last-block CLS attention top-K; set decomposition into shared core and set differences; keep-only "
              "masked-teacher p(y) on each part; image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 (keep-only); Sagawa et al. ICLR 2020 "
             "(Waterbirds groups)",
}

K = 29
NBOOT = 2000
BATCH = 50
ON, FR = "maskedkd", "maskedkd_frozen"
GROUPS = (0, 1, 2, 3)
FOCUS = 2
SETS = ("O", "F", "I", "DO", "DF")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _rank(v):
    r = numpy.empty(len(v))
    r[numpy.argsort(v, kind="mergesort")] = numpy.arange(len(v))
    return r


def _spear(a, b):
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / numpy.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def _ci(v, gen):
    v = v[~numpy.isnan(v)]
    n = len(v)
    if n == 0:
        return [None, None, None, 0]
    bs = numpy.array([v[gen.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return [round(float(v.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4), int(n)]


def _pt(api, T, x, y, keep):
    """keep-only masked-teacher p(y); rows with an empty set get one dummy token and are returned as nan."""
    empty = keep.sum(1) == 0
    k = keep.clone()
    k[empty, 0] = True
    with torch.no_grad():
        p = api.teacher_forward_keepmask(T, x, k).float().softmax(-1).gather(1, y[:, None])[:, 0]
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

    per = {}
    nd = {}
    for s in seeds:
        ko = _keeps(api, api.student(studs[ON][s]), batches)
        kf = _keeps(api, api.student(studs[FR][s]), batches)
        acc = {z: [] for z in SETS}
        dsz = []
        for (x, y), o, f in zip(batches, ko, kf):
            parts = {"O": o, "F": f, "I": o & f, "DO": o & ~f, "DF": f & ~o}
            for z in SETS:
                acc[z].append(_pt(api, T, x, y, parts[z]))
            dsz.append((o & ~f).sum(1).float().cpu())
        per[s] = {z: torch.cat(acc[z]).numpy() for z in SETS}
        nd[s] = torch.cat(dsz).numpy()

    gen = numpy.random.default_rng(20261116)
    stack = {z: numpy.stack([per[s][z] for s in seeds]) for z in SETS}
    with numpy.errstate(all="ignore"):
        m = {z: numpy.nanmean(stack[z], 0) for z in SETS}
        contrasts = {
            "DO_minus_DF": numpy.nanmean(stack["DO"] - stack["DF"], 0),
            "O_minus_I": numpy.nanmean(stack["O"] - stack["I"], 0),
            "F_minus_I": numpy.nanmean(stack["F"] - stack["I"], 0),
            "O_minus_F": numpy.nanmean(stack["O"] - stack["F"], 0),
        }
    dmean = numpy.mean(numpy.stack([nd[s] for s in seeds]), 0)

    out = {"k": K, "n_val": int(len(groups)), "seeds": seeds,
           "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": {mm: {str(s): r for s, r in v.items()} for mm, v in studs.items()}}
    out["pt_by_set"] = {z: {str(g): _ci(m[z][groups == g], gen) for g in GROUPS} for z in SETS}
    out["contrasts"] = {c: {str(g): _ci(v[groups == g], gen) for g in GROUPS} for c, v in contrasts.items()}
    out["mean_diff_size"] = {str(g): round(float(dmean[groups == g].mean()), 3) for g in GROUPS}
    out["seedwise_group2"] = {z: [round(float(numpy.nanmean(per[s][z][groups == FOCUS])), 4) for s in seeds]
                              for z in SETS}

    sel = (groups == FOCUS) & ~numpy.isnan(contrasts["DO_minus_DF"]) & ~numpy.isnan(contrasts["O_minus_F"])
    a, b = contrasts["O_minus_F"][sel], contrasts["DO_minus_DF"][sel]
    n = len(a)
    bs = []
    for _ in range(NBOOT):
        idx = gen.integers(0, n, n)
        bs.append(_spear(a[idx], b[idx]))
    bs = numpy.array(bs)
    out["group2_spearman_OmF_vs_DOmDF"] = [round(_spear(a, b), 4), round(float(numpy.quantile(bs, 0.025)), 4),
                                           round(float(numpy.quantile(bs, 0.975)), 4), int(n)]

    c = out["contrasts"]
    out["verdict"] = {
        "P1_group2_DO_minus_DF_upper_lt0": bool(c["DO_minus_DF"][str(FOCUS)][2] is not None
                                                and c["DO_minus_DF"][str(FOCUS)][2] < 0),
        "P2_group2_O_minus_I_upper_lt0": bool(c["O_minus_I"][str(FOCUS)][2] is not None
                                              and c["O_minus_I"][str(FOCUS)][2] < 0),
        "R_group2_F_minus_I": c["F_minus_I"][str(FOCUS)],
        "R_group2_O_minus_F": c["O_minus_F"][str(FOCUS)],
    }
    api.write_json("summary", out)
