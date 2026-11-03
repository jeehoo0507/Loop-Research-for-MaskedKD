"""H-A51 (round 298, analysis only, no training): on Waterbirds K=29, which property of the kept token set separates
a correct masked teacher from a wrong one on the minority group where online and frozen MaskedKD differ most?

H-A42 found that on group 2 the online maskedkd@29 students' own last-block CLS-attention top-K is a much worse
teacher input than the frozen-trained students' (masked teacher acc -0.198), yet the teacher-rollout mass inside the
kept set (cov) differs by <= 1.3%p. Mass coverage of a broad all-block rollout map is a weak set measure. Here two
sharper set measures are put next to cov:
  jac   Jaccard of the kept set with the teacher-rollout top-K set (same K)
  disp  spatial dispersion of the kept set: mean distance (patch units) of kept tokens to their centroid
        (translation-invariant, no position prior)
and each is related per image to pt = p_T^mask(y) (keep-only masked teacher, softmax at T=1).

Students: Waterbirds final maskedkd@29 and maskedkd_frozen@29, seeds 1, 2, 3, final epoch. Mask rule for every
student: its own last-block head-mean CLS attention top-K (= what an online MaskedKD selector would pick).
Un-augmented val. Group 2 rows: 6 students x n2 images. Spearman over the pooled rows; image bootstrap (all 6 students'
rows of a drawn image move together), 2000 draws, numpy default_rng 20261115.
No labels, groups or positions enter any mask; labels enter only pt, groups only slice results.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29 group 2, does the Jaccard of a student's attention top-K with the teacher-rollout "
                "top-K explain masked-teacher correctness better than rollout-mass coverage, and is it lower for online "
                "than frozen-trained MaskedKD students?",
    "method": "last-block CLS attention top-K; attention rollout top-K Jaccard; kept-set spatial dispersion; keep-only "
              "masked-teacher p(y); Spearman with image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Hooker et al. NeurIPS 2019 "
             "(keep-only); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

K = 29
NBOOT = 2000
BATCH = 50
METHODS = ("maskedkd", "maskedkd_frozen")
GROUPS = (0, 1, 2, 3)
FOCUS = 2
FEATS = ("cov", "jac", "disp")
STATS = ("pt", "cov", "jac", "disp")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    m = r[:, 0, 1:]
    return m / m.sum(1, keepdim=True)


def _disp(keep):
    n = keep.shape[1]
    g = int(round(n ** 0.5))
    ii = torch.arange(n, device=keep.device)
    rc = torch.stack([(ii // g).float(), (ii % g).float()], 1)
    w = keep.float()
    c = (w @ rc) / w.sum(1, keepdim=True)
    d = ((rc[None] - c[:, None]) ** 2).sum(-1).sqrt()
    return (d * w).sum(1) / w.sum(1)


def _rank(v):
    r = numpy.empty(len(v))
    r[numpy.argsort(v, kind="mergesort")] = numpy.arange(len(v))
    return r


def _spear(a, b):
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / numpy.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def _ci(v, gen):
    n = len(v)
    bs = numpy.array([v[gen.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return [round(float(v.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4), int(n)]


def run(api):
    runs = api.runs()
    studs = {m: {} for m in METHODS}
    for rid, r in sorted(runs.items()):
        if (r.get("dataset") == "waterbirds" and r["tier"] == "final" and r["method"] in METHODS
                and r.get("keep") == K and r["seed"] in (1, 2, 3)):
            studs[r["method"]][r["seed"]] = rid
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        tm = _rollout(out.attn)
        batches.append((x, y, tm, _topk(tm, K)))
    groups = api.val_groups().numpy()

    per = {m: {} for m in METHODS}
    for m in METHODS:
        for seed, rid in sorted(studs[m].items()):
            S = api.student(rid)
            last = len(S.blocks) - 1
            acc = {s: [] for s in STATS}
            for x, y, tm, tk in batches:
                with torch.no_grad():
                    so = api.student_forward(S, x, attn_layers=[last])
                keep = _topk(so.attn[last].float().mean(1)[:, 0, 1:], K)
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float().softmax(-1)
                inter = (keep & tk).sum(1).float()
                acc["pt"].append(lm.gather(1, y[:, None])[:, 0].float().cpu())
                acc["cov"].append((tm * keep).sum(1).float().cpu())
                acc["jac"].append((inter / (2 * K - inter)).cpu())
                acc["disp"].append(_disp(keep).float().cpu())
            per[m][seed] = {s: torch.cat(acc[s]).numpy() for s in STATS}

    gen = numpy.random.default_rng(20261115)
    out = {"k": K, "n_val": int(len(groups)), "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": {m: {str(s): r for s, r in v.items()} for m, v in studs.items()}}
    # group means per method (seeds averaged per image) and online - frozen paired over images
    d = {m: {s: numpy.mean(numpy.stack([per[m][sd][s] for sd in sorted(per[m])]), 0) for s in STATS}
         for m in METHODS}
    on, fr = METHODS
    out["by_method"] = {m: {s: {str(g): _ci(d[m][s][groups == g], gen) for g in GROUPS} for s in STATS}
                        for m in METHODS}
    out["online_minus_frozen"] = {s: {str(g): _ci((d[on][s] - d[fr][s])[groups == g], gen) for g in GROUPS}
                                  for s in STATS}
    out["seedwise_group_mean"] = {m: {s: {str(g): [round(float(per[m][sd][s][groups == g].mean()), 4)
                                                   for sd in sorted(per[m])] for g in GROUPS} for s in STATS}
                                  for m in METHODS}

    # pooled Spearman on group FOCUS rows (6 students x n images), image bootstrap
    sel = groups == FOCUS
    rows = [(m, sd) for m in METHODS for sd in sorted(per[m])]
    A = {s: numpy.stack([per[m][sd][s][sel] for m, sd in rows]) for s in STATS}
    n = int(sel.sum())
    point = {f: _spear(A[f].ravel(), A["pt"].ravel()) for f in FEATS}
    boot = {f: [] for f in FEATS}
    for _ in range(NBOOT):
        idx = gen.integers(0, n, n)
        p = A["pt"][:, idx].ravel()
        for f in FEATS:
            boot[f].append(_spear(A[f][:, idx].ravel(), p))
    boot = {f: numpy.array(v) for f, v in boot.items()}

    def q(v):
        return [round(float(numpy.quantile(v, 0.025)), 4), round(float(numpy.quantile(v, 0.975)), 4)]

    sp = {f: [round(point[f], 4)] + q(boot[f]) for f in FEATS}
    sp["jac_minus_cov"] = [round(point["jac"] - point["cov"], 4)] + q(boot["jac"] - boot["cov"])
    sp["absdisp_minus_abscov"] = [round(abs(point["disp"]) - abs(point["cov"]), 4)] + q(
        numpy.abs(boot["disp"]) - numpy.abs(boot["cov"]))
    # same Spearman within each method separately (record)
    sp_m = {}
    for m in METHODS:
        ix = [i for i, (mm, _) in enumerate(rows) if mm == m]
        sp_m[m] = {f: round(_spear(A[f][ix].ravel(), A["pt"][ix].ravel()), 4) for f in FEATS}
    out["group2_pooled_spearman_with_pt"] = sp
    out["group2_spearman_by_method"] = sp_m
    out["group2_n_rows"] = int(A["pt"].size)

    omf = out["online_minus_frozen"]
    out["verdict"] = {
        "P1_group2_jac_online_lower": bool(omf["jac"][str(FOCUS)][2] < 0),
        "P2_group2_spearman_jac_gt_cov": bool(sp["jac_minus_cov"][1] > 0),
        "R_group2_disp_online_minus_frozen": omf["disp"][str(FOCUS)],
        "R_group2_cov_online_minus_frozen": omf["cov"][str(FOCUS)],
    }
    api.write_json("summary", out)
