"""H-A53 (round 300, analysis only, no training): on Waterbirds K=29, do the tokens only the online MaskedKD student
picks (D_O) push the masked teacher toward the online student's own prediction (self-confirmation, the loop the
gate-3 question asks about), and is the group-2 harm of D_O concentrated where the online student is itself wrong?

H-A52: in group 2, pt(D_O) - pt(D_F) = -0.185 and adding D_O to the shared core lowers pt (-0.144). It did not say
whether D_O follows the student's own belief. Per seed pair (online s, frozen s), same sets as H-A52:
  O = online set, F = frozen set, D_O = O \\ F, D_F = F \\ O
  yO = online student argmax (final), yF = frozen student argmax (final)
  conf_O = p_T(D_O)(yO) - p_T(D_F)(yO)       label-free: do online-only tokens favour the online student's own class?
  conf_F = p_T(D_F)(yF) - p_T(D_O)(yF)       control: same for the frozen student
  d_pt   = p_T(D_O)(y)  - p_T(D_F)(y)        H-A52 contrast, split in group 2 by online correctness (yO == y)

Students: Waterbirds final maskedkd@29 and maskedkd_frozen@29, seeds 1, 2, 3, final epoch; mask rule for every student:
its own last-block head-mean CLS attention top-K. Un-augmented val. Keep-only masked teacher, T 1.
Group means: per image seed mean (missing skipped), image bootstrap. Group-2 correctness split: (image, seed) rows,
image-cluster bootstrap. 2000 draws, numpy default_rng 20261117.
No labels, groups or positions enter any mask; labels enter only pt and the correctness split, groups only slice.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29, do online-only MaskedKD tokens push the masked teacher toward the online "
                "student's own prediction, and is their group-2 harm concentrated where the online student is wrong?",
    "method": "last-block CLS attention top-K; set differences vs frozen-trained student; keep-only masked-teacher "
              "probability of the student's own argmax; image (cluster) bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 (keep-only); Sagawa et al. ICLR 2020 "
             "(Waterbirds groups)",
}

K = 29
NBOOT = 2000
BATCH = 50
ON, FR = "maskedkd", "maskedkd_frozen"
GROUPS = (0, 1, 2, 3)
FOCUS = 2


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


def _student(api, S, batches):
    last = len(S.blocks) - 1
    keeps, preds = [], []
    for x, _ in batches:
        with torch.no_grad():
            so = api.student_forward(S, x, attn_layers=[last])
        keeps.append(_topk(so.attn[last].float().mean(1)[:, 0, 1:], K))
        preds.append(so.logits.float().argmax(-1).cpu())
    return keeps, preds


def _pick(p, idx):
    return p.gather(1, idx[:, None])[:, 0]


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
    ys = torch.cat([y.cpu() for _, y in batches]).numpy()
    groups = api.val_groups().numpy()

    per = {}
    for s in seeds:
        ko, po = _student(api, api.student(studs[ON][s]), batches)
        kf, pf = _student(api, api.student(studs[FR][s]), batches)
        rows = {"conf_O": [], "conf_F": [], "d_pt": [], "okO": [], "okF": []}
        for (x, y), o, f, yo, yf in zip(batches, ko, kf, po, pf):
            yc = y.cpu()
            pdo = _probs(api, T, x, o & ~f)
            pdf = _probs(api, T, x, f & ~o)
            rows["conf_O"].append(_pick(pdo, yo) - _pick(pdf, yo))
            rows["conf_F"].append(_pick(pdf, yf) - _pick(pdo, yf))
            rows["d_pt"].append(_pick(pdo, yc) - _pick(pdf, yc))
            rows["okO"].append((yo == yc).float())
            rows["okF"].append((yf == yc).float())
        per[s] = {z: torch.cat(v).numpy() for z, v in rows.items()}

    gen = numpy.random.default_rng(20261117)
    stack = {z: numpy.stack([per[s][z] for s in seeds]) for z in per[seeds[0]]}
    with numpy.errstate(all="ignore"):
        m = {z: numpy.nanmean(stack[z], 0) for z in ("conf_O", "conf_F", "d_pt")}
        m["confO_minus_confF"] = numpy.nanmean(stack["conf_O"] - stack["conf_F"], 0)

    out = {"k": K, "n_val": int(len(groups)), "seeds": seeds,
           "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": {mm: {str(s): r for s, r in v.items()} for mm, v in studs.items()}}
    out["all"] = {z: _ci(v, gen) for z, v in m.items()}
    out["by_group"] = {z: {str(g): _ci(v[groups == g], gen) for g in GROUPS} for z, v in m.items()}
    out["acc_by_group"] = {w: {str(g): round(float(stack[w][:, groups == g].mean()), 4) for g in GROUPS}
                           for w in ("okO", "okF")}

    # group-2 split of d_pt by online correctness: (image, seed) rows, image-cluster bootstrap
    img = numpy.where(groups == FOCUS)[0]
    dp = stack["d_pt"][:, img]
    ok = stack["okO"][:, img]
    valid = ~numpy.isnan(dp)

    def _split(cols):
        d, o, v = dp[:, cols], ok[:, cols], valid[:, cols]
        w = d[(o == 0) & v]
        r = d[(o == 1) & v]
        if len(w) == 0 or len(r) == 0:
            return float("nan"), float("nan"), float("nan"), len(w), len(r)
        return float(w.mean()), float(r.mean()), float(w.mean() - r.mean()), len(w), len(r)

    base = _split(numpy.arange(len(img)))
    bs = []
    for _ in range(NBOOT):
        bs.append(_split(gen.integers(0, len(img), len(img)))[2])
    bs = numpy.array(bs)
    bs = bs[~numpy.isnan(bs)]
    out["group2_dpt_wrong_vs_right"] = {
        "wrong_mean": round(base[0], 4), "right_mean": round(base[1], 4),
        "diff": [round(base[2], 4), round(float(numpy.quantile(bs, 0.025)), 4),
                 round(float(numpy.quantile(bs, 0.975)), 4)],
        "n_rows_wrong": base[3], "n_rows_right": base[4]}

    a = out["all"]
    g2 = out["group2_dpt_wrong_vs_right"]["diff"]
    out["verdict"] = {
        "P1_all_conf_O_lower_gt0": bool(a["conf_O"][1] is not None and a["conf_O"][1] > 0),
        "P2_group2_dpt_wrong_minus_right_upper_lt0": bool(g2[2] < 0),
        "R_all_conf_F": a["conf_F"],
        "R_all_confO_minus_confF": a["confO_minus_confF"],
    }
    api.write_json("summary", out)
