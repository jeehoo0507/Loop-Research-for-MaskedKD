"""H-A198 (round 462, analysis only, no training): Waterbirds per-group version of the iso-KL token budget of H-A192
(Caltech-256) / H-A196 (CUB) / H-A197 (Aircraft), same K grid, view draws and sink definition. Does the MaskedKD rule
(last-block CLS attention top-K) need more teacher tokens on minority-group images than on majority-group images of
the same class to preserve the full-teacher soft label as well as student rollout of blocks 6-11 (lateroll) does at
K=29? Groups are read only for the analysis (2*y + place); the rules never see them.

Students (waterbirds, seed 1): full@196 (primary), maskedkd@98 (record).
Views: clean val + V = 2 fixed crop+flip views per val image (same draw formula as H-A181/192/196/197).
Rules (decomposition controls, not selectors):
  maskedkd  last-block (11) CLS attention top-K, K = 12, 29, 39, 49, 59, 69, 79, 98
  nosink    same scores, sinks (block-10 output patch norm > median + 3 MAD per view) set to -1, same K (record)
  lateroll  student rollout blocks 6-11 with residual identity, K = 12, 29, 49 (targets)
Per image: KL(full teacher on the view || keep-only masked teacher on the same view), T=1, mean over the V views.
Per group g: iso-KL budget B_g = smallest K on the maskedkd grid (linear interpolation) at which the group-mean KL falls
to the group-mean lateroll@29 KL; inf when not reached by K=98, capped at 98 for differences (fraction reached kept).
P1: B_2 - B_3 (waterbird on land minus waterbird on water), aug views, full student.
P2: B_1 - B_0 (landbird on water minus landbird on land), aug views, full student.
CI: 95% percentile bootstrap, images resampled within each group (numpy default_rng(20261223), 2000 resamples).
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on Waterbirds crop+flip views of val, does the last-block CLS attention rule (MaskedKD) need more teacher "
                "tokens on minority-group images than on majority-group images of the same class to match the "
                "masked-teacher KL that block 6-11 rollout reaches with 29 tokens?",
    "method": "keep-only masked-teacher KL on RandomResizedCrop+flip views over a token-budget grid, per-group iso-KL "
              "budget by linear interpolation, group-stratified image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Darcet et al. ICLR 2024 (ViTs need "
             "registers); Sagawa et al. ICLR 2020 (Waterbirds groups); Efron & Tibshirani 1993",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261223
V = 2
IMG = 224
MK_KS = (12, 29, 39, 49, 59, 69, 79, 98)
LR_KS = (12, 29, 49)
MAD_K = 3.0
STUDENTS = {
    "full": "final_full_k196_s1_1030_201811_b764",
    "maskedkd98": "waterbirds_final_maskedkd_k98_s1_1031_130203_5bef",
}
TRAINED = {  # seed 1, 2, 3 (record only: trained val group accuracy, no forward pass)
    "maskedkd@29": ["waterbirds_final_maskedkd_k29_s1_1031_201502_4be8", "waterbirds_final_maskedkd_k29_s2_1103_170931_92bd",
                    "waterbirds_final_maskedkd_k29_s3_1103_170935_907a"],
    "lateroll@29": ["waterbirds_final_lateroll_k29_s1_1104_025003_b396", "waterbirds_final_lateroll_k29_s2_1104_025008_c94e",
                    "waterbirds_final_lateroll_k29_s3_1104_025012_629b"],
    "maskedkd@98": ["waterbirds_final_maskedkd_k98_s1_1031_130203_5bef", "waterbirds_final_maskedkd_k98_s2_1031_201502_95ac",
                    "waterbirds_final_maskedkd_k98_s3_1031_201502_0f31"],
}
CELLS = [("maskedkd", k) for k in MK_KS] + [("nosink", k) for k in MK_KS] + [("lateroll", k) for k in LR_KS]


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _topk_mask(scores, k):
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(1, scores.topk(k, dim=1).indices, True)
    return keep


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _grids(n, device):
    """V crop+flip affine grids per val image, harness GpuAugment RandomResizedCrop formula, fixed draws (H-A181)."""
    gen = numpy.random.default_rng(20261203)
    area = gen.uniform(0.08, 1.0, (V, n))
    r = numpy.exp(gen.uniform(math.log(3 / 4), math.log(4 / 3), (V, n)))
    w = numpy.minimum(numpy.sqrt(area * r), 1.0)
    h = numpy.minimum(numpy.sqrt(area / r), 1.0)
    cx = (gen.uniform(0, 1, (V, n)) * 2 - 1) * (1 - w)
    cy = (gen.uniform(0, 1, (V, n)) * 2 - 1) * (1 - h)
    flip = numpy.where(gen.uniform(0, 1, (V, n)) < 0.5, -1.0, 1.0)
    th = numpy.zeros((V, n, 2, 3))
    th[:, :, 0, 0] = w * flip
    th[:, :, 0, 2] = cx
    th[:, :, 1, 1] = h
    th[:, :, 1, 2] = cy
    return torch.tensor(th, dtype=torch.float32, device=device)


def _view(x, theta):
    if theta is None:
        return x
    grid = F.affine_grid(theta, (len(x), 3, IMG, IMG), align_corners=False)
    return F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _iso(curve, target):
    """curve: (B, len(MK_KS)) mean KL per K; target: (B,). First K where curve <= target, linear interpolation."""
    ks = numpy.asarray(MK_KS, dtype=numpy.float64)
    out = numpy.full(len(target), numpy.inf)
    for b in range(len(target)):
        c, t = curve[b], target[b]
        if c[0] <= t:
            out[b] = ks[0]
            continue
        for j in range(1, len(ks)):
            if c[j] <= t:
                out[b] = ks[j - 1] + (c[j - 1] - t) / (c[j - 1] - c[j]) * (ks[j] - ks[j - 1])
                break
    return out


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"]] for rid in STUDENTS.values()})
    Tm = api.teacher.eval()
    xs, ys = [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
    y_all = torch.cat(ys)
    n = len(y_all)
    grp = api.val_groups().cpu().numpy().astype(int)
    th = _grids(n, y_all.device)
    vts = ["clean"] + list(range(V))

    views, tl = [], {vt: [] for vt in vts}
    off = 0
    for x in xs:
        b = len(x)
        vb = [("clean", None)] + [(v, th[v, off:off + b]) for v in range(V)]
        for vt, theta in vb:
            with torch.no_grad():
                tl[vt].append(Tm(_view(x, theta)).float().cpu())
        views.append(vb)
        off += b
    yn = y_all.cpu().numpy().astype(int)
    tlp = {vt: _logp(torch.cat(tl[vt]).numpy().astype(numpy.float64)) for vt in vts}
    full_acc = {"clean": round(float((tlp["clean"].argmax(1) == yn).mean() * 100), 3),
                "aug": round(float(numpy.mean([(tlp[v].argmax(1) == yn).mean() for v in range(V)]) * 100), 3)}

    kl, acc, s_acc = {}, {}, {}
    for sname, rid in STUDENTS.items():
        S = api.student(rid)
        depth = len(S.blocks)
        gl = {(vt, c): [] for vt in vts for c in CELLS}
        sl = []
        for x, vb in zip(xs, views):
            for vt, theta in vb:
                xv = _view(x, theta)
                with torch.no_grad():
                    if vt == "clean":
                        sl.append(api.student_forward(S, xv).logits.float().cpu())
                    out = api.student_forward(S, xv, attn_layers=list(range(6, depth)), token_layers=[depth - 2])
                    att = [out.attn[i] for i in range(6, depth)]
                    last = att[-1].float().mean(1)[:, 0, 1:]
                    nrm = out.tokens[depth - 2][:, 1:].float().norm(dim=-1)
                    med = nrm.median(dim=1, keepdim=True).values
                    mad = (nrm - med).abs().median(dim=1, keepdim=True).values
                    sink = nrm > med + MAD_K * mad
                    sc = {"maskedkd": last, "nosink": last.masked_fill(sink, -1.0), "lateroll": _rollout(att)}
                    for c in CELLS:
                        m = _topk_mask(sc[c[0]], c[1])
                        gl[(vt, c)].append(api.teacher_forward_keepmask(Tm, xv, m).float().cpu())
        del S
        torch.cuda.empty_cache()
        s_acc[sname] = round(float((torch.cat(sl).numpy().argmax(1) == yn).mean() * 100), 3)

        for tag, vt_set in (("clean", ["clean"]), ("aug", list(range(V)))):
            for c in CELLS:
                ka, aa = [], []
                for vt in vt_set:
                    glp = _logp(torch.cat(gl[(vt, c)]).numpy().astype(numpy.float64))
                    ka.append((numpy.exp(tlp[vt]) * (tlp[vt] - glp)).sum(1))
                    aa.append((glp.argmax(1) == yn).astype(numpy.float64) * 100)
                kl[(sname, tag) + c] = numpy.mean(ka, 0)
                acc[(sname, tag) + c] = numpy.mean(aa, 0)

    rng = numpy.random.default_rng(SEED)
    gidx = {g: numpy.flatnonzero(grp == g) for g in range(4)}
    bidx = {g: gidx[g][rng.integers(0, len(gidx[g]), size=(N_BOOT, len(gidx[g])))] for g in range(4)}

    def _ci(bs):
        return [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]

    def iso_g(s, tag, rule, k_target, g):
        """(point, bootstrap array capped at 98, fraction reached by 98)."""
        pts = numpy.stack([kl[(s, tag, rule, k)] for k in MK_KS], 1)
        tgt = kl[(s, tag, "lateroll", k_target)]
        ii = gidx[g]
        point = _iso(pts[ii].mean(0, keepdims=True), numpy.array([tgt[ii].mean()]))[0]
        bs = _iso(pts[bidx[g]].mean(1), tgt[bidx[g]].mean(1))
        return point, numpy.minimum(bs, 98.0), float(numpy.isfinite(bs).mean())

    def _fmt(v):
        return round(float(v), 2) if numpy.isfinite(v) else "inf"

    def contrast(s, tag, rule, kt, ga, gb):
        pa, ba, fa = iso_g(s, tag, rule, kt, ga)
        pb, bb, fb = iso_g(s, tag, rule, kt, gb)
        d = ba - bb
        return {"point": round(float(min(pa, 98.0) - min(pb, 98.0)), 2), "ci": _ci(d),
                "B_a": _fmt(pa), "B_b": _fmt(pb), "frac_reached_a": round(fa, 4), "frac_reached_b": round(fb, 4)}

    p1 = contrast("full", "aug", "maskedkd", 29, 2, 3)
    p2 = contrast("full", "aug", "maskedkd", 29, 1, 0)
    rec = {}
    for s in STUDENTS:
        rec[s] = {"student_acc": s_acc[s]}
        for tag in ("clean", "aug"):
            per_g = {}
            for g in range(4):
                ii = gidx[g]
                per_g[str(g)] = {
                    "n": int(len(ii)),
                    "kl_mean": {f"{c[0]}@{c[1]}": round(float(kl[(s, tag) + c][ii].mean()), 4) for c in CELLS},
                    "masked_teacher_acc": {f"{c[0]}@{c[1]}": round(float(acc[(s, tag) + c][ii].mean()), 3)
                                           for c in CELLS},
                    "iso_budget": {f"{rule}_vs_lateroll@{kt}": (lambda r: {"point": _fmt(r[0]), "ci": _ci(r[1]),
                                                                           "frac_reached_by_98": round(r[2], 4)})(
                        iso_g(s, tag, rule, kt, g)) for rule in ("maskedkd", "nosink") for kt in LR_KS},
                    "kl_diff_maskedkd29_minus_lateroll29": {
                        "mean": round(float((kl[(s, tag, "maskedkd", 29)] - kl[(s, tag, "lateroll", 29)])[ii].mean()), 4),
                        "ci": _ci((kl[(s, tag, "maskedkd", 29)] - kl[(s, tag, "lateroll", 29)])[bidx[g]].mean(1))},
                }
            rec[s][tag] = {
                "per_group": per_g,
                "contrasts": {f"{rule}_vs_lateroll@{kt}": {"g2-g3": contrast(s, tag, rule, kt, 2, 3),
                                                            "g1-g0": contrast(s, tag, rule, kt, 1, 0)}
                              for rule in ("maskedkd", "nosink") for kt in LR_KS},
            }
    tr = {m: {g: numpy.array([float(runs[r]["val_group_acc_last"][str(g)]) for r in rids]) for g in range(4)}
          for m, rids in TRAINED.items()}

    def _tci(d):
        se = d.std(ddof=1) / math.sqrt(len(d))
        return {"mean": round(float(d.mean()), 3), "per_seed": [round(float(v), 3) for v in d],
                "t95_ci": [round(float(d.mean() - 4.303 * se), 3), round(float(d.mean() + 4.303 * se), 3)]}

    trained = {"seed_mean_group_acc": {m: {str(g): round(float(v[g].mean()), 3) for g in range(4)} for m, v in tr.items()},
               "lateroll@29-maskedkd@29_per_group": {str(g): _tci(tr["lateroll@29"][g] - tr["maskedkd@29"][g])
                                                     for g in range(4)},
               "datasets": sorted({str(runs[r].get("dataset")) for rids in TRAINED.values() for r in rids})}
    api.write_json("summary", {
        "trained_record": trained,
        "n_val": int(n), "group_n": {str(g): int(len(gidx[g])) for g in range(4)}, "views_per_image": V,
        "full_teacher_acc": full_acc,
        "full_teacher_acc_aug_per_group": {str(g): round(float(numpy.mean(
            [(tlp[v].argmax(1)[gidx[g]] == yn[gidx[g]]).mean() for v in range(V)]) * 100), 3) for g in range(4)},
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_iso_budget_g2_minus_g3_aug_full": p1,
        "P2_iso_budget_g1_minus_g0_aug_full": p2,
        "record": rec,
        "note": "group = 2*y + place (0 landbird/land, 1 landbird/water, 2 waterbird/land, 3 waterbird/water); "
                "KL(full teacher on view || keep-only masked teacher on same view), T=1, nats; aug = mean over V "
                "crop+flip views per image (H-A181 draws); sinks = block-10 output patch norm > median + 3 MAD per "
                "view; iso budget per group = smallest K on the maskedkd/nosink grid (linear interpolation) where "
                "group-mean KL <= group-mean lateroll@K_target KL, per bootstrap resample, inf if not reached by K=98 "
                "(capped at 98 in differences); 95% percentile bootstrap, images resampled within each group "
                "(2000 resamples); seed-1 students only"})
