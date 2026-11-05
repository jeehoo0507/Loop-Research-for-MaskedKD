"""H-A200 (round 464, analysis only, no training): does the Waterbirds minority-group iso-KL budget gap of H-A198 grow
while a student is trained with the MaskedKD@29 rule (online feedback, gate 3), compared with a Full-KD student?
Same K grid, view draws, sink definition and budget definition as H-A198; students are the e25 / e50 / final
snapshots of two seed-1 runs. Each snapshot scores its own tokens and gives its own lateroll@29 target.
Groups are read only for the analysis (2*y + place); the rules never see them.

Students (waterbirds, seed 1): maskedkd@29 run (primary), full@196 run (control); epochs 25, 50, final (100).
Views: clean val + V = 2 fixed crop+flip views per val image (same draw formula as H-A181/192/196/197/198).
Rules (decomposition controls, not selectors):
  maskedkd  last-block (11) CLS attention top-K, K = 12, 29, 39, 49, 59, 69, 79, 98
  nosink    same scores, sinks (block-10 output patch norm > median + 3 MAD per view) set to -1, same K (record)
  lateroll  student rollout blocks 6-11 with residual identity, K = 12, 29, 49 (targets)
Per image: KL(full teacher on the view || keep-only masked teacher on the same view), T=1, mean over the V views.
Per group g and snapshot: iso-KL budget B_g = smallest K on the maskedkd grid (linear interpolation) at which the
group-mean KL falls to the group-mean lateroll@29 KL of the same snapshot; inf when not reached by K=98, capped at 98.
Gap G = B_1 - B_0 (landbird: water background minus land background).
P1: G(final) - G(e25), maskedkd@29 student, aug views.  P2: same for the full@196 student.
CI: 95% percentile bootstrap, images resampled within each group, the same resample indices for every snapshot and
student (paired), numpy default_rng(20261225), 2000 resamples.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on Waterbirds crop+flip views of val, does the minority-minus-majority landbird gap in the teacher-token "
                "budget that the last-block CLS attention rule needs to match block 6-11 rollout@29 grow from epoch 25 "
                "to the end of training for a student trained with MaskedKD@29, and for a Full-KD student?",
    "method": "keep-only masked-teacher KL on RandomResizedCrop+flip views over a token-budget grid, per-group iso-KL "
              "budget by linear interpolation for e25/e50/final snapshots, paired group-stratified image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Darcet et al. ICLR 2024 (ViTs need "
             "registers); Sagawa et al. ICLR 2020 (Waterbirds groups); Efron & Tibshirani 1993",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261225
V = 2
IMG = 224
MK_KS = (12, 29, 39, 49, 59, 69, 79, 98)
LR_KS = (12, 29, 49)
MAD_K = 3.0
RUNS = {
    "mk29": "waterbirds_final_maskedkd_k29_s1_1031_201502_4be8",
    "full": "final_full_k196_s1_1030_201811_b764",
}
EPOCHS = (25, 50, "final")
CELLS = [("maskedkd", k) for k in MK_KS] + [("nosink", k) for k in MK_KS] + [("lateroll", k) for k in LR_KS]
PAIRS = {"g1-g0": (1, 0), "g2-g3": (2, 3)}


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
                                      runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid]["val_wga_last"]]
                                for rid in RUNS.values()})
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

    kl, acc, s_acc, s_gacc = {}, {}, {}, {}
    for rname, rid in RUNS.items():
        for ep in EPOCHS:
            sname = (rname, str(ep))
            S = api.student(rid, epoch=None if ep == "final" else ep)
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
            correct = torch.cat(sl).numpy().argmax(1) == yn
            s_acc[sname] = round(float(correct.mean() * 100), 3)
            s_gacc[sname] = {str(g): round(float(correct[grp == g].mean() * 100), 3) for g in range(4)}

            for tag, vt_set in (("clean", ["clean"]), ("aug", list(range(V)))):
                for c in CELLS:
                    ka, aa = [], []
                    for vt in vt_set:
                        glp = _logp(torch.cat(gl[(vt, c)]).numpy().astype(numpy.float64))
                        ka.append((numpy.exp(tlp[vt]) * (tlp[vt] - glp)).sum(1))
                        aa.append((glp.argmax(1) == yn).astype(numpy.float64) * 100)
                    kl[sname + (tag,) + c] = numpy.mean(ka, 0)
                    acc[sname + (tag,) + c] = numpy.mean(aa, 0)

    rng = numpy.random.default_rng(SEED)
    gidx = {g: numpy.flatnonzero(grp == g) for g in range(4)}
    bidx = {g: gidx[g][rng.integers(0, len(gidx[g]), size=(N_BOOT, len(gidx[g])))] for g in range(4)}

    def _ci(bs):
        return [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]

    def _fmt(v):
        return round(float(v), 2) if numpy.isfinite(v) else "inf"

    def iso_g(s, tag, rule, kt, g):
        """(point, bootstrap array capped at 98, fraction reached by 98) for snapshot s."""
        pts = numpy.stack([kl[s + (tag, rule, k)] for k in MK_KS], 1)
        tgt = kl[s + (tag, "lateroll", kt)]
        ii = gidx[g]
        point = _iso(pts[ii].mean(0, keepdims=True), numpy.array([tgt[ii].mean()]))[0]
        bs = _iso(pts[bidx[g]].mean(1), tgt[bidx[g]].mean(1))
        return point, numpy.minimum(bs, 98.0), float(numpy.isfinite(bs).mean())

    def gap(s, tag, rule, kt, pair):
        ga, gb = PAIRS[pair]
        pa, ba, fa = iso_g(s, tag, rule, kt, ga)
        pb, bb, fb = iso_g(s, tag, rule, kt, gb)
        return min(pa, 98.0) - min(pb, 98.0), ba - bb, {"B_a": _fmt(pa), "B_b": _fmt(pb), "frac_reached_a": round(fa, 4),
                                                        "frac_reached_b": round(fb, 4)}

    def change(rname, tag, rule, kt, pair, e_from, e_to):
        p0, b0, i0 = gap((rname, str(e_from)), tag, rule, kt, pair)
        p1, b1, i1 = gap((rname, str(e_to)), tag, rule, kt, pair)
        return {"point": round(float(p1 - p0), 2), "ci": _ci(b1 - b0), "gap_from": round(float(p0), 2),
                "gap_to": round(float(p1), 2), "from": i0, "to": i1}, b1 - b0

    p1, bs1 = change("mk29", "aug", "maskedkd", 29, "g1-g0", 25, "final")
    p2, bs2 = change("full", "aug", "maskedkd", 29, "g1-g0", 25, "final")
    p_minus = numpy.array(bs1) - numpy.array(bs2)

    rec = {}
    for rname in RUNS:
        for ep in EPOCHS:
            s = (rname, str(ep))
            key = f"{rname}_e{ep}"
            rec[key] = {"student_acc": s_acc[s], "student_group_acc_clean": s_gacc[s]}
            for tag in ("clean", "aug"):
                per_g = {}
                for g in range(4):
                    ii = gidx[g]
                    d = kl[s + (tag, "maskedkd", 29)] - kl[s + (tag, "lateroll", 29)]
                    per_g[str(g)] = {
                        "n": int(len(ii)),
                        "kl_mean": {f"{c[0]}@{c[1]}": round(float(kl[s + (tag,) + c][ii].mean()), 4) for c in CELLS},
                        "masked_teacher_acc": {f"{c[0]}@{c[1]}": round(float(acc[s + (tag,) + c][ii].mean()), 3)
                                               for c in CELLS},
                        "iso_budget_maskedkd_vs_lateroll@29": (lambda r: {"point": _fmt(r[0]), "ci": _ci(r[1]),
                                                                          "frac_reached_by_98": round(r[2], 4)})(
                            iso_g(s, tag, "maskedkd", 29, g)),
                        "kl_diff_maskedkd29_minus_lateroll29": {"mean": round(float(d[ii].mean()), 4),
                                                                "ci": _ci(d[bidx[g]].mean(1))},
                    }
                gaps = {}
                for rule in ("maskedkd", "nosink"):
                    for kt in LR_KS:
                        for pair in PAIRS:
                            pt, bs, info = gap(s, tag, rule, kt, pair)
                            gaps[f"{rule}_vs_lateroll@{kt}_{pair}"] = dict(point=round(float(pt), 2), ci=_ci(bs), **info)
                rec[key][tag] = {"per_group": per_g, "gaps": gaps}

    changes = {}
    for rname in RUNS:
        for tag in ("clean", "aug"):
            for rule in ("maskedkd", "nosink"):
                for pair in PAIRS:
                    for e_from, e_to in ((25, 50), (50, "final"), (25, "final")):
                        changes[f"{rname}_{tag}_{rule}_vs_lateroll@29_{pair}_e{e_from}_to_e{e_to}"] = change(
                            rname, tag, rule, 29, pair, e_from, e_to)[0]

    api.write_json("summary", {
        "n_val": int(n), "group_n": {str(g): int(len(gidx[g])) for g in range(4)}, "views_per_image": V,
        "full_teacher_acc": full_acc,
        "full_teacher_acc_aug_per_group": {str(g): round(float(numpy.mean(
            [(tlp[v].argmax(1)[gidx[g]] == yn[gidx[g]]).mean() for v in range(V)]) * 100), 3) for g in range(4)},
        "student_acc_check_final": {r: [s_acc[(r, "final")], runs[RUNS[r]]["val_acc_last"]] for r in RUNS},
        "P1_mk29_gap_g1_minus_g0_final_minus_e25_aug": p1,
        "P2_full_gap_g1_minus_g0_final_minus_e25_aug": p2,
        "record_P1_minus_P2": {"point": round(float(p1["point"] - p2["point"]), 2), "ci": _ci(p_minus)},
        "changes": changes,
        "record": rec,
        "note": "group = 2*y + place (0 landbird/land, 1 landbird/water, 2 waterbird/land, 3 waterbird/water); "
                "KL(full teacher on view || keep-only masked teacher on same view), T=1, nats; aug = mean over V "
                "crop+flip views per image (H-A181 draws); sinks = block-10 output patch norm > median + 3 MAD per "
                "view; iso budget per group = smallest K on the maskedkd/nosink grid (linear interpolation) where "
                "group-mean KL <= group-mean lateroll@K_target KL of the same snapshot, per bootstrap resample, inf "
                "if not reached by K=98 (capped at 98 in differences); 95% percentile bootstrap, images resampled "
                "within each group with the same indices for every snapshot (2000 resamples); seed-1 runs only"})
