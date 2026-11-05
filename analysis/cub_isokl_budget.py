"""H-A196 (round 460, analysis only, no training): CUB-200-2011 replication of H-A192 (Caltech-256), same code with
student ids, bootstrap seed and description changed. How many teacher tokens does the MaskedKD rule (last-block CLS
attention top-K) need to preserve the full-teacher soft label as well as student rollout of blocks 6-11 (lateroll)
does at K=29, on the crop+flip views training sees (H-A181/188 draw formula)?

Students (cub, seed 1, same runs as H-A195): full@196 (primary), maskedkd@98 (record).
Views: clean val + V = 2 fixed crop+flip views per val image (same draws as H-A181/188/189/190/191).
Rules (decomposition controls, not selectors):
  maskedkd  last-block (11) CLS attention top-K, K = 12, 29, 39, 49, 59, 69, 79, 98
  nosink    same scores, sinks (block-10 output patch norm > median + 3 MAD per view) set to -1, same K (record)
  lateroll  student rollout blocks 6-11 with residual identity, K = 12, 29, 49 (targets)
Per image: KL(full teacher on the view || keep-only masked teacher on the same view), T=1, mean over the V views.
P1: KL_maskedkd@49 - KL_lateroll@29, aug views, full student.
P2: KL_maskedkd@69 - KL_lateroll@29, aug views, full student.
Record: iso-KL budget = smallest K on the maskedkd (and nosink) grid, linearly interpolated, at which the mean KL falls
to the lateroll@29 (and @12, @49) mean, per bootstrap resample (inf when not reached by K=98).
CI: 95% percentile bootstrap over val images (numpy default_rng(20261221), 2000 resamples, shared index).
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on CUB-200-2011 crop+flip views of val, how many teacher tokens does the last-block CLS attention rule "
                "(MaskedKD) need to match the masked-teacher KL that block 6-11 rollout reaches with 29 tokens?",
    "method": "keep-only masked-teacher KL on RandomResizedCrop+flip views over a token-budget grid, iso-KL budget by "
              "linear interpolation, paired image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Darcet et al. ICLR 2024 (ViTs need "
             "registers); Szegedy et al. 2015 (RandomResizedCrop); Efron & Tibshirani 1993; Wah et al. 2011 (CUB)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261221
V = 2
IMG = 224
MK_KS = (12, 29, 39, 49, 59, 69, 79, 98)
LR_KS = (12, 29, 49)
MAD_K = 3.0
STUDENTS = {
    "full": "cub_final_full_k196_s1_1030_153055_d87b",
    "maskedkd98": "cub_final_maskedkd_k98_s1_1030_153055_cdbc",
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
    idx = rng.integers(0, n, size=(N_BOOT, n))

    def _ci(bs):
        return [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]

    def boot(dv):
        dv = numpy.asarray(dv, dtype=numpy.float64)
        return {"mean": round(float(dv.mean()), 4), "ci": _ci(dv[idx].mean(1))}

    def iso(s, tag, rule, k_target):
        pts = numpy.stack([kl[(s, tag, rule, k)] for k in MK_KS], 1)
        tgt = kl[(s, tag, "lateroll", k_target)]
        point = _iso(pts.mean(0, keepdims=True), numpy.array([tgt.mean()]))[0]
        bs = _iso(pts[idx].mean(1), tgt[idx].mean(1))
        fin = numpy.isfinite(bs)
        return {"point": round(float(point), 2) if numpy.isfinite(point) else "inf",
                "ci": [round(float(numpy.percentile(bs, q)), 2) if numpy.isfinite(numpy.percentile(bs, q)) else "inf"
                       for q in (2.5, 97.5)],
                "frac_reached_by_98": round(float(fin.mean()), 4),
                "token_ratio_point": round(float(point) / k_target, 3) if numpy.isfinite(point) else "inf"}

    p1 = boot(kl[("full", "aug", "maskedkd", 49)] - kl[("full", "aug", "lateroll", 29)])
    p2 = boot(kl[("full", "aug", "maskedkd", 69)] - kl[("full", "aug", "lateroll", 29)])
    rec = {}
    for s in STUDENTS:
        rec[s] = {"student_acc": s_acc[s]}
        for tag in ("clean", "aug"):
            rec[s][tag] = {
                "kl_mean": {f"{c[0]}@{c[1]}": round(float(kl[(s, tag) + c].mean()), 4) for c in CELLS},
                "masked_teacher_acc": {f"{c[0]}@{c[1]}": round(float(acc[(s, tag) + c].mean()), 3) for c in CELLS},
                "iso_budget": {f"{rule}_vs_lateroll@{kt}": iso(s, tag, rule, kt)
                               for rule in ("maskedkd", "nosink") for kt in LR_KS},
                "kl_diff": {f"maskedkd@{k}-lateroll@29": boot(kl[(s, tag, "maskedkd", k)]
                                                              - kl[(s, tag, "lateroll", 29)]) for k in MK_KS},
                "kl_diff_nosink": {f"nosink@{k}-lateroll@29": boot(kl[(s, tag, "nosink", k)]
                                                                   - kl[(s, tag, "lateroll", 29)]) for k in MK_KS},
            }
    api.write_json("summary", {
        "n_val": int(n), "views_per_image": V, "full_teacher_acc": full_acc,
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_kl_maskedkd49_minus_lateroll29_aug_full": p1,
        "P2_kl_maskedkd69_minus_lateroll29_aug_full": p2,
        "record": rec,
        "note": "KL(full teacher on view || keep-only masked teacher on same view), T=1, nats; aug = mean over V "
                "crop+flip views per image (H-A181 draws); sinks = block-10 output patch norm > median + 3 MAD per "
                "view; iso_budget = smallest K on the maskedkd/nosink grid (linear interpolation) where mean KL <= "
                "lateroll@K_target mean KL, per bootstrap resample, inf if not reached by K=98; 95% percentile "
                "bootstrap over val images (2000 resamples, shared index); seed-1 students only"})
