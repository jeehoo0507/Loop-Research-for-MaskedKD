"""H-A202 (round 466, analysis only, no training): CUB-200-2011 headroom of the token score, in iso-KL token units.
H-A196: on CUB crop+flip views the MaskedKD rule needs 44.5 teacher tokens to match the masked-teacher KL that student
rollout of blocks 6-11 (lateroll) reaches with 29. How far below 29 could a better score go? Two teacher-side scores
bound this (analysis references, not selectors; no new training method):
  tview   teacher all-block rollout (residual identity) computed on the view itself = an oracle that needs a full
          teacher pass, so it saves no FLOPs; it is the ceiling for a rollout-type score
  tcache  clean-image teacher all-block rollout warped to the view with the crop grid and patch-averaged (as
          GpuAugment builds ctx.teacher_map; H-A173 code) = the cached teacher signal teacher_topk uses, cost-free
Student rules (same definitions as H-A196): maskedkd = last-block CLS attention top-K; lateroll = student rollout
blocks 6-11 top-K, K = 12, 29, 49 (targets).
K grid for maskedkd, tview, tcache: 6, 12, 16, 20, 24, 29, 39, 49.
Students (cub, seed 1, same runs as H-A196): full@196 (primary), maskedkd@98 (record). Teacher rules do not depend on
the student; their KL is computed once.
Per image: KL(full teacher on the view || keep-only masked teacher on the same view), T=1, mean over the V=2 views
(H-A181 draws).
P1: KL_tview@20 - KL_lateroll@29 (aug, full student).
P2: KL_tcache@29 - KL_lateroll@29 (aug, full student).
Record: iso-KL budgets of tview/tcache/maskedkd vs lateroll@12/29/49, all K differences, masked-teacher accuracy,
clean views, maskedkd98 student. Check: full student lateroll aug KL 12/29/49 = 2.5129/0.8575/0.3739 and maskedkd iso
budget vs lateroll@29 = 44.53 (H-A196).
CI: 95% percentile bootstrap over val images (numpy default_rng(20261227), 2000 resamples, shared index).
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on CUB-200-2011 crop+flip views of val, how many teacher tokens do teacher-side rollout scores (on the "
                "view = oracle, cached clean map warped to the view = cost-free) need to match the masked-teacher KL "
                "that student block 6-11 rollout reaches with 29 tokens?",
    "method": "keep-only masked-teacher KL on RandomResizedCrop+flip views over a token-budget grid, iso-KL budget by "
              "linear interpolation, paired image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Szegedy et al. 2015 "
             "(RandomResizedCrop); Efron & Tibshirani 1993; Wah et al. 2011 (CUB)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261227
V = 2
IMG = 224
G = 14
KS = (6, 12, 16, 20, 24, 29, 39, 49)
LR_KS = (12, 29, 49)
STUDENTS = {
    "full": "cub_final_full_k196_s1_1030_153055_d87b",
    "maskedkd98": "cub_final_maskedkd_k98_s1_1030_153055_cdbc",
}
T_RULES = ("tview", "tcache")
S_CELLS = [("maskedkd", k) for k in KS] + [("lateroll", k) for k in LR_KS]
T_CELLS = [(r, k) for r in T_RULES for k in KS]


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


def _iso(curve, target):
    """curve: (B, len(KS)) mean KL per K; target: (B,). First K where curve <= target, linear interpolation."""
    ks = numpy.asarray(KS, dtype=numpy.float64)
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
    tdepth = len(Tm.blocks)
    xs, ys = [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
    y_all = torch.cat(ys)
    n = len(y_all)
    yn = y_all.cpu().numpy().astype(int)
    th = _grids(n, y_all.device)
    vts = ["clean"] + list(range(V))

    # per batch: list of (view tag, view image); teacher logits; teacher-rule masked logits
    views, tl = [], {vt: [] for vt in vts}
    tg = {(vt, c): [] for vt in vts for c in T_CELLS}
    off = 0
    for x in xs:
        b = len(x)
        with torch.no_grad():
            out = api.student_forward(Tm, x, attn_layers=list(range(tdepth)))
            tm = _rollout([out.attn[i] for i in range(tdepth)]).float()
        tmap = (tm / tm.sum(1, keepdim=True)).reshape(b, 1, G, G)
        vb = [("clean", x, tm)]
        for v in range(V):
            grid = F.affine_grid(th[v, off:off + b], (b, 3, IMG, IMG), align_corners=False)
            xa = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            ma = F.grid_sample(tmap.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            vb.append((v, xa, F.adaptive_avg_pool2d(ma, G).flatten(1)))
        for vt, xv, cache in vb:
            with torch.no_grad():
                tl[vt].append(Tm(xv).float().cpu())
                if vt == "clean":
                    own = tm
                else:
                    o = api.student_forward(Tm, xv, attn_layers=list(range(tdepth)))
                    own = _rollout([o.attn[i] for i in range(tdepth)]).float()
                sc = {"tview": own, "tcache": cache}
                for c in T_CELLS:
                    m = _topk_mask(sc[c[0]], c[1])
                    tg[(vt, c)].append(api.teacher_forward_keepmask(Tm, xv, m).float().cpu())
        views.append([(vt, xv) for vt, xv, _ in vb])
        off += b
    tlp = {vt: _logp(torch.cat(tl[vt]).numpy().astype(numpy.float64)) for vt in vts}
    full_acc = {"clean": round(float((tlp["clean"].argmax(1) == yn).mean() * 100), 3),
                "aug": round(float(numpy.mean([(tlp[v].argmax(1) == yn).mean() for v in range(V)]) * 100), 3)}

    def _kl_acc(store, key_fn, cells):
        kl, acc = {}, {}
        for tag, vt_set in (("clean", ["clean"]), ("aug", list(range(V)))):
            for c in cells:
                ka, aa = [], []
                for vt in vt_set:
                    glp = _logp(torch.cat(store[(vt, c)]).numpy().astype(numpy.float64))
                    ka.append((numpy.exp(tlp[vt]) * (tlp[vt] - glp)).sum(1))
                    aa.append((glp.argmax(1) == yn).astype(numpy.float64) * 100)
                kl[key_fn(tag, c)] = numpy.mean(ka, 0)
                acc[key_fn(tag, c)] = numpy.mean(aa, 0)
        return kl, acc

    kl, acc = _kl_acc(tg, lambda tag, c: ("teacher", tag) + c, T_CELLS)
    s_acc = {}
    for sname, rid in STUDENTS.items():
        S = api.student(rid)
        depth = len(S.blocks)
        gl = {(vt, c): [] for vt in vts for c in S_CELLS}
        sl = []
        for vb in views:
            for vt, xv in vb:
                with torch.no_grad():
                    if vt == "clean":
                        sl.append(api.student_forward(S, xv).logits.float().cpu())
                    out = api.student_forward(S, xv, attn_layers=list(range(6, depth)))
                    att = [out.attn[i] for i in range(6, depth)]
                    sc = {"maskedkd": att[-1].float().mean(1)[:, 0, 1:], "lateroll": _rollout(att)}
                    for c in S_CELLS:
                        m = _topk_mask(sc[c[0]], c[1])
                        gl[(vt, c)].append(api.teacher_forward_keepmask(Tm, xv, m).float().cpu())
        del S
        torch.cuda.empty_cache()
        s_acc[sname] = round(float((torch.cat(sl).numpy().argmax(1) == yn).mean() * 100), 3)
        k2, a2 = _kl_acc(gl, lambda tag, c, s=sname: (s, tag) + c, S_CELLS)
        kl.update(k2)
        acc.update(a2)

    rng = numpy.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))

    def _ci(bs):
        return [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]

    def boot(dv):
        dv = numpy.asarray(dv, dtype=numpy.float64)
        return {"mean": round(float(dv.mean()), 4), "ci": _ci(dv[idx].mean(1))}

    def curve_key(s, tag, rule, k):
        return ("teacher", tag, rule, k) if rule in T_RULES else (s, tag, rule, k)

    def iso(s, tag, rule, k_target):
        pts = numpy.stack([kl[curve_key(s, tag, rule, k)] for k in KS], 1)
        tgt = kl[(s, tag, "lateroll", k_target)]
        point = _iso(pts.mean(0, keepdims=True), numpy.array([tgt.mean()]))[0]
        bs = _iso(pts[idx].mean(1), tgt[idx].mean(1))
        fin = numpy.isfinite(bs)
        return {"point": round(float(point), 2) if numpy.isfinite(point) else "inf",
                "ci": [round(float(numpy.percentile(bs, q)), 2) if numpy.isfinite(numpy.percentile(bs, q)) else "inf"
                       for q in (2.5, 97.5)],
                "frac_reached_by_49": round(float(fin.mean()), 4),
                "frac_at_grid_min_6": round(float((bs == KS[0]).mean()), 4),
                "token_ratio_point": round(float(point) / k_target, 3) if numpy.isfinite(point) else "inf"}

    p1 = boot(kl[("teacher", "aug", "tview", 20)] - kl[("full", "aug", "lateroll", 29)])
    p2 = boot(kl[("teacher", "aug", "tcache", 29)] - kl[("full", "aug", "lateroll", 29)])
    rec = {"teacher": {}}
    for tag in ("clean", "aug"):
        rec["teacher"][tag] = {
            "kl_mean": {f"{c[0]}@{c[1]}": round(float(kl[("teacher", tag) + c].mean()), 4) for c in T_CELLS},
            "masked_teacher_acc": {f"{c[0]}@{c[1]}": round(float(acc[("teacher", tag) + c].mean()), 3)
                                   for c in T_CELLS},
            "tcache_minus_tview": {f"@{k}": boot(kl[("teacher", tag, "tcache", k)] - kl[("teacher", tag, "tview", k)])
                                   for k in KS},
        }
    for s in STUDENTS:
        rec[s] = {"student_acc": s_acc[s]}
        for tag in ("clean", "aug"):
            rec[s][tag] = {
                "kl_mean": {f"{c[0]}@{c[1]}": round(float(kl[(s, tag) + c].mean()), 4) for c in S_CELLS},
                "masked_teacher_acc": {f"{c[0]}@{c[1]}": round(float(acc[(s, tag) + c].mean()), 3) for c in S_CELLS},
                "iso_budget": {f"{rule}_vs_lateroll@{kt}": iso(s, tag, rule, kt)
                               for rule in ("tview", "tcache", "maskedkd") for kt in LR_KS},
                "kl_diff": {f"{rule}@{k}-lateroll@29": boot(kl[curve_key(s, tag, rule, k)]
                                                            - kl[(s, tag, "lateroll", 29)])
                            for rule in ("tview", "tcache", "maskedkd") for k in KS},
            }
    api.write_json("summary", {
        "n_val": int(n), "views_per_image": V, "full_teacher_acc": full_acc,
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_kl_tview20_minus_lateroll29_aug_full": p1,
        "P2_kl_tcache29_minus_lateroll29_aug_full": p2,
        "record": rec,
        "note": "KL(full teacher on view || keep-only masked teacher on same view), T=1, nats; aug = mean over V "
                "crop+flip views per image (H-A181 draws); tview = teacher all-block rollout on the view (oracle, "
                "needs a full teacher pass); tcache = clean teacher all-block rollout normalised, warped by the crop "
                "grid and 14x14 average-pooled (H-A173); iso_budget = smallest K on the 6..49 grid (linear "
                "interpolation) where mean KL <= lateroll@K_target mean KL, per bootstrap resample, inf if not "
                "reached by K=49, K=6 if already below at K=6; 95% percentile bootstrap over val images (2000 "
                "resamples, shared index); seed-1 students only"})
