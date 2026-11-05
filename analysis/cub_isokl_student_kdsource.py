"""H-A204 (round 468, analysis only, no training): CUB-200-2011, does the student lateroll score beat the teacher's own
block 6-11 rollout because the student was trained with KD?
H-A203: at K=29 on crop+flip views, teacher rollout of blocks 6-11 (tlate) preserves the masked-teacher KL slightly worse
than the full-KD student's rollout of blocks 6-11 (lateroll): +0.099 [+0.053, +0.150]. The full-KD student was trained to
match the teacher on all 196 tokens. This analysis asks whether that KD training is what makes the student score
teacher-aligned, by measuring the same rules on a CE-only student (no teacher signal at all) and on the full-KD student
at epoch 25 (analysis references, not selectors; no new training method):
Students (cub, seed 1): full@196 final (primary, H-A203), ce final (primary), full@196 e25 (record).
Student rules (same definitions as H-A196/H-A203): maskedkd = last-block CLS attention top-K; lateroll = student rollout
blocks 6-11 (residual identity) top-K. Teacher rule: tlate (H-A203), reproduced as a check.
K grid for all rules: 6, 12, 16, 20, 24, 29, 39, 49.
Per image: KL(full teacher on the view || keep-only masked teacher on the same view), T=1, mean over the V=2 views
(H-A181 draws).
P1: KL_lateroll@29(ce) - KL_lateroll@29(full) (aug).
P2: KL_lateroll@29(ce) - KL_tlate@29 (aug).
Record: iso-KL budgets of every curve vs full lateroll@29, maskedkd ce - full, e25 values, masked-teacher accuracy, clean
views. Check: tlate aug KL @29 = 0.9561, full lateroll aug KL @29 = 0.8575, full maskedkd @29 = 1.7313 (H-A203).
CI: 95% percentile bootstrap over val images (numpy default_rng(20261228), 2000 resamples, shared index).
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on CUB-200-2011 crop+flip views of val, does the student block 6-11 rollout preserve the masked-teacher "
                "KL better than the teacher's own block 6-11 rollout only because the student was trained with KD?",
    "method": "keep-only masked-teacher KL on RandomResizedCrop+flip views over a token-budget grid for a CE-only "
              "student, a full-KD student and its epoch-25 snapshot, iso-KL budget by linear interpolation, paired "
              "image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Hinton et al. 2015 (KD); "
             "Efron & Tibshirani 1993; Wah et al. 2011 (CUB)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261228
V = 2
IMG = 224
LATE0 = 6
KS = (6, 12, 16, 20, 24, 29, 39, 49)
STUDENTS = {
    "full": ("cub_final_full_k196_s1_1030_153055_d87b", None),
    "ce": ("cub_final_ce_k0_s1_1030_153055_1256", None),
    "full_e25": ("cub_final_full_k196_s1_1030_153055_d87b", 25),
}
S_RULES = ("maskedkd", "lateroll")
S_CELLS = [(r, k) for r in S_RULES for k in KS]
T_CELLS = [("tlate", k) for k in KS]


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
    api.write_json("students", {s: [rid, ep, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                    runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                for s, (rid, ep) in STUDENTS.items()})
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

    views, tl = [], {vt: [] for vt in vts}
    tg = {(vt, c): [] for vt in vts for c in T_CELLS}
    off = 0
    for x in xs:
        b = len(x)
        vb = [("clean", x)]
        for v in range(V):
            grid = F.affine_grid(th[v, off:off + b], (b, 3, IMG, IMG), align_corners=False)
            vb.append((v, F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros",
                                        align_corners=False)))
        for vt, xv in vb:
            with torch.no_grad():
                tl[vt].append(Tm(xv).float().cpu())
                o = api.student_forward(Tm, xv, attn_layers=list(range(LATE0, tdepth)))
                sc = _rollout([o.attn[i] for i in range(LATE0, tdepth)]).float()
                for c in T_CELLS:
                    m = _topk_mask(sc, c[1])
                    tg[(vt, c)].append(api.teacher_forward_keepmask(Tm, xv, m).float().cpu())
        views.append(vb)
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
    for sname, (rid, ep) in STUDENTS.items():
        S = api.student(rid, epoch=ep)
        depth = len(S.blocks)
        gl = {(vt, c): [] for vt in vts for c in S_CELLS}
        sl = []
        for vb in views:
            for vt, xv in vb:
                with torch.no_grad():
                    if vt == "clean":
                        sl.append(api.student_forward(S, xv).logits.float().cpu())
                    out = api.student_forward(S, xv, attn_layers=list(range(LATE0, depth)))
                    att = [out.attn[i] for i in range(LATE0, depth)]
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

    def iso(owner, tag, rule):
        pts = numpy.stack([kl[(owner, tag, rule, k)] for k in KS], 1)
        tgt = kl[("full", tag, "lateroll", 29)]
        point = _iso(pts.mean(0, keepdims=True), numpy.array([tgt.mean()]))[0]
        bs = _iso(pts[idx].mean(1), tgt[idx].mean(1))
        fin = numpy.isfinite(bs)
        return {"point": round(float(point), 2) if numpy.isfinite(point) else "inf",
                "ci": [round(float(numpy.percentile(bs, q)), 2) if numpy.isfinite(numpy.percentile(bs, q)) else "inf"
                       for q in (2.5, 97.5)],
                "frac_reached_by_49": round(float(fin.mean()), 4),
                "frac_at_grid_min_6": round(float((bs == KS[0]).mean()), 4)}

    p1 = boot(kl[("ce", "aug", "lateroll", 29)] - kl[("full", "aug", "lateroll", 29)])
    p2 = boot(kl[("ce", "aug", "lateroll", 29)] - kl[("teacher", "aug", "tlate", 29)])
    rec = {}
    for tag in ("clean", "aug"):
        r = {"kl_mean": {}, "masked_teacher_acc": {}, "iso_budget_vs_full_lateroll@29": {}, "diff": {}}
        for owner, cells in [("teacher", T_CELLS)] + [(s, S_CELLS) for s in STUDENTS]:
            for c in cells:
                key = f"{owner}:{c[0]}@{c[1]}"
                r["kl_mean"][key] = round(float(kl[(owner, tag) + c].mean()), 4)
                r["masked_teacher_acc"][key] = round(float(acc[(owner, tag) + c].mean()), 3)
            for rule in sorted({c[0] for c in cells}):
                r["iso_budget_vs_full_lateroll@29"][f"{owner}:{rule}"] = iso(owner, tag, rule)
        for k in KS:
            for rule in S_RULES:
                r["diff"][f"ce-full:{rule}@{k}"] = boot(kl[("ce", tag, rule, k)] - kl[("full", tag, rule, k)])
                r["diff"][f"e25-full:{rule}@{k}"] = boot(kl[("full_e25", tag, rule, k)] - kl[("full", tag, rule, k)])
            for s in STUDENTS:
                r["diff"][f"{s}:lateroll-tlate@{k}"] = boot(kl[(s, tag, "lateroll", k)]
                                                            - kl[("teacher", tag, "tlate", k)])
        rec[tag] = r
    api.write_json("summary", {
        "n_val": int(n), "views_per_image": V, "teacher_depth": int(tdepth), "late_blocks": [LATE0, int(tdepth) - 1],
        "full_teacher_acc": full_acc,
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s][0]]["val_acc_last"] if STUDENTS[s][1] is None else None]
                              for s in STUDENTS},
        "P1_kl_lateroll29_ce_minus_full_aug": p1,
        "P2_kl_lateroll29_ce_minus_tlate29_aug": p2,
        "record": rec,
        "note": "KL(full teacher on view || keep-only masked teacher on same view), T=1, nats; aug = mean over V "
                "crop+flip views per image (H-A181 draws); tlate = teacher rollout of blocks 6..depth-1 on the view "
                "(oracle, H-A203); lateroll = student rollout of blocks 6..depth-1; maskedkd = student last-block "
                "CLS attention; ce = seed-1 CE-only student (never saw the teacher); full_e25 = full-KD seed-1 "
                "student at epoch 25; iso_budget = smallest K on the 6..49 grid (linear interpolation) where mean KL "
                "<= full-student lateroll@29 mean KL, per bootstrap resample, inf if not reached by K=49; 95% "
                "percentile bootstrap over val images (2000 resamples, shared index); seed-1 students only"})
