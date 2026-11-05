"""H-A184 (round 448, analysis only, no training): Caltech-256 split of the masked-teacher KL gap into TCKD and wNCKD.
H-A180/182 found lateroll keeps the masked teacher much closer to the full teacher than MaskedKD at K=29 (KL -1.14) and
nearly equal at K=98 (-0.05). Is the extra MaskedKD loss a loss of true-class probability (TCKD) or of the shape among
the 256 non-target classes (wNCKD = (1 - p_t) * NCKD)? Caltech-256 has many classes, so NCKD could carry more here.

Student: caltech256 seed 1 full@196 (primary; same as H-A180), maskedkd@98 as record.
Rules, K = 29, 98: maskedkd (student last-block CLS attention), lateroll (student rollout over blocks 6-11), trustfilt
(lateroll restricted to the teacher's top ceil(1.5K) rollout tokens). Clean val views, T=1.
Per image: exact split KL(full teacher || keep-only masked teacher) = TCKD + wNCKD (DKD chain rule, Zhao et al. 2022).
P1: lateroll - maskedkd mean TCKD, K=29, full student; P2: same for wNCKD. 95% percentile bootstrap over val images
    (numpy default_rng(20261207), 2000 resamples, same resample indices for P1 and P2).
Record: raw KL gap (must equal H-A180/182 K=29 -1.14 -> check), TCKD share of the gap, K=98, trustfilt, maskedkd98.
Limit: one seed, clean val views, fixed final students, decomposition is descriptive.
"""
import math

import numpy
import torch

META = {
    "question": "on Caltech-256 val, is the lateroll - MaskedKD masked-teacher KL gap a true-class probability loss "
                "(TCKD) or a non-target shape loss (wNCKD)?",
    "method": "exact DKD chain-rule split of keep-only masked-teacher KL into TCKD and (1-p_t)-weighted NCKD, "
              "image bootstrap",
    "paper": "Zhao et al. CVPR 2022 (DKD); Hooker et al. NeurIPS 2019 (keep-only); Abnar & Zuidema ACL 2020 (rollout); "
             "Son et al. ECCV 2024 (MaskedKD); Efron & Tibshirani 1993 (bootstrap)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261207
EPS = 1e-12
KS = (29, 98)
WIDEN = 1.5
RULES = ("maskedkd", "lateroll", "trustfilt")
STUDENTS = {
    "full": "caltech256_final_full_k196_s1_1103_123120_d6de",
    "maskedkd98": "caltech256_final_maskedkd_k98_s1_1103_123120_9e43",
}


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _dkd(tp, sp, y):
    """Exact split KL = TCKD + wNCKD (per image)."""
    n = len(y)
    p, q = numpy.exp(tp), numpy.exp(sp)
    pt, qt = p[numpy.arange(n), y], q[numpy.arange(n), y]
    pt_c, qt_c = numpy.clip(pt, EPS, 1 - EPS), numpy.clip(qt, EPS, 1 - EPS)
    tckd = pt_c * numpy.log(pt_c / qt_c) + (1 - pt_c) * numpy.log((1 - pt_c) / (1 - qt_c))
    mask = numpy.ones_like(p, dtype=bool)
    mask[numpy.arange(n), y] = False
    pn = numpy.where(mask, p, 0.0) / (1 - pt_c)[:, None]
    qn = numpy.where(mask, q, 0.0) / (1 - qt_c)[:, None]
    nckd = (numpy.where(mask, pn * (numpy.log(pn + EPS) - numpy.log(qn + EPS)), 0.0)).sum(1)
    return tckd, (1 - pt_c) * nckd, qt


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


def _filt(s, t, k):
    m = int(math.ceil(WIDEN * k))
    allowed = t.argsort(1, descending=True).argsort(1) < m
    s = s.masked_fill(~allowed, -1.0)
    return s.argsort(1, descending=True).argsort(1) < k


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"]] for rid in STUDENTS.values()})
    Tm = api.teacher.eval()
    batches, tl, tmap = [], [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
            out = api.student_forward(Tm, x, attn_layers=list(range(len(Tm.blocks))))
            tmap.append(_rollout(out.attn).float())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    tlp = _logp(torch.cat(tl).numpy().astype(numpy.float64))
    tp = numpy.exp(tlp)

    img, s_acc = {}, {}
    for sname, rid in STUDENTS.items():
        S = api.student(rid)
        gl = {(r, k): [] for r in RULES for k in KS}
        sl = []
        for (x, _), tm in zip(batches, tmap):
            with torch.no_grad():
                sl.append(api.student_forward(S, x).logits.float().cpu())
                last = api.student_forward(S, x, attn_layers=[len(S.blocks) - 1]).attn[-1].float().mean(1)[:, 0, 1:]
                out = api.student_forward(S, x, attn_layers=list(range(6, len(S.blocks))))
                lr = _rollout([a for a in out.attn if a is not None])
                for k in KS:
                    masks = {"maskedkd": _topk_mask(last, k), "lateroll": _topk_mask(lr, k),
                             "trustfilt": _filt(lr, tm, k)}
                    for r in RULES:
                        gl[(r, k)].append(api.teacher_forward_keepmask(Tm, x, masks[r]).float().cpu())
        del S
        torch.cuda.empty_cache()
        s_acc[sname] = round(float((torch.cat(sl).numpy().argmax(1) == y).mean() * 100), 3)
        for (r, k), v in gl.items():
            glp = _logp(torch.cat(v).numpy().astype(numpy.float64))
            kl = (tp * (tlp - glp)).sum(1)
            tckd, wnckd, qt = _dkd(tlp, glp, y)
            img[(sname, r, k)] = {"kl": kl, "tckd": tckd, "wnckd": wnckd, "p_true": qt,
                                  "acc": (glp.argmax(1) == y).astype(numpy.float64) * 100,
                                  "resid": float(numpy.abs(kl - tckd - wnckd).max())}

    rng = numpy.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))

    def boot(s, a, b, k, f):
        d = img[(s, a, k)][f] - img[(s, b, k)][f]
        bs = d[idx].mean(1)
        return {"mean": round(float(d.mean()), 4),
                "ci": [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]}

    p1 = boot("full", "lateroll", "maskedkd", 29, "tckd")
    p2 = boot("full", "lateroll", "maskedkd", 29, "wnckd")
    raw = boot("full", "lateroll", "maskedkd", 29, "kl")
    rec = {}
    for s in STUDENTS:
        for k in KS:
            rec[f"{s}@K{k}"] = {
                "means": {r: {f: round(float(img[(s, r, k)][f].mean()), 4)
                              for f in ("kl", "tckd", "wnckd", "p_true", "acc")} for r in RULES},
                "split_residual_max": max(img[(s, r, k)]["resid"] for r in RULES),
                "diff": {f"{a}-{b}": {f: boot(s, a, b, k, f) for f in ("kl", "tckd", "wnckd")}
                         for a, b in [("lateroll", "maskedkd"), ("trustfilt", "maskedkd"), ("trustfilt", "lateroll")]},
            }
    api.write_json("summary", {
        "n_val": int(n), "teacher_acc": round(float((tlp.argmax(1) == y).mean() * 100), 3),
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_lateroll_minus_maskedkd_tckd_full_K29": p1,
        "P2_lateroll_minus_maskedkd_wnckd_full_K29": p2,
        "raw_kl_gap_full_K29": raw,
        "tckd_share_of_kl_gap": round(p1["mean"] / raw["mean"], 4) if raw["mean"] != 0 else None,
        "record": rec,
        "note": "KL(full teacher || keep-only masked teacher) = TCKD + wNCKD, T=1, nats; 95% percentile bootstrap "
                "over val images (2000 resamples, shared indices); seed-1 students only"})
