"""H-A182 (round 446, analysis only, no training): budget curve of the Caltech-256 masked-teacher fidelity gap.
H-A180 measured K=29 (lateroll - maskedkd KL -1.14) and K=98 (-0.047) only. Does the gap keep growing as K shrinks
(K=12), and where is it between (K=49)? And is MaskedKD's last-block CLS attention better than a random keep at all?

Students (dataset caltech256, seed 1, same as H-A180): full@196 (primary), maskedkd@98, ce.
Rules per student, K = 12, 29, 49: maskedkd (student last-block CLS attention), lateroll (student rollout over blocks
6-11), trustfilt (lateroll restricted to the teacher's top ceil(1.5K) rollout tokens), teacher_topk (teacher all-block
rollout), random (uniform random keep, scores from numpy default_rng(20261204), same draw for every student/K order).
Per image: KL(full teacher || keep-only masked teacher) at T=1, masked-teacher correctness. Clean val views.
P1: lateroll - maskedkd mean KL, K=12, full@196 student; 95% percentile bootstrap over val images
    (numpy default_rng(20261205), 2000 resamples).
P2: lateroll - maskedkd, K=49, same scheme.
P3: random - maskedkd, K=29, same scheme.
Record: K=29 re-measure (must equal H-A180 -> check), other students, trustfilt/teacher_topk pairs, accuracy.
Limit: one seed per student (image-level CI only), clean val views, fixed students.
"""
import math

import numpy
import torch

META = {
    "question": "on Caltech-256 val, how does the lateroll - MaskedKD masked-teacher KL gap change with K (12, 29, 49), "
                "and is the MaskedKD masked teacher closer to the full teacher than a random keep?",
    "method": "keep-only masked-teacher KL to the full teacher under several token rules incl. random, attention "
              "rollout, image bootstrap",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); Son et al. "
             "ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD); Efron & Tibshirani 1993 (bootstrap)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261205
RAND_SEED = 20261204
KS = (12, 29, 49)
WIDEN = 1.5
RULES = ("maskedkd", "lateroll", "trustfilt", "teacher_topk", "random")
STUDENTS = {
    "full": "caltech256_final_full_k196_s1_1103_123120_d6de",
    "maskedkd98": "caltech256_final_maskedkd_k98_s1_1103_123120_9e43",
    "ce": "caltech256_final_ce_k0_s1_1103_123120_2db8",
}


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


def _filt(s, t, k):
    m = int(math.ceil(WIDEN * k))
    allowed = t.argsort(1, descending=True).argsort(1) < m
    s = s.masked_fill(~allowed, -1.0)
    return s.argsort(1, descending=True).argsort(1) < k


def _boot(d, rng):
    d = numpy.asarray(d, dtype=numpy.float64)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    bs = d[idx].mean(1)
    return {"mean": round(float(d.mean()), 4),
            "ci": [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]}


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"]] for rid in STUDENTS.values()})
    Tm = api.teacher.eval()
    rrng = numpy.random.default_rng(RAND_SEED)
    batches, tl, tmap, rscore = [], [], [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
            out = api.student_forward(Tm, x, attn_layers=list(range(len(Tm.blocks))))
            tm = _rollout(out.attn).float()
            tmap.append(tm)
            rscore.append(torch.from_numpy(rrng.random(tuple(tm.shape))).to(tm.device))
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    tlp = _logp(torch.cat(tl).numpy().astype(numpy.float64))
    tp = numpy.exp(tlp)

    kl, acc, s_acc = {}, {}, {}
    for sname, rid in STUDENTS.items():
        S = api.student(rid)
        gl = {(r, k): [] for r in RULES for k in KS}
        sl = []
        for (x, _), tm, rs in zip(batches, tmap, rscore):
            with torch.no_grad():
                sl.append(api.student_forward(S, x).logits.float().cpu())
                last = api.student_forward(S, x, attn_layers=[len(S.blocks) - 1]).attn[-1].float().mean(1)[:, 0, 1:]
                out = api.student_forward(S, x, attn_layers=list(range(6, len(S.blocks))))
                lr = _rollout([a for a in out.attn if a is not None])
                for k in KS:
                    masks = {"maskedkd": _topk_mask(last, k), "lateroll": _topk_mask(lr, k),
                             "trustfilt": _filt(lr, tm, k), "teacher_topk": _topk_mask(tm, k),
                             "random": _topk_mask(rs, k)}
                    for r in RULES:
                        gl[(r, k)].append(api.teacher_forward_keepmask(Tm, x, masks[r]).float().cpu())
        del S
        torch.cuda.empty_cache()
        s_acc[sname] = round(float((torch.cat(sl).numpy().argmax(1) == y).mean() * 100), 3)
        for (r, k), v in gl.items():
            glp = _logp(torch.cat(v).numpy().astype(numpy.float64))
            kl[(sname, r, k)] = (tp * (tlp - glp)).sum(1)
            acc[(sname, r, k)] = (glp.argmax(1) == y).astype(numpy.float64) * 100

    rng = numpy.random.default_rng(SEED)

    def d(s, a, b, k, src=None):
        src = kl if src is None else src
        return _boot(src[(s, a, k)] - src[(s, b, k)], rng)

    p1 = d("full", "lateroll", "maskedkd", 12)
    p2 = d("full", "lateroll", "maskedkd", 49)
    p3 = d("full", "random", "maskedkd", 29)
    rec = {}
    for s in STUDENTS:
        for k in KS:
            rec[f"{s}@K{k}"] = {
                "kl_mean": {r: round(float(kl[(s, r, k)].mean()), 4) for r in RULES},
                "masked_teacher_acc": {r: round(float(acc[(s, r, k)].mean()), 3) for r in RULES},
                "kl_diff": {f"{a}-{b}": d(s, a, b, k) for a, b in
                            [("lateroll", "maskedkd"), ("trustfilt", "maskedkd"), ("trustfilt", "lateroll"),
                             ("teacher_topk", "maskedkd"), ("random", "maskedkd"), ("random", "lateroll")]},
                "acc_diff": {f"{a}-{b}": d(s, a, b, k, acc) for a, b in
                             [("lateroll", "maskedkd"), ("random", "maskedkd")]},
            }
    api.write_json("summary", {
        "n_val": int(n), "teacher_acc": round(float((tlp.argmax(1) == y).mean() * 100), 3),
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_lateroll_minus_maskedkd_kl_full_K12": p1,
        "P2_lateroll_minus_maskedkd_kl_full_K49": p2,
        "P3_random_minus_maskedkd_kl_full_K29": p3,
        "record": rec,
        "note": "KL(full teacher || keep-only masked teacher), T=1, nats; 95% percentile bootstrap over val images "
                "(2000 resamples); seed-1 students only"})
