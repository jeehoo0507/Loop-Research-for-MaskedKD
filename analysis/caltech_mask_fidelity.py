"""H-A180 (round 444, analysis only, no training): does the masked-teacher soft-label fidelity order of the K=29
selector rules seen on CUB/Aircraft/cub_lt (student rollout rules closer to the full teacher than MaskedKD's last-block
CLS attention; H-A127, H-A171, H-A175) carry over to the held-out Caltech-256? No Caltech-256 analysis exists yet.
Only the stage-2 gate runs exist on Caltech-256 (seed 1: ce, full@196, maskedkd@98), so the rules are applied to fixed
trained students, not to each rule's own training-time student.

Students (dataset caltech256, seed 1): full@196 (primary, rule-neutral), maskedkd@98, ce.
Rules per student, K = 29 and 98: maskedkd (student last-block CLS attention), lateroll (student rollout over blocks
6-11), trustfilt (lateroll restricted to the teacher's top ceil(1.5K) rollout tokens), teacher_topk (teacher all-block
rollout; student-independent). Teacher map = teacher all-block rollout recomputed on clean val.
Per image: KL(full teacher || keep-only masked teacher) at T=1, masked-teacher correctness.
P1: lateroll - maskedkd mean KL, K=29, full@196 student; 95% percentile bootstrap over val images
    (numpy default_rng(20261202), 2000 resamples).
P2: trustfilt - maskedkd, same scheme.
Record: K=98 versions, the maskedkd@98 and ce students, teacher_topk, masked-teacher accuracy, student accuracy check.
Limit: one seed per student (image-level CI only), clean val views, fixed students, correlation-free description.
"""
import math

import numpy
import torch

META = {
    "question": "on Caltech-256 val, is the lateroll / trustfilt masked teacher closer to the full teacher than the "
                "MaskedKD (last-block CLS attention) masked teacher at K=29, as on the stage-1 datasets?",
    "method": "keep-only masked-teacher KL to the full teacher under several token rules, attention rollout, "
              "image bootstrap",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); Son et al. "
             "ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD); Efron & Tibshirani 1993 (bootstrap)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261202
KS = (29, 98)
WIDEN = 1.5
RULES = ("maskedkd", "lateroll", "trustfilt", "teacher_topk")
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

    kl, acc, s_acc = {}, {}, {}
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
                             "trustfilt": _filt(lr, tm, k), "teacher_topk": _topk_mask(tm, k)}
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

    rec = {}
    for s in STUDENTS:
        for k in KS:
            rec[f"{s}@K{k}"] = {
                "kl_mean": {r: round(float(kl[(s, r, k)].mean()), 4) for r in RULES},
                "masked_teacher_acc": {r: round(float(acc[(s, r, k)].mean()), 3) for r in RULES},
                "kl_diff": {f"{a}-{b}": d(s, a, b, k) for a, b in
                            [("lateroll", "maskedkd"), ("trustfilt", "maskedkd"), ("trustfilt", "lateroll"),
                             ("teacher_topk", "maskedkd"), ("teacher_topk", "lateroll")]},
                "acc_diff": {f"{a}-{b}": d(s, a, b, k, acc) for a, b in
                             [("lateroll", "maskedkd"), ("trustfilt", "maskedkd")]},
            }
    api.write_json("summary", {
        "n_val": int(n), "teacher_acc": round(float((tlp.argmax(1) == y).mean() * 100), 3),
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_lateroll_minus_maskedkd_kl_full_K29": rec["full@K29"]["kl_diff"]["lateroll-maskedkd"],
        "P2_trustfilt_minus_maskedkd_kl_full_K29": rec["full@K29"]["kl_diff"]["trustfilt-maskedkd"],
        "record": rec,
        "note": "KL(full teacher || keep-only masked teacher), T=1, nats; 95% percentile bootstrap over val images "
                "(2000 resamples); seed-1 students only"})
