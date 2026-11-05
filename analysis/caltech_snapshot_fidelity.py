"""H-A185 (round 449, analysis only, no training): does the Caltech-256 K=29 lateroll - MaskedKD masked-teacher KL gap
exist early in training, or does it appear as the student trains? H-A180~184 used final students only, but during
training the selector acts on the student of that epoch. Snapshots: e5, e10 and final (e20).

Students (dataset caltech256, seed 1, same runs as H-A180): full@196 (primary), maskedkd@98 (record; trained with its
own last-block masks). Rules, K=29: maskedkd (student last-block CLS attention), lateroll (student rollout over blocks
6-11), trustfilt (lateroll restricted to the teacher's top ceil(1.5K) rollout tokens).
Per image: KL(full teacher || keep-only masked teacher) at T=1, masked-teacher correctness. Clean val views.
P1: lateroll - maskedkd mean KL, full@196 student at e5; 95% percentile bootstrap over val images
    (numpy default_rng(20261208), 2000 resamples, one shared resample index for all statistics).
P2: (lateroll - maskedkd at final) - (lateroll - maskedkd at e5), same student, paired per image, same scheme.
Record: e10, final re-measure (must equal H-A180 -> check), trustfilt pairs, maskedkd98 student, masked-teacher acc,
        snapshot student accuracy vs metrics history.
Limit: one seed per student (image-level CI only), clean val views, three snapshots only.
"""
import math

import numpy
import torch

META = {
    "question": "on Caltech-256 val, is the K=29 lateroll - MaskedKD masked-teacher KL gap already present in early "
                "student snapshots (e5, e10) or does it grow during training?",
    "method": "keep-only masked-teacher KL to the full teacher under student-attention token rules at training "
              "snapshots, attention rollout, paired image bootstrap",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); Son et al. "
             "ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD); Efron & Tibshirani 1993 (bootstrap)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261208
K = 29
WIDEN = 1.5
RULES = ("maskedkd", "lateroll", "trustfilt")
EPOCHS = (5, 10, None)
STUDENTS = {
    "full": "caltech256_final_full_k196_s1_1103_123120_d6de",
    "maskedkd98": "caltech256_final_maskedkd_k98_s1_1103_123120_9e43",
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


def _ep(e):
    return "final" if e is None else f"e{e}"


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid]["snapshots"]]
                                for rid in STUDENTS.values()})
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
        for e in EPOCHS:
            S = api.student(rid, epoch=e)
            gl = {r: [] for r in RULES}
            sl = []
            for (x, _), tm in zip(batches, tmap):
                with torch.no_grad():
                    sl.append(api.student_forward(S, x).logits.float().cpu())
                    last = api.student_forward(S, x, attn_layers=[len(S.blocks) - 1]).attn[-1].float().mean(1)[:, 0, 1:]
                    out = api.student_forward(S, x, attn_layers=list(range(6, len(S.blocks))))
                    lr = _rollout([a for a in out.attn if a is not None])
                    masks = {"maskedkd": _topk_mask(last, K), "lateroll": _topk_mask(lr, K),
                             "trustfilt": _filt(lr, tm, K)}
                    for r in RULES:
                        gl[r].append(api.teacher_forward_keepmask(Tm, x, masks[r]).float().cpu())
            del S
            torch.cuda.empty_cache()
            s_acc[(sname, _ep(e))] = round(float((torch.cat(sl).numpy().argmax(1) == y).mean() * 100), 3)
            for r, v in gl.items():
                glp = _logp(torch.cat(v).numpy().astype(numpy.float64))
                kl[(sname, _ep(e), r)] = (tp * (tlp - glp)).sum(1)
                acc[(sname, _ep(e), r)] = (glp.argmax(1) == y).astype(numpy.float64) * 100

    rng = numpy.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))

    def boot(dv):
        dv = numpy.asarray(dv, dtype=numpy.float64)
        bs = dv[idx].mean(1)
        return {"mean": round(float(dv.mean()), 4),
                "ci": [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]}

    def gap(s, ep, a="lateroll", b="maskedkd", src=None):
        src = kl if src is None else src
        return src[(s, ep, a)] - src[(s, ep, b)]

    p1 = boot(gap("full", "e5"))
    p2 = boot(gap("full", "final") - gap("full", "e5"))
    rec = {}
    for s in STUDENTS:
        for e in EPOCHS:
            ep = _ep(e)
            rec[f"{s}@{ep}"] = {
                "student_acc": s_acc[(s, ep)],
                "kl_mean": {r: round(float(kl[(s, ep, r)].mean()), 4) for r in RULES},
                "masked_teacher_acc": {r: round(float(acc[(s, ep, r)].mean()), 3) for r in RULES},
                "kl_diff": {f"{a}-{b}": boot(gap(s, ep, a, b)) for a, b in
                            [("lateroll", "maskedkd"), ("trustfilt", "maskedkd"), ("trustfilt", "lateroll")]},
                "acc_diff_lateroll-maskedkd": boot(gap(s, ep, src=acc)),
            }
        rec[f"{s}_gap_change"] = {
            "final-e5": boot(gap(s, "final") - gap(s, "e5")),
            "final-e10": boot(gap(s, "final") - gap(s, "e10")),
            "e10-e5": boot(gap(s, "e10") - gap(s, "e5")),
            "maskedkd_kl_final-e5": boot(kl[(s, "final", "maskedkd")] - kl[(s, "e5", "maskedkd")]),
            "lateroll_kl_final-e5": boot(kl[(s, "final", "lateroll")] - kl[(s, "e5", "lateroll")]),
        }
    api.write_json("summary", {
        "n_val": int(n), "teacher_acc": round(float((tlp.argmax(1) == y).mean() * 100), 3),
        "student_acc_final_check": {s: [s_acc[(s, "final")], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_lateroll_minus_maskedkd_kl_full_e5_K29": p1,
        "P2_gap_final_minus_gap_e5_full_K29": p2,
        "record": rec,
        "note": "KL(full teacher || keep-only masked teacher), T=1, nats; 95% percentile bootstrap over val images "
                "(2000 resamples, shared index); seed-1 students only; snapshots e5, e10, final(e20)"})
