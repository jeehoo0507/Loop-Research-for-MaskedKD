"""H-A187 (round 451, analysis only, no training): on Caltech-256, is the K=29 lateroll - MaskedKD masked-teacher
KL gap (H-A180 -1.14) a position effect? Does a fixed centre@K mask (no student, pure position prior) preserve the
teacher as well as lateroll, and is lateroll's choice more central than MaskedKD's? No Caltech position analysis
exists (CUB spatial check: cub_sink_spatial, H-C11 decomposition).

Students (caltech256, seed 1, same runs as H-A180): full@196 (primary), maskedkd@98 (record).
Rules: maskedkd (last-block CLS attention top-K), lateroll (student rollout over blocks 6-11 top-K, as H-A180~186),
       center (K patches nearest the image centre, fixed for every image), random (numpy default_rng(20261105)).
Per image: KL(full teacher || keep-only masked teacher), T=1; share of the choice inside centre@K; mean normalized
distance to the centre (0 = centre, 1 = corner).
P1: KL_lateroll - KL_center, full student, K=29.
P2: in-centre share lateroll - in-centre share maskedkd, full student, K=29.
CI: 95% percentile bootstrap over val images (numpy default_rng(20261210), 2000 resamples, shared index).
Record: K=12/49, maskedkd98 student, center - maskedkd, center - random, masked-teacher acc, accuracy check.
"""
import math

import numpy
import torch

META = {
    "question": "on Caltech-256 val, does a fixed centre@K mask preserve the teacher as well as lateroll at K=29, "
                "and is lateroll's selection more central than MaskedKD's?",
    "method": "keep-only masked-teacher KL for student-attention, rollout, fixed-centre and random token masks; "
              "centre@K overlap and centre distance of selected tokens; paired image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Hinton et al. 2015 (KD); "
             "Efron & Tibshirani 1993 (bootstrap)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261210
RND_SEED = 20261105
KS = (12, 29, 49)
SIDE = 14
RULES = ("maskedkd", "lateroll", "center", "random")
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


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"]] for rid in STUDENTS.values()})
    Tm = api.teacher.eval()
    batches, tl = [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    tlp = _logp(torch.cat(tl).numpy().astype(numpy.float64))
    tp = numpy.exp(tlp)

    npatch = SIDE * SIDE
    pos = torch.arange(npatch)
    c = (SIDE - 1) / 2
    dist = (((pos // SIDE - c) ** 2 + (pos % SIDE - c) ** 2).float().sqrt() / math.sqrt(2 * c * c))
    gen = numpy.random.default_rng(RND_SEED)
    rnd_all = [torch.from_numpy(gen.random((x.shape[0], npatch))).float() for x, _ in batches]

    kl, acc, inc, dst, s_acc = {}, {}, {}, {}, {}
    for sname, rid in STUDENTS.items():
        S = api.student(rid)
        depth = len(S.blocks)
        gl = {(r, k): [] for r in RULES for k in KS}
        ic = {(r, k): [] for r in RULES for k in KS}
        ds = {(r, k): [] for r in RULES for k in KS}
        sl = []
        for (x, _), rnd in zip(batches, rnd_all):
            B = x.shape[0]
            d = dist.to(x.device).expand(B, npatch)
            with torch.no_grad():
                out = api.student_forward(S, x, attn_layers=list(range(6, depth)))
                sl.append(api.student_forward(S, x).logits.float().cpu())
                last = out.attn[depth - 1].float().mean(1)[:, 0, 1:]
                lr = _rollout([out.attn[i] for i in range(6, depth)])
                cen_score = (-d - pos.to(x.device).float() * 1e-6)
                for k in KS:
                    masks = {"maskedkd": _topk_mask(last, k), "lateroll": _topk_mask(lr, k),
                             "center": _topk_mask(cen_score, k), "random": _topk_mask(rnd.to(x.device), k)}
                    cen = masks["center"]
                    for r in RULES:
                        ic[(r, k)].append(((masks[r] & cen).float().sum(1) / k).cpu())
                        ds[(r, k)].append(((d * masks[r]).sum(1) / k).cpu())
                        gl[(r, k)].append(api.teacher_forward_keepmask(Tm, x, masks[r]).float().cpu())
        del S
        torch.cuda.empty_cache()
        s_acc[sname] = round(float((torch.cat(sl).numpy().argmax(1) == y).mean() * 100), 3)
        for key in gl:
            inc[(sname,) + key] = torch.cat(ic[key]).numpy().astype(numpy.float64)
            dst[(sname,) + key] = torch.cat(ds[key]).numpy().astype(numpy.float64)
            glp = _logp(torch.cat(gl[key]).numpy().astype(numpy.float64))
            kl[(sname,) + key] = (tp * (tlp - glp)).sum(1)
            acc[(sname,) + key] = (glp.argmax(1) == y).astype(numpy.float64) * 100

    rng = numpy.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))

    def boot(dv):
        dv = numpy.asarray(dv, dtype=numpy.float64)
        bs = dv[idx].mean(1)
        return {"mean": round(float(dv.mean()), 4),
                "ci": [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]}

    p1 = boot(kl[("full", "lateroll", 29)] - kl[("full", "center", 29)])
    p2 = boot(inc[("full", "lateroll", 29)] - inc[("full", "maskedkd", 29)])
    rec = {}
    for s in STUDENTS:
        rec[s] = {"student_acc": s_acc[s]}
        for k in KS:
            rec[s][f"K{k}"] = {
                "kl_mean": {r: round(float(kl[(s, r, k)].mean()), 4) for r in RULES},
                "masked_teacher_acc": {r: round(float(acc[(s, r, k)].mean()), 3) for r in RULES},
                "in_center_share": {r: round(float(inc[(s, r, k)].mean()), 4) for r in RULES},
                "mean_center_dist": {r: round(float(dst[(s, r, k)].mean()), 4) for r in RULES},
                "kl_diff": {f"{a}-{b}": boot(kl[(s, a, k)] - kl[(s, b, k)]) for a, b in
                            [("lateroll", "center"), ("center", "maskedkd"), ("lateroll", "maskedkd"),
                             ("center", "random"), ("maskedkd", "random")]},
                "in_center_diff_lateroll-maskedkd": boot(inc[(s, "lateroll", k)] - inc[(s, "maskedkd", k)]),
            }
    api.write_json("summary", {
        "n_val": int(n), "teacher_acc": round(float((tlp.argmax(1) == y).mean() * 100), 3),
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_kl_lateroll_minus_center_full_K29": p1,
        "P2_in_center_share_lateroll_minus_maskedkd_full_K29": p2,
        "record": rec,
        "note": "KL(full teacher || keep-only masked teacher), T=1, nats; center = K patches nearest the 14x14 grid "
                "centre (fixed); random = numpy default_rng(20261105), same draw for both students; 95% percentile "
                "bootstrap over val images (2000 resamples, shared index); seed-1 students only"})
