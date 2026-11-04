"""H-A161 (round 425, analysis only, no training): H-A160 showed the teacher-area filter makes the masked teacher
lose true-class probability (CUB masked-teacher accuracy -11.7 %p vs lateroll), yet trustfilt run accuracy is not
lower than lateroll beyond seed noise. Is the link also weak at the image level? On images where the filter turns
the masked teacher wrong (trustfilt masked teacher wrong more often than lateroll's, seed-averaged), is the trustfilt
student also wrong more often than the lateroll student, compared with images where the two masked teachers agree
in correctness?

Runs: CUB final K=29 seeds 1-3, arms lateroll and trustfilt (6 runs, ids from H-A160).
Per image i (clean val): tw_a(i) = seed mean of 1[masked teacher of arm a wrong], sw_a(i) = seed mean of 1[student
of arm a wrong] (student logits from the plain fused forward). dT = tw_trustfilt - tw_lateroll,
dS = sw_trustfilt - sw_lateroll. Groups fixed on the full sample: G+ = dT > 0, G0 = dT == 0, G- = dT < 0.
P1: C = mean dS[G+] - mean dS[G0] (%p), image bootstrap 95% CI (numpy default_rng, 2000).
Limit: clean val views (training uses crop+flip, H-A77), final students only; correlation only; no selector.
"""
import math

import numpy
import torch

META = {
    "question": "on CUB final K=29 runs, are images where the teacher-area filter turns the masked teacher wrong "
                "(trustfilt vs lateroll) also images where the trustfilt student is wrong more often?",
    "method": "per-image seed-averaged masked-teacher and student error differences, group contrast with image "
              "bootstrap CI, seed-wise contrast t CI",
    "paper": "Zhao et al. CVPR 2022 (decoupled KD); Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & "
             "Zuidema ACL 2020 (attention rollout); Son et al. ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261171
K = 29
WIDEN = 1.5
T_CRIT = 4.302653  # t(0.975, df 2)
RUNS = {
    "lateroll": ["final_lateroll_k29_s1_1101_104303_ef38", "final_lateroll_k29_s2_1101_125958_6470",
                 "final_lateroll_k29_s3_1101_140937_ec98"],
    "trustfilt": ["final_trustfilt_k29_s1_1103_095835_0073", "final_trustfilt_k29_s2_1103_095842_42fd",
                  "final_trustfilt_k29_s3_1103_095847_a7af"],
}


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


def _allowed(t, k):
    m = int(math.ceil(WIDEN * k))
    return t.argsort(1, descending=True).argsort(1) < m


def _filt(s, t, k):
    s = s.masked_fill(~_allowed(t, k), -1.0)
    return s.argsort(1, descending=True).argsort(1) < k


def _tcrit(d):
    d = numpy.asarray(d, dtype=float)
    h = T_CRIT * d.std(ddof=1) / math.sqrt(len(d))
    return [round(float(d.mean()), 4), round(float(d.mean() - h), 4), round(float(d.mean() + h), 4)]


def _contrast(dT, dS):
    gp, g0 = dT > 0, dT == 0
    if gp.sum() == 0 or g0.sum() == 0:
        return float("nan")
    return float((dS[gp].mean() - dS[g0].mean()) * 100)


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                for arm in RUNS for rid in RUNS[arm]})
    Tm = api.teacher.eval()
    batches, tmap, tl = [], [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().argmax(1).cpu())
            out = api.student_forward(Tm, x, attn_layers=list(range(len(Tm.blocks))))
            tmap.append(_rollout(out.attn).float())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    full_wrong = (torch.cat(tl).numpy() != y).astype(float)

    tw, sw, per_run = {}, {}, {}
    for arm, ids in RUNS.items():
        for rid in ids:
            S = api.student(rid)
            gp, sp = [], []
            for (x, _), tm in zip(batches, tmap):
                with torch.no_grad():
                    sp.append(S(x).float().argmax(1).cpu())
                    out = api.student_forward(S, x, attn_layers=list(range(6, len(S.blocks))))
                    lr = _rollout([a for a in out.attn if a is not None])
                    keep = _filt(lr, tm, K) if arm == "trustfilt" else _topk_mask(lr, K)
                    gp.append(api.teacher_forward_keepmask(Tm, x, keep).float().argmax(1).cpu())
            del S
            torch.cuda.empty_cache()
            tw[rid] = (torch.cat(gp).numpy() != y).astype(float)
            sw[rid] = (torch.cat(sp).numpy() != y).astype(float)
            per_run[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": float(runs[rid]["val_acc_last"]),
                            "student_acc_remeasured": round(float((1 - sw[rid].mean()) * 100), 2),
                            "masked_teacher_acc": round(float((1 - tw[rid].mean()) * 100), 2),
                            "p_student_wrong_given_mt_wrong": round(float(sw[rid][tw[rid] == 1].mean()), 4)
                            if (tw[rid] == 1).any() else None,
                            "p_student_wrong_given_mt_right": round(float(sw[rid][tw[rid] == 0].mean()), 4)
                            if (tw[rid] == 0).any() else None}

    def arm_mean(d, arm):
        return numpy.mean([d[r] for r in RUNS[arm]], axis=0)

    dT = arm_mean(tw, "trustfilt") - arm_mean(tw, "lateroll")
    dS = arm_mean(sw, "trustfilt") - arm_mean(sw, "lateroll")
    C = _contrast(dT, dS)
    rng = numpy.random.default_rng(SEED)
    bs_c, bs_r = [], []
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n)
        bs_c.append(_contrast(dT[ix], dS[ix]))
        a, b = dT[ix], dS[ix]
        bs_r.append(float(numpy.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan"))
    bs_c, bs_r = numpy.array(bs_c), numpy.array(bs_r)
    P1 = [round(C, 4), round(float(numpy.nanquantile(bs_c, 0.025)), 4),
          round(float(numpy.nanquantile(bs_c, 0.975)), 4)]
    r = float(numpy.corrcoef(dT, dS)[0, 1])

    seed_c = []
    for a, b in zip(RUNS["trustfilt"], RUNS["lateroll"]):
        seed_c.append(_contrast(tw[a] - tw[b], sw[a] - sw[b]))

    groups = {}
    for name, g in (("G_plus", dT > 0), ("G_zero", dT == 0), ("G_minus", dT < 0)):
        groups[name] = {"n": int(g.sum()),
                        "mean_dS_pp": round(float(dS[g].mean() * 100), 4) if g.any() else None,
                        "mean_dT_pp": round(float(dT[g].mean() * 100), 4) if g.any() else None,
                        "full_teacher_wrong_rate": round(float(full_wrong[g].mean()), 4) if g.any() else None,
                        "lateroll_student_wrong": round(float(arm_mean(sw, "lateroll")[g].mean()), 4)
                        if g.any() else None,
                        "trustfilt_student_wrong": round(float(arm_mean(sw, "trustfilt")[g].mean()), 4)
                        if g.any() else None}

    def verdict(c):
        return "positive" if c[1] > 0 else ("negative" if c[2] < 0 else "uncertain")

    api.write_json("summary", {
        "n_val": int(n), "K": K, "n_runs": len(per_run),
        "P1_contrast_dS_Gplus_minus_Gzero_pp_boot": P1,
        "record": {
            "mean_dT_pp": round(float(dT.mean() * 100), 4), "mean_dS_pp": round(float(dS.mean() * 100), 4),
            "pearson_dT_dS": round(r, 4),
            "pearson_dT_dS_boot": [round(r, 4), round(float(numpy.nanquantile(bs_r, 0.025)), 4),
                                   round(float(numpy.nanquantile(bs_r, 0.975)), 4)],
            "seedwise_contrast_pp": [round(c, 4) for c in seed_c],
            "seedwise_contrast_t": _tcrit(seed_c),
            "groups": groups,
        },
        "per_run": per_run,
        "verdict": {"P1": verdict(P1)},
        "note": "CI entries are [mean, low, high]; dT, dS = trustfilt - lateroll error rates (seed mean); "
                "groups fixed on the full sample by dT sign; seed-wise contrast pairs seed s of the two arms"})
