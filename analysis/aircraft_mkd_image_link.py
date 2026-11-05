"""H-A162 (round 426, analysis only, no training): Aircraft counterpart of cub_mkd_image_link. H-A161 found that images where the teacher-area filter turns the
masked teacher wrong are not clearly images where the trustfilt student is wrong more often. Does the same weak
image-level link hold for the other arm pair at the same cost, maskedkd vs lateroll? On images where the maskedkd
mask (student last-block CLS attention) turns the masked teacher wrong more often than the lateroll mask, is the
maskedkd student also wrong more often than the lateroll student, compared with images where the two masked teachers
agree in correctness?

Runs: Aircraft final K=29 seeds 1-3, arms lateroll and maskedkd (6 runs; lateroll ids from H-A161).
Per image i (clean val): tw_a(i) = seed mean of 1[masked teacher of arm a wrong], sw_a(i) = seed mean of 1[student
of arm a wrong] (student logits from the plain fused forward). dT = tw_maskedkd - tw_lateroll,
dS = sw_maskedkd - sw_lateroll. Groups fixed on the full sample: G+ = dT > 0, G0 = dT == 0, G- = dT < 0.
P1: C = mean dS[G+] - mean dS[G0] (%p), image bootstrap 95% CI (numpy default_rng, 2000).
Limit: clean val views (training uses crop+flip, H-A77), final students only; correlation only; no selector.
"""
import math

import numpy
import torch

META = {
    "question": "on Aircraft final K=29 runs, are images where the maskedkd mask turns the masked teacher wrong "
                "(maskedkd vs lateroll) also images where the maskedkd student is wrong more often?",
    "method": "per-image seed-averaged masked-teacher and student error differences, group contrast with image "
              "bootstrap CI, seed-wise contrast t CI",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); "
             "Son et al. ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261174
K = 29
START = 6
T_CRIT = 4.302653  # t(0.975, df 2)
RUNS = {
    "lateroll": ["aircraft_final_lateroll_k29_s1_1103_065756_f0d9", "aircraft_final_lateroll_k29_s2_1103_065801_38e1",
                 "aircraft_final_lateroll_k29_s3_1103_065805_2130"],
    "maskedkd": ["aircraft_final_maskedkd_k29_s1_1102_132227_6542", "aircraft_final_maskedkd_k29_s2_1102_180321_52e2",
                 "aircraft_final_maskedkd_k29_s3_1102_180336_347e"],
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
    batches, tl = [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().argmax(1).cpu())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    full_wrong = (torch.cat(tl).numpy() != y).astype(float)

    tw, sw, per_run = {}, {}, {}
    for arm, ids in RUNS.items():
        for rid in ids:
            S = api.student(rid)
            depth = len(S.blocks)
            gp, sp = [], []
            for x, _ in batches:
                with torch.no_grad():
                    sp.append(S(x).float().argmax(1).cpu())
                    if arm == "maskedkd":
                        out = api.student_forward(S, x, attn_layers=[depth - 1])
                        keep = _topk_mask(out.attn[depth - 1].float().mean(1)[:, 0, 1:], K)
                    else:
                        out = api.student_forward(S, x, attn_layers=list(range(START, depth)))
                        keep = _topk_mask(_rollout([a for a in out.attn if a is not None]), K)
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

    dT = arm_mean(tw, "maskedkd") - arm_mean(tw, "lateroll")
    dS = arm_mean(sw, "maskedkd") - arm_mean(sw, "lateroll")
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
    for a, b in zip(RUNS["maskedkd"], RUNS["lateroll"]):
        seed_c.append(_contrast(tw[a] - tw[b], sw[a] - sw[b]))

    groups = {}
    for name, g in (("G_plus", dT > 0), ("G_zero", dT == 0), ("G_minus", dT < 0)):
        groups[name] = {"n": int(g.sum()),
                        "mean_dS_pp": round(float(dS[g].mean() * 100), 4) if g.any() else None,
                        "mean_dT_pp": round(float(dT[g].mean() * 100), 4) if g.any() else None,
                        "full_teacher_wrong_rate": round(float(full_wrong[g].mean()), 4) if g.any() else None,
                        "lateroll_student_wrong": round(float(arm_mean(sw, "lateroll")[g].mean()), 4)
                        if g.any() else None,
                        "maskedkd_student_wrong": round(float(arm_mean(sw, "maskedkd")[g].mean()), 4)
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
        "note": "CI entries are [mean, low, high]; dT, dS = maskedkd - lateroll error rates (seed mean); "
                "groups fixed on the full sample by dT sign; seed-wise contrast pairs seed s of the two arms"})
