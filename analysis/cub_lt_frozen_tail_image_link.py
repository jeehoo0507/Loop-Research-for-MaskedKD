"""H-A165 (round 429, analysis only, no training): test of the "common cause" reading left open by H-A164.
H-A163: on cub_lt tail, images where the maskedkd rule (the student's own last-block CLS attention top-29) turns the
masked teacher wrong are images where the maskedkd student is wrong more often (C +11.38). H-A164: the teacher-region
filter pair shows no such link (C +0.04). Reading: the maskedkd mask is the student's own attention, so a student that
sees an image badly picks a bad mask and is itself wrong (common cause), whether or not the mask trained it.

Test: the maskedkd_frozen arm was trained with masks from the step-0 student (not reproducible here: the analysis API
has no step-0 student), so its *final* student's own last-block attention never chose its training masks. Apply the
maskedkd rule to the frozen-arm final student and repeat the H-A163 contrast against lateroll. If the link appears
there too, own-attention mask failure tracks student error without any feedback (common cause reading fits).

Runs: cub_lt final K=29 seeds 1-3, arms lateroll, maskedkd_frozen, maskedkd (9 runs).
Per image i (clean val): tw_a(i) = seed mean of 1[masked teacher of arm a wrong], sw_a(i) = seed mean of 1[student
of arm a wrong]. For arm a in {frozen, maskedkd}: dT_a = tw_a - tw_lateroll, dS_a = sw_a - sw_lateroll; groups on the
tail sample: G+ = dT_a > 0, G0 = dT_a == 0. C_a = mean dS_a[G+] - mean dS_a[G0] (%p).
P1: C_frozen on tail images, image bootstrap 95% CI (numpy default_rng, 2000).
Record: C_maskedkd (same code as H-A163, should reproduce +11.38), paired-bootstrap C_maskedkd - C_frozen on the same
resampled tail images, head and all-image contrasts, seed-wise contrasts. Groups enter only the evaluation.
Limit: clean val views (training uses crop+flip, H-A77), final students only; correlation only; no selector.
"""
import math

import numpy
import torch

META = {
    "question": "on cub_lt final K=29 tail images, does the maskedkd rule applied to the maskedkd_frozen student "
                "(whose own attention never chose its training masks) show the same image-level link between "
                "masked-teacher errors and student errors as online maskedkd (H-A163)?",
    "method": "per-image seed-averaged masked-teacher and student error differences vs lateroll, group contrast "
              "inside the tail group with image bootstrap CI, paired bootstrap difference between arms",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); "
             "Son et al. ECCV 2024 (MaskedKD); Arazo et al. IJCNN 2020 (confirmation bias)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261179
K = 29
START = 6
T_CRIT = 4.302653  # t(0.975, df 2)
RUNS = {
    "lateroll": ["cub_lt_final_lateroll_k29_s1_1104_084157_2b15", "cub_lt_final_lateroll_k29_s2_1104_084204_9f09",
                 "cub_lt_final_lateroll_k29_s3_1104_084209_a3f8"],
    "frozen": ["cub_lt_final_maskedkd_frozen_k29_s1_1104_074618_8245",
               "cub_lt_final_maskedkd_frozen_k29_s2_1104_074622_647c",
               "cub_lt_final_maskedkd_frozen_k29_s3_1104_074627_dd2d"],
    "maskedkd": ["cub_lt_final_maskedkd_k29_s1_1103_234609_76d1", "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"],
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


def _ci(point, bs):
    bs = numpy.asarray(bs)
    return [round(point, 4), round(float(numpy.nanquantile(bs, 0.025)), 4),
            round(float(numpy.nanquantile(bs, 0.975)), 4)]


def _boot2(dTf, dSf, dTm, dSm, rng):
    """One shared resample of images per draw: C_frozen, C_maskedkd, their difference, Pearson for frozen."""
    n = len(dTf)
    bf, bm, bd, br = [], [], [], []
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n)
        cf, cm = _contrast(dTf[ix], dSf[ix]), _contrast(dTm[ix], dSm[ix])
        bf.append(cf)
        bm.append(cm)
        bd.append(cm - cf)
        a, b = dTf[ix], dSf[ix]
        br.append(float(numpy.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan"))
    cf, cm = _contrast(dTf, dSf), _contrast(dTm, dSm)
    r = float(numpy.corrcoef(dTf, dSf)[0, 1]) if dTf.std() > 0 and dSf.std() > 0 else float("nan")
    return _ci(cf, bf), _ci(cm, bm), _ci(cm - cf, bd), _ci(r, br)


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid].get("val_wga_last")]
                                for arm in RUNS for rid in RUNS[arm]})
    groups = api.val_groups().cpu().numpy()
    Tm = api.teacher.eval()
    batches, tl = [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().argmax(1).cpu())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    full_wrong = (torch.cat(tl).numpy() != y).astype(float)
    tail, head = groups[:n] == 1, groups[:n] == 0

    tw, sw, per_run = {}, {}, {}
    for arm, ids in RUNS.items():
        for rid in ids:
            S = api.student(rid)
            depth = len(S.blocks)
            gp, sp = [], []
            for x, _ in batches:
                with torch.no_grad():
                    sp.append(S(x).float().argmax(1).cpu())
                    if arm == "lateroll":
                        out = api.student_forward(S, x, attn_layers=list(range(START, depth)))
                        keep = _topk_mask(_rollout([a for a in out.attn if a is not None]), K)
                    else:
                        out = api.student_forward(S, x, attn_layers=[depth - 1])
                        keep = _topk_mask(out.attn[depth - 1].float().mean(1)[:, 0, 1:], K)
                    gp.append(api.teacher_forward_keepmask(Tm, x, keep).float().argmax(1).cpu())
            del S
            torch.cuda.empty_cache()
            tw[rid] = (torch.cat(gp).numpy() != y).astype(float)
            sw[rid] = (torch.cat(sp).numpy() != y).astype(float)
            t1 = tw[rid][tail]
            s1 = sw[rid][tail]
            per_run[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": float(runs[rid]["val_acc_last"]),
                            "val_wga_last": runs[rid].get("val_wga_last"),
                            "student_acc_remeasured": round(float((1 - sw[rid].mean()) * 100), 2),
                            "student_tail_acc_remeasured": round(float((1 - s1.mean()) * 100), 2),
                            "masked_teacher_acc": round(float((1 - tw[rid].mean()) * 100), 2),
                            "masked_teacher_tail_acc": round(float((1 - t1.mean()) * 100), 2),
                            "tail_p_student_wrong_given_mt_wrong": round(float(s1[t1 == 1].mean()), 4)
                            if (t1 == 1).any() else None,
                            "tail_p_student_wrong_given_mt_right": round(float(s1[t1 == 0].mean()), 4)
                            if (t1 == 0).any() else None}

    def arm_mean(d, arm):
        return numpy.mean([d[r] for r in RUNS[arm]], axis=0)

    dTf = arm_mean(tw, "frozen") - arm_mean(tw, "lateroll")
    dSf = arm_mean(sw, "frozen") - arm_mean(sw, "lateroll")
    dTm = arm_mean(tw, "maskedkd") - arm_mean(tw, "lateroll")
    dSm = arm_mean(sw, "maskedkd") - arm_mean(sw, "lateroll")
    rng = numpy.random.default_rng(SEED)
    out = {}
    for name, m in (("tail", tail), ("head", head), ("all", numpy.ones(n, dtype=bool))):
        cf, cm, cd, rf = _boot2(dTf[m], dSf[m], dTm[m], dSm[m], rng)
        out[name] = {"C_frozen_pp_boot": cf, "C_maskedkd_pp_boot": cm, "C_maskedkd_minus_frozen_pp_boot": cd,
                     "pearson_dT_dS_frozen_boot": rf,
                     "mean_dT_frozen_pp": round(float(dTf[m].mean() * 100), 4),
                     "mean_dS_frozen_pp": round(float(dSf[m].mean() * 100), 4),
                     "mean_dT_maskedkd_pp": round(float(dTm[m].mean() * 100), 4),
                     "mean_dS_maskedkd_pp": round(float(dSm[m].mean() * 100), 4)}

    seed_f, seed_m = [], []
    for f, m, b in zip(RUNS["frozen"], RUNS["maskedkd"], RUNS["lateroll"]):
        seed_f.append(_contrast((tw[f] - tw[b])[tail], (sw[f] - sw[b])[tail]))
        seed_m.append(_contrast((tw[m] - tw[b])[tail], (sw[m] - sw[b])[tail]))

    def group_table(dT, dS, arm, m):
        res = {}
        for name, g in (("G_plus", dT > 0), ("G_zero", dT == 0), ("G_minus", dT < 0)):
            g = g & m
            res[name] = {"n": int(g.sum()),
                         "mean_dS_pp": round(float(dS[g].mean() * 100), 4) if g.any() else None,
                         "mean_dT_pp": round(float(dT[g].mean() * 100), 4) if g.any() else None,
                         "full_teacher_wrong_rate": round(float(full_wrong[g].mean()), 4) if g.any() else None,
                         "lateroll_student_wrong": round(float(arm_mean(sw, "lateroll")[g].mean()), 4)
                         if g.any() else None,
                         arm + "_student_wrong": round(float(arm_mean(sw, arm)[g].mean()), 4) if g.any() else None}
        return res

    def verdict(c):
        return "positive" if c[1] > 0 else ("negative" if c[2] < 0 else "uncertain")

    P1 = out["tail"]["C_frozen_pp_boot"]
    api.write_json("summary", {
        "n_val": int(n), "n_tail": int(tail.sum()), "n_head": int(head.sum()), "K": K, "n_runs": len(per_run),
        "P1_tail_C_frozen_pp_boot": P1,
        "record": {
            "contrasts": out,
            "tail_seedwise_C_frozen_pp": [round(c, 4) for c in seed_f],
            "tail_seedwise_C_frozen_t": _tcrit(seed_f),
            "tail_seedwise_C_maskedkd_pp": [round(c, 4) for c in seed_m],
            "tail_seedwise_C_maskedkd_t": _tcrit(seed_m),
            "tail_seedwise_C_maskedkd_minus_frozen_t": _tcrit(numpy.array(seed_m) - numpy.array(seed_f)),
            "groups_tail_frozen": group_table(dTf, dSf, "frozen", tail),
            "groups_tail_maskedkd": group_table(dTm, dSm, "maskedkd", tail),
        },
        "per_run": per_run,
        "verdict": {"P1": verdict(P1), "maskedkd_minus_frozen_tail": verdict(out["tail"]["C_maskedkd_minus_frozen_pp_boot"])},
        "note": "CI entries are [point, low, high]; dT, dS = arm - lateroll error rates (seed mean); groups by dT "
                "sign; tail = val group 1; one rng shared in order tail, head, all; frozen mask = maskedkd rule on "
                "the frozen-arm final student (not its training mask)"})
