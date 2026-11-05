"""H-A169 (round 433, analysis only, no training): Waterbirds version of H-A164 (same code as H-A168 with the
trustfilt - lateroll pair; run ids, RNG seed and arm differ). H-A168 found the maskedkd - lateroll image-level link
on val group 2 uncertain (C +10.67 [-1.40, +23.65]). On Waterbirds the teacher-area filter is the arm with the
largest worst-group gain over lateroll (H-A57 v0 - lateroll WGA +8.77). Does the filter's change in masked-teacher
correctness on group 2 images line up with the trustfilt student's correctness change there?

Runs: Waterbirds final K=29 seeds 1-3, arms lateroll and trustfilt (6 runs).
Trustfilt mask = lateroll (student rollout from block 6) top-K inside the teacher full-rollout top ceil(1.5 K).
Per image i (clean val): tw_a(i) = seed mean of 1[masked teacher of arm a wrong], sw_a(i) = seed mean of 1[student
of arm a wrong]. dT = tw_trustfilt - tw_lateroll, dS = sw_trustfilt - sw_lateroll. Groups fixed per val group g:
G+ = dT > 0, G0 = dT == 0, G- = dT < 0.
P1: C = mean dS[G+] - mean dS[G0] on val group 2 images (%p), image bootstrap 95% CI (numpy default_rng, 2000).
Record: same contrast in val groups 0, 1, 3 and all images; G- contrast (mean dS[G-] - mean dS[G0]) on group 2.
Groups enter only the evaluation, never a mask.
Limit: clean val views (training uses crop+flip, H-A77), teacher map recomputed on clean val (not the cached map),
final students only; correlation only; no selector.
"""
import math

import numpy
import torch

META = {
    "question": "on Waterbirds final K=29 group-2 images, are images where the teacher-area filter turns the masked "
                "teacher wrong (trustfilt vs lateroll) also images where the trustfilt student is wrong more often?",
    "method": "per-image seed-averaged masked-teacher and student error differences, group contrast inside each val "
              "group with image bootstrap CI, seed-wise contrast t CI",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); "
             "Son et al. ECCV 2024 (MaskedKD); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261183
K = 29
START = 6
WIDEN = 1.5
PRIMARY = 2
T_CRIT = 4.302653  # t(0.975, df 2)
RUNS = {
    "lateroll": ["waterbirds_final_lateroll_k29_s1_1104_025003_b396", "waterbirds_final_lateroll_k29_s2_1104_025008_c94e",
                 "waterbirds_final_lateroll_k29_s3_1104_025012_629b"],
    "trustfilt": ["waterbirds_final_trustfilt_k29_s1_1103_171121_24be",
                  "waterbirds_final_trustfilt_k29_s2_1103_171126_b33d",
                  "waterbirds_final_trustfilt_k29_s3_1103_171130_db65"],
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


def _contrast(dT, dS, neg=False):
    gp, g0 = (dT < 0 if neg else dT > 0), dT == 0
    if gp.sum() == 0 or g0.sum() == 0:
        return float("nan")
    return float((dS[gp].mean() - dS[g0].mean()) * 100)


def _boot(dT, dS, rng):
    n = len(dT)
    bs_c, bs_r = [], []
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n)
        a, b = dT[ix], dS[ix]
        bs_c.append(_contrast(a, b))
        bs_r.append(float(numpy.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan"))
    bs_c, bs_r = numpy.array(bs_c), numpy.array(bs_r)
    C = _contrast(dT, dS)
    r = float(numpy.corrcoef(dT, dS)[0, 1]) if dT.std() > 0 and dS.std() > 0 else float("nan")
    return ([round(C, 4), round(float(numpy.nanquantile(bs_c, 0.025)), 4),
             round(float(numpy.nanquantile(bs_c, 0.975)), 4)],
            [round(r, 4), round(float(numpy.nanquantile(bs_r, 0.025)), 4),
             round(float(numpy.nanquantile(bs_r, 0.975)), 4)])


def _boot_neg(dT, dS, rng):
    n = len(dT)
    bs = numpy.array([_contrast(dT[ix], dS[ix], neg=True) for ix in (rng.integers(0, n, n) for _ in range(N_BOOT))])
    return [round(_contrast(dT, dS, neg=True), 4), round(float(numpy.nanquantile(bs, 0.025)), 4),
            round(float(numpy.nanquantile(bs, 0.975)), 4)]


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid].get("val_wga_last")]
                                for arm in RUNS for rid in RUNS[arm]})
    groups = api.val_groups().cpu().numpy()
    Tm = api.teacher.eval()
    batches, tl, tmap = [], [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().argmax(1).cpu())
            out = api.student_forward(Tm, x, attn_layers=list(range(len(Tm.blocks))))
            tmap.append(_rollout(out.attn).float())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    full_wrong = (torch.cat(tl).numpy() != y).astype(float)
    groups = groups[:n]
    gmask = {g: groups == g for g in range(4)}
    prim = gmask[PRIMARY]

    tw, sw, per_run = {}, {}, {}
    for arm, ids in RUNS.items():
        for rid in ids:
            S = api.student(rid)
            depth = len(S.blocks)
            gp, sp = [], []
            for (x, _), tm in zip(batches, tmap):
                with torch.no_grad():
                    sp.append(S(x).float().argmax(1).cpu())
                    out = api.student_forward(S, x, attn_layers=list(range(START, depth)))
                    lr = _rollout([out.attn[i] for i in range(START, depth)])
                    keep = _filt(lr, tm, K) if arm == "trustfilt" else _topk_mask(lr, K)
                    gp.append(api.teacher_forward_keepmask(Tm, x, keep).float().argmax(1).cpu())
            del S
            torch.cuda.empty_cache()
            tw[rid] = (torch.cat(gp).numpy() != y).astype(float)
            sw[rid] = (torch.cat(sp).numpy() != y).astype(float)
            t1, s1 = tw[rid][prim], sw[rid][prim]
            per_run[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": float(runs[rid]["val_acc_last"]),
                            "val_wga_last": runs[rid].get("val_wga_last"),
                            "student_acc_remeasured": round(float((1 - sw[rid].mean()) * 100), 2),
                            "student_group_acc_remeasured": {g: round(float((1 - sw[rid][gmask[g]].mean()) * 100), 2)
                                                             for g in range(4)},
                            "masked_teacher_acc": round(float((1 - tw[rid].mean()) * 100), 2),
                            "masked_teacher_group_acc": {g: round(float((1 - tw[rid][gmask[g]].mean()) * 100), 2)
                                                         for g in range(4)},
                            "g2_p_student_wrong_given_mt_wrong": round(float(s1[t1 == 1].mean()), 4)
                            if (t1 == 1).any() else None,
                            "g2_p_student_wrong_given_mt_right": round(float(s1[t1 == 0].mean()), 4)
                            if (t1 == 0).any() else None}

    def arm_mean(d, arm):
        return numpy.mean([d[r] for r in RUNS[arm]], axis=0)

    dT = arm_mean(tw, "trustfilt") - arm_mean(tw, "lateroll")
    dS = arm_mean(sw, "trustfilt") - arm_mean(sw, "lateroll")
    rng = numpy.random.default_rng(SEED)
    P1, r_prim = _boot(dT[prim], dS[prim], rng)
    other = {}
    for g in (0, 1, 3):
        other[g] = _boot(dT[gmask[g]], dS[gmask[g]], rng)
    C_all, r_all = _boot(dT, dS, rng)
    neg_prim = _boot_neg(dT[prim], dS[prim], rng)

    seed_c = []
    for a, b in zip(RUNS["trustfilt"], RUNS["lateroll"]):
        seed_c.append(_contrast((tw[a] - tw[b])[prim], (sw[a] - sw[b])[prim]))

    def group_table(m):
        out = {}
        for name, g in (("G_plus", dT > 0), ("G_zero", dT == 0), ("G_minus", dT < 0)):
            g = g & m
            out[name] = {"n": int(g.sum()),
                         "mean_dS_pp": round(float(dS[g].mean() * 100), 4) if g.any() else None,
                         "mean_dT_pp": round(float(dT[g].mean() * 100), 4) if g.any() else None,
                         "full_teacher_wrong_rate": round(float(full_wrong[g].mean()), 4) if g.any() else None,
                         "lateroll_student_wrong": round(float(arm_mean(sw, "lateroll")[g].mean()), 4)
                         if g.any() else None,
                         "trustfilt_student_wrong": round(float(arm_mean(sw, "trustfilt")[g].mean()), 4)
                         if g.any() else None}
        return out

    def verdict(c):
        return "positive" if c[1] > 0 else ("negative" if c[2] < 0 else "uncertain")

    api.write_json("summary", {
        "n_val": int(n), "n_group": {g: int(gmask[g].sum()) for g in range(4)}, "K": K, "n_runs": len(per_run),
        "P1_group2_contrast_dS_Gplus_minus_Gzero_pp_boot": P1,
        "record": {
            "group_mean_dT_pp": {g: round(float(dT[gmask[g]].mean() * 100), 4) for g in range(4)},
            "group_mean_dS_pp": {g: round(float(dS[gmask[g]].mean() * 100), 4) for g in range(4)},
            "pearson_dT_dS_group2_boot": r_prim,
            "contrast_pp_boot_other_groups": {g: other[g][0] for g in other},
            "pearson_dT_dS_other_groups_boot": {g: other[g][1] for g in other},
            "all_contrast_pp_boot": C_all, "pearson_dT_dS_all_boot": r_all,
            "group2_Gminus_contrast_pp_boot": neg_prim,
            "group2_seedwise_contrast_pp": [round(c, 4) for c in seed_c],
            "group2_seedwise_contrast_t": _tcrit(seed_c),
            "groups_table": {g: group_table(gmask[g]) for g in range(4)},
        },
        "per_run": per_run,
        "verdict": {"P1": verdict(P1)},
        "note": "CI entries are [mean, low, high]; dT, dS = trustfilt - lateroll error rates (seed mean); "
                "groups by dT sign; primary = val group 2; one rng shared in order g2, g0, g1, g3, all, g2 G-"})
