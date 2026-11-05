"""H-A163 (round 427, analysis only, no training): cub_lt tail-group version of H-A162 (same code; run ids, RNG seed
and the tail restriction differ). H-A161/H-A162 found on CUB and Aircraft that images where a mask turns the masked
teacher wrong are not clearly images where that arm's student is wrong more often. cub_lt tail is where the
maskedkd vs lateroll accuracy gap is largest (H-A67 lateroll - maskedkd tail +5.78). Does the image-level link
appear there?

Runs: cub_lt final K=29 seeds 1-3, arms lateroll and maskedkd (6 runs, ids from H-A101).
Per image i (clean val): tw_a(i) = seed mean of 1[masked teacher of arm a wrong], sw_a(i) = seed mean of 1[student
of arm a wrong]. dT = tw_maskedkd - tw_lateroll, dS = sw_maskedkd - sw_lateroll. Groups fixed on the tail sample
(val group 1): G+ = dT > 0, G0 = dT == 0, G- = dT < 0.
P1: C = mean dS[G+] - mean dS[G0] on tail images (%p), image bootstrap 95% CI (numpy default_rng, 2000).
Record: same contrast on head images and all images. Groups enter only the evaluation, never a mask.
Limit: clean val views (training uses crop+flip, H-A77), final students only; correlation only; no selector.
"""
import math

import numpy
import torch

META = {
    "question": "on cub_lt final K=29 tail images, are images where the maskedkd mask turns the masked teacher wrong "
                "(maskedkd vs lateroll) also images where the maskedkd student is wrong more often?",
    "method": "per-image seed-averaged masked-teacher and student error differences, group contrast inside the tail "
              "group with image bootstrap CI, seed-wise contrast t CI",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); "
             "Son et al. ECCV 2024 (MaskedKD); Kang et al. ICLR 2020 (long-tail decoupling)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261175
K = 29
START = 6
T_CRIT = 4.302653  # t(0.975, df 2)
RUNS = {
    "lateroll": ["cub_lt_final_lateroll_k29_s1_1104_084157_2b15", "cub_lt_final_lateroll_k29_s2_1104_084204_9f09",
                 "cub_lt_final_lateroll_k29_s3_1104_084209_a3f8"],
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
    r = float(numpy.corrcoef(dT, dS)[0, 1])
    return ([round(C, 4), round(float(numpy.nanquantile(bs_c, 0.025)), 4),
             round(float(numpy.nanquantile(bs_c, 0.975)), 4)],
            [round(r, 4), round(float(numpy.nanquantile(bs_r, 0.025)), 4),
             round(float(numpy.nanquantile(bs_r, 0.975)), 4)])


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

    dT = arm_mean(tw, "maskedkd") - arm_mean(tw, "lateroll")
    dS = arm_mean(sw, "maskedkd") - arm_mean(sw, "lateroll")
    rng = numpy.random.default_rng(SEED)
    P1, r_tail = _boot(dT[tail], dS[tail], rng)
    C_head, r_head = _boot(dT[head], dS[head], rng)
    C_all, r_all = _boot(dT, dS, rng)

    seed_c = []
    for a, b in zip(RUNS["maskedkd"], RUNS["lateroll"]):
        seed_c.append(_contrast((tw[a] - tw[b])[tail], (sw[a] - sw[b])[tail]))

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
                         "maskedkd_student_wrong": round(float(arm_mean(sw, "maskedkd")[g].mean()), 4)
                         if g.any() else None}
        return out

    def verdict(c):
        return "positive" if c[1] > 0 else ("negative" if c[2] < 0 else "uncertain")

    api.write_json("summary", {
        "n_val": int(n), "n_tail": int(tail.sum()), "n_head": int(head.sum()), "K": K, "n_runs": len(per_run),
        "P1_tail_contrast_dS_Gplus_minus_Gzero_pp_boot": P1,
        "record": {
            "tail_mean_dT_pp": round(float(dT[tail].mean() * 100), 4),
            "tail_mean_dS_pp": round(float(dS[tail].mean() * 100), 4),
            "head_mean_dT_pp": round(float(dT[head].mean() * 100), 4),
            "head_mean_dS_pp": round(float(dS[head].mean() * 100), 4),
            "pearson_dT_dS_tail_boot": r_tail,
            "head_contrast_pp_boot": C_head, "pearson_dT_dS_head_boot": r_head,
            "all_contrast_pp_boot": C_all, "pearson_dT_dS_all_boot": r_all,
            "tail_seedwise_contrast_pp": [round(c, 4) for c in seed_c],
            "tail_seedwise_contrast_t": _tcrit(seed_c),
            "groups_tail": group_table(tail),
            "groups_head": group_table(head),
        },
        "per_run": per_run,
        "verdict": {"P1": verdict(P1)},
        "note": "CI entries are [mean, low, high]; dT, dS = maskedkd - lateroll error rates (seed mean); "
                "groups by dT sign; tail = val group 1; one rng shared in order tail, head, all"})
