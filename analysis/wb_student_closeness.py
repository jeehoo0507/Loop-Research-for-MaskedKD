"""H-A178 (round 442, analysis only, no training): Waterbirds version of H-A176 (cub_lt) / H-A177 (Aircraft) student
closeness, read per group. On Waterbirds final K=29 the trustfilt student has a higher WGA than the Full KD student
(72.7 vs 67.4, gate 1 post-hoc), although the Full KD student was trained on the unmasked teacher itself. This asks
whether the trustfilt student is closer to the full teacher than the Full KD student on the worst group (group 2), or
whether its group-2 gain comes without imitating the teacher (farther from it). trustfilt - maskedkd is recorded too
(P2), but it is close to a restatement of the WGA gap (38.6 vs 72.7), as H-A176 was for cub_lt.

Runs: waterbirds final seeds 1-3 of maskedkd, rollout, lateroll, trustmask_v1, trustfilt, teacher_topk (K=29), full@196
and ce. Final students, clean val, groups from api.val_groups(). 2 classes: KL = TCKD (no non-target part).
Per run: KL(full teacher || student) T=1 mean over val and per group, top-1 agreement with the teacher (all, group 2,
teacher-wrong images), student accuracy and worst-group accuracy (must equal metrics val_acc_last / val_wga_last).
P1: trustfilt - full student KL on group 2, seed-paired t 95% CI (df 2).
P2: trustfilt - maskedkd student KL on group 2, seed-paired t 95% CI (df 2).
Record: all-val KL differences; per-group KL; agreement; ce - maskedkd; lateroll - maskedkd; trustfilt - lateroll;
Spearman(group-2 KL, val WGA) over the 18 K=29 runs with one-sided permutation p (numpy default_rng 20261200, 10000);
image bootstrap of P1 over group-2 images (same rng after the permutation, 2000).
Limit: clean val; correlation only; no selector is built from it.
"""
import math

import numpy
import torch

META = {
    "question": "on Waterbirds K=29, is the trustfilt student (higher WGA than Full KD) closer to the full teacher than "
                "the Full KD student on the worst group, or is its worst-group gain not imitation of the teacher?",
    "method": "teacher-student output divergence per group on held-out images, seed-paired t CI, run-level Spearman "
              "with permutation p, image bootstrap",
    "paper": "Hinton et al. 2015 (KD); Stanton et al. NeurIPS 2021 (does KD really work? student-teacher fidelity); "
             "Son et al. ECCV 2024 (MaskedKD); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

BATCH = 50
N_BOOT = 2000
N_PERM = 10000
SEED = 20261200
WORST = 2
T_CRIT = 4.302653  # t(0.975, df 2)
RUNS = {
    "maskedkd": ["waterbirds_final_maskedkd_k29_s1_1031_201502_4be8", "waterbirds_final_maskedkd_k29_s2_1103_170931_92bd",
                 "waterbirds_final_maskedkd_k29_s3_1103_170935_907a"],
    "rollout": ["waterbirds_final_rollout_k29_s1_1031_201502_e135", "waterbirds_final_rollout_k29_s2_1103_170940_2b72",
                "waterbirds_final_rollout_k29_s3_1103_170944_dd2c"],
    "lateroll": ["waterbirds_final_lateroll_k29_s1_1104_025003_b396", "waterbirds_final_lateroll_k29_s2_1104_025008_c94e",
                 "waterbirds_final_lateroll_k29_s3_1104_025012_629b"],
    "trustmask_v1": ["waterbirds_final_trustmask_v1_k29_s1_1103_170855_2e3c",
                     "waterbirds_final_trustmask_v1_k29_s2_1103_170951_279c",
                     "waterbirds_final_trustmask_v1_k29_s3_1103_170956_fdca"],
    "trustfilt": ["waterbirds_final_trustfilt_k29_s1_1103_171121_24be", "waterbirds_final_trustfilt_k29_s2_1103_171126_b33d",
                  "waterbirds_final_trustfilt_k29_s3_1103_171130_db65"],
    "teacher_topk": ["waterbirds_final_teacher_topk_k29_s1_1103_170914_f015",
                     "waterbirds_final_teacher_topk_k29_s2_1103_170918_4535",
                     "waterbirds_final_teacher_topk_k29_s3_1103_170923_d12a"],
    "full": ["final_full_k196_s1_1030_201811_b764", "final_full_k196_s2_1030_212126_a1dc",
             "final_full_k196_s3_1030_220958_8d7e"],
    "ce": ["waterbirds_final_ce_k0_s1_1031_130203_4b89", "waterbirds_final_ce_k0_s2_1031_201229_603c",
           "waterbirds_final_ce_k0_s3_1031_201502_f7a9"],
}
K29 = ("maskedkd", "rollout", "lateroll", "trustmask_v1", "trustfilt", "teacher_topk")
KEYS = ("kl", "kl_g0", "kl_g1", "kl_g2", "kl_g3", "agree", "agree_g2", "agree_twrong", "acc", "wga")


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _tcrit(d):
    d = numpy.asarray(d, dtype=float)
    h = T_CRIT * d.std(ddof=1) / math.sqrt(len(d))
    return [round(float(d.mean()), 4), round(float(d.mean() - h), 4), round(float(d.mean() + h), 4),
            [round(float(v), 4) for v in d]]


def _rank(a):
    return numpy.argsort(numpy.argsort(a)).astype(float)


def _spearman(a, b):
    return float(numpy.corrcoef(_rank(a), _rank(b))[0, 1])


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid].get("val_wga_last")]
                                for arm in RUNS for rid in RUNS[arm]})
    Tm = api.teacher.eval()
    batches, tl = [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    groups = api.val_groups().cpu().numpy()[:n].astype(int)
    gm = {g: groups == g for g in range(4)}
    tlp = _logp(torch.cat(tl).numpy().astype(numpy.float64))
    t_pred = tlp.argmax(1)
    t_wrong = t_pred != y

    per, img = {}, {}
    for arm, ids in RUNS.items():
        for rid in ids:
            S = api.student(rid)
            sl = []
            for x, _ in batches:
                with torch.no_grad():
                    sl.append(api.student_forward(S, x).logits.float().cpu())
            del S
            torch.cuda.empty_cache()
            slp = _logp(torch.cat(sl).numpy().astype(numpy.float64))
            kl = (numpy.exp(tlp) * (tlp - slp)).sum(1)
            s_pred = slp.argmax(1)
            agree = (s_pred == t_pred).astype(float)
            correct = (s_pred == y).astype(float)
            img[rid] = kl
            r = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": runs[rid]["val_acc_last"],
                 "val_wga_last": runs[rid].get("val_wga_last"), "kl": float(kl.mean()),
                 "agree": 100 * float(agree.mean()), "agree_g2": 100 * float(agree[gm[WORST]].mean()),
                 "agree_twrong": 100 * float(agree[t_wrong].mean()), "acc": 100 * float(correct.mean()),
                 "wga": 100 * float(min(correct[gm[g]].mean() for g in range(4) if gm[g].any()))}
            for g in range(4):
                r[f"kl_g{g}"] = float(kl[gm[g]].mean())
            per[rid] = r

    def paired(a, b, k):
        return _tcrit([per[x][k] - per[z][k] for x, z in zip(RUNS[a], RUNS[b])])

    P1 = paired("trustfilt", "full", "kl_g2")
    P2 = paired("trustfilt", "maskedkd", "kl_g2")

    def verdict(c):
        return "positive" if c[1] > 0 else ("negative" if c[2] < 0 else "uncertain")

    rng = numpy.random.default_rng(SEED)
    ids29 = [r for a in K29 for r in RUNS[a]]
    klg2 = numpy.array([per[r]["kl_g2"] for r in ids29])
    wgas = numpy.array([per[r]["wga"] for r in ids29])
    rho = _spearman(klg2, wgas)
    perm = numpy.array([_spearman(klg2, rng.permutation(wgas)) for _ in range(N_PERM)])
    g2 = gm[WORST]
    d = numpy.mean([img[x][g2] - img[z][g2] for x, z in zip(RUNS["trustfilt"], RUNS["full"])], axis=0)
    m = len(d)
    bs = numpy.array([d[rng.integers(0, m, m)].mean() for _ in range(N_BOOT)])

    record = {"P1_boot_g2": [round(float(d.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
                             round(float(numpy.quantile(bs, 0.975)), 4)],
              "spearman_klg2_vs_wga_18runs": [round(rho, 4), round(float((perm <= rho).mean()), 4),
                                              round(float((perm >= rho).mean()), 4)]}
    for a, b in (("trustfilt", "full"), ("trustfilt", "maskedkd"), ("trustfilt", "lateroll"), ("lateroll", "maskedkd"),
                 ("trustmask_v1", "maskedkd"), ("teacher_topk", "maskedkd"), ("rollout", "maskedkd"),
                 ("full", "maskedkd"), ("full", "lateroll"), ("ce", "maskedkd"), ("ce", "full")):
        for k in KEYS:
            if (a, b) in (("trustfilt", "full"), ("trustfilt", "maskedkd")) and k == "kl_g2":
                continue
            record[f"{a}_minus_{b}_{k}_seed_t"] = paired(a, b, k)
    record["arm_means"] = {a: {k: round(float(numpy.mean([per[r][k] for r in RUNS[a]])), 4) for k in KEYS}
                           for a in RUNS}

    api.write_json("summary", {
        "n_val": int(n), "n_per_group": {g: int(gm[g].sum()) for g in range(4)},
        "n_teacher_wrong": int(t_wrong.sum()), "teacher_acc": round(100 * float((~t_wrong).mean()), 2),
        "teacher_acc_per_group": {g: round(100 * float((~t_wrong[gm[g]]).mean()), 2) for g in range(4)},
        "n_runs": len(per),
        "P1_trustfilt_minus_full_student_kl_g2_seed_t": P1,
        "P2_trustfilt_minus_maskedkd_student_kl_g2_seed_t": P2,
        "verdict": {"P1": verdict(P1), "P2": verdict(P2)},
        "record": record,
        "per_run": {r: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in per[r].items()} for r in per},
        "note": "CI entries are [mean, low, high(, per-seed diffs s1..s3)]; KL = KL(full teacher || student) T=1 (= TCKD, "
                "2 classes); agree in %; acc/wga must equal metrics val_acc_last/val_wga_last; spearman entry = "
                "[rho, p_lower, p_upper]; one rng: permutation then P1_boot_g2"})
