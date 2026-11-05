"""H-A177 (round 441, analysis only, no training): Aircraft version of H-A176 (cub_lt student closeness).
H-A176 found the cub_lt K=29 trustfilt student closer to the full teacher than the maskedkd student of the same seed
(KL -0.074), but trustfilt was also +2.3 %p more accurate there, so closeness partly restated accuracy (circular).
On Aircraft final K=29 the two arms are about equally accurate (trustfilt 65.57, maskedkd 65.29 val acc, seed means),
so this asks whether the closeness gap exists without an accuracy gap. The non-target part (wNCKD) is the less
accuracy-bound component and is P2.

Runs: aircraft final seeds 1-3 of maskedkd, rollout, lateroll, trustmask_v1, trustfilt, teacher_topk (K=29), full@196
and ce (references). Final students, clean val (700 images, no groups).
Per run: mean KL(full teacher || student) T=1, exact TCKD + wNCKD split, top-1 agreement with the teacher, agreement on
images the teacher gets wrong, student accuracy (must equal metrics val_acc_last: check).
P1: trustfilt - maskedkd student KL, seed-paired t 95% CI (df 2).
P2: trustfilt - maskedkd student wNCKD, seed-paired t 95% CI (df 2).
Record: TCKD and agreement differences; trustfilt - lateroll, lateroll - maskedkd, trustmask_v1 - maskedkd,
teacher_topk - maskedkd, rollout - maskedkd, full - each, ce - maskedkd; Spearman(student KL, val acc) over the 18 K=29
runs with one-sided permutation p (numpy default_rng 20261190, 10000); image bootstrap of P1 (same rng after the
permutation, 2000).
Limit: clean val; correlation only; no selector is built from it.
"""
import math

import numpy
import torch

META = {
    "question": "on Aircraft K=29, where trustfilt and maskedkd students are about equally accurate, is the trained "
                "trustfilt student still closer to the full teacher (KL, TCKD/wNCKD, top-1 agreement) than the "
                "maskedkd student of the same seed?",
    "method": "teacher-student output divergence on held-out images with exact DKD split, seed-paired t CI, run-level "
              "Spearman with permutation p, image bootstrap",
    "paper": "Hinton et al. 2015 (KD); Zhao et al. CVPR 2022 (decoupled KD); Stanton et al. NeurIPS 2021 (does KD "
             "really work? student-teacher fidelity); Son et al. ECCV 2024 (MaskedKD); Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
N_PERM = 10000
SEED = 20261190
EPS = 1e-12
T_CRIT = 4.302653  # t(0.975, df 2)
RUNS = {
    "maskedkd": ["aircraft_final_maskedkd_k29_s1_1102_132227_6542", "aircraft_final_maskedkd_k29_s2_1102_180321_52e2",
                 "aircraft_final_maskedkd_k29_s3_1102_180336_347e"],
    "rollout": ["aircraft_final_rollout_k29_s1_1102_132227_c3d7", "aircraft_final_rollout_k29_s2_1102_180325_20b6",
                "aircraft_final_rollout_k29_s3_1102_180341_8606"],
    "lateroll": ["aircraft_final_lateroll_k29_s1_1103_065756_f0d9", "aircraft_final_lateroll_k29_s2_1103_065801_38e1",
                 "aircraft_final_lateroll_k29_s3_1103_065805_2130"],
    "trustmask_v1": ["aircraft_final_trustmask_v1_k29_s1_1103_083809_92a4",
                     "aircraft_final_trustmask_v1_k29_s2_1103_083818_f341",
                     "aircraft_final_trustmask_v1_k29_s3_1103_083822_5bcd"],
    "trustfilt": ["aircraft_final_trustfilt_k29_s1_1103_074527_357e", "aircraft_final_trustfilt_k29_s2_1103_074535_db76",
                  "aircraft_final_trustfilt_k29_s3_1103_074540_2203"],
    "teacher_topk": ["aircraft_final_teacher_topk_k29_s1_1103_074554_cd8b",
                     "aircraft_final_teacher_topk_k29_s2_1103_074559_69a5",
                     "aircraft_final_teacher_topk_k29_s3_1103_074603_470b"],
    "full": ["aircraft_final_full_k196_s1_1102_123044_ed5d", "aircraft_final_full_k196_s2_1102_132227_123b",
             "aircraft_final_full_k196_s3_1102_132227_69cc"],
    "ce": ["aircraft_final_ce_k0_s1_1102_123044_6d5f", "aircraft_final_ce_k0_s2_1102_132227_80b7",
           "aircraft_final_ce_k0_s3_1102_132227_55cb"],
}
K29 = ("maskedkd", "rollout", "lateroll", "trustmask_v1", "trustfilt", "teacher_topk")
KEYS = ("kl", "tckd", "wnckd", "agree", "agree_twrong", "acc")


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _dkd(tp, sp, y):
    """Exact split KL = TCKD + wNCKD (per image)."""
    n = len(y)
    p, q = numpy.exp(tp), numpy.exp(sp)
    pt, qt = p[numpy.arange(n), y], q[numpy.arange(n), y]
    pt_c, qt_c = numpy.clip(pt, EPS, 1 - EPS), numpy.clip(qt, EPS, 1 - EPS)
    tckd = pt_c * numpy.log(pt_c / qt_c) + (1 - pt_c) * numpy.log((1 - pt_c) / (1 - qt_c))
    mask = numpy.ones_like(p, dtype=bool)
    mask[numpy.arange(n), y] = False
    pn = numpy.where(mask, p, 0.0) / (1 - pt_c)[:, None]
    qn = numpy.where(mask, q, 0.0) / (1 - qt_c)[:, None]
    nckd = (numpy.where(mask, pn * (numpy.log(pn + EPS) - numpy.log(qn + EPS)), 0.0)).sum(1)
    return tckd, (1 - pt_c) * nckd


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
                                      runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                for arm in RUNS for rid in RUNS[arm]})
    Tm = api.teacher.eval()
    batches, tl = [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
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
            tckd, wnckd = _dkd(tlp, slp, y)
            s_pred = slp.argmax(1)
            agree = (s_pred == t_pred).astype(float)
            correct = (s_pred == y).astype(float)
            img[rid] = kl
            per[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": runs[rid]["val_acc_last"],
                        "kl": float(kl.mean()), "tckd": float(tckd.mean()), "wnckd": float(wnckd.mean()),
                        "agree": 100 * float(agree.mean()), "agree_twrong": 100 * float(agree[t_wrong].mean()),
                        "acc": 100 * float(correct.mean())}

    def paired(a, b, k):
        return _tcrit([per[x][k] - per[z][k] for x, z in zip(RUNS[a], RUNS[b])])

    P1 = paired("trustfilt", "maskedkd", "kl")
    P2 = paired("trustfilt", "maskedkd", "wnckd")

    def verdict(c):
        return "positive" if c[1] > 0 else ("negative" if c[2] < 0 else "uncertain")

    rng = numpy.random.default_rng(SEED)
    ids29 = [r for a in K29 for r in RUNS[a]]
    kls = numpy.array([per[r]["kl"] for r in ids29])
    accs = numpy.array([per[r]["acc"] for r in ids29])
    rho = _spearman(kls, accs)
    perm = numpy.array([_spearman(kls, rng.permutation(accs)) for _ in range(N_PERM)])
    d = numpy.mean([img[x] - img[z] for x, z in zip(RUNS["trustfilt"], RUNS["maskedkd"])], axis=0)
    bs = numpy.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])

    record = {"P1_boot": [round(float(d.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
                          round(float(numpy.quantile(bs, 0.975)), 4)],
              "spearman_kl_vs_acc_18runs": [round(rho, 4), round(float((perm <= rho).mean()), 4),
                                            round(float((perm >= rho).mean()), 4)]}
    for a, b in (("trustfilt", "maskedkd"), ("trustfilt", "lateroll"), ("lateroll", "maskedkd"),
                 ("trustmask_v1", "maskedkd"), ("teacher_topk", "maskedkd"), ("rollout", "maskedkd"),
                 ("full", "maskedkd"), ("full", "trustfilt"), ("full", "lateroll"), ("ce", "maskedkd")):
        for k in KEYS:
            if (a, b) == ("trustfilt", "maskedkd") and k in ("kl", "wnckd"):
                continue
            record[f"{a}_minus_{b}_{k}_seed_t"] = paired(a, b, k)
    record["arm_means"] = {a: {k: round(float(numpy.mean([per[r][k] for r in RUNS[a]])), 4) for k in KEYS}
                           for a in RUNS}

    api.write_json("summary", {
        "n_val": int(n), "n_teacher_wrong": int(t_wrong.sum()),
        "teacher_acc": round(100 * float((~t_wrong).mean()), 2), "n_runs": len(per),
        "P1_trustfilt_minus_maskedkd_student_kl_seed_t": P1,
        "P2_trustfilt_minus_maskedkd_student_wnckd_seed_t": P2,
        "verdict": {"P1": verdict(P1), "P2": verdict(P2)},
        "record": record,
        "per_run": {r: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in per[r].items()} for r in per},
        "note": "CI entries are [mean, low, high(, per-seed diffs s1..s3)]; KL = KL(full teacher || student) T=1; "
                "agree in %; acc must equal metrics val_acc_last; spearman entry = [rho, p_lower, p_upper]; one rng: "
                "permutation then P1_boot"})
