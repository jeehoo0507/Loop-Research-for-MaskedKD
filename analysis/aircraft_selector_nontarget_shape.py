"""H-A128 (round 406, analysis only, no training): Aircraft replication of H-A127
(analysis/cub_selector_nontarget_shape.py, code unchanged except runs, bootstrap seed and text). H-A126 found that on
Aircraft at K=29 the MaskedKD - Rollout@29 and MaskedKD - TrustMask_v1@29 student NLL gaps survive held-out
temperature scaling (+0.056, +0.096). Is the gap that is
left a non-target "shape" loss, i.e. does the student's distribution over the wrong classes stay further from the full
teacher's even after the global temperature is removed (Hinton et al. 2015 dark knowledge; Zhao et al. CVPR 2022 NCKD)?
H-A14 measured NCKD at T=1 only, on CUB lateroll; this adds the held-out temperature and the K=29 selector axis.
Runs from analysis/aircraft_selector_tempscale.py (H-A126).

Per run: student logits L on Aircraft val (un-augmented, same 21 runs as H-A126). 2-fold by image index parity: fit one
temperature T on one fold (grid of 400 log-spaced points in [0.1, 10], minimise mean NLL), apply it on the other fold.
Teacher: clean full teacher at T=1. Per image, with the true label y removed and the rest renormalised:
nt_i = KL(teacher_nt || student_nt at T). D_i = mean_s (nt A_s(i) - nt B_s(i)).
Primary P1: mean D for MaskedKD@29 - Rollout@29 (nats), image bootstrap 2000 (temperatures fixed), numpy
default_rng(20261138), 95% quantile CI. P2: mean D for MaskedKD@29 - TrustMask_v1@29, same bootstrap.
Record: every selector@29 - Full, raw (T=1) nt gap, tempered full KL(teacher || student) gap and its binary target
part, top-5 non-target overlap with the teacher (x100), fitted T per run; per-seed t CI (df 2).
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3, K=29), after held-out 2-fold temperature scaling, is the student's "
                "non-target class distribution further from the full teacher's for MaskedKD than for Rollout / "
                "TrustMask_v1?",
    "method": "per run, fit one softmax temperature on half of val (index parity), apply on the other half; per image "
              "KL(teacher || student) over the non-target classes (true label removed, renormalised); paired "
              "MaskedKD@29 - selector@29 and selector@29 - Full; image bootstrap and seed t CI",
    "paper": "Hinton et al. 2015 (dark knowledge); Zhao et al. CVPR 2022 (DKD, NCKD); Guo et al. ICML 2017 "
             "(temperature scaling); Son et al. ECCV 2024 (MaskedKD); Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261138
TOPK = 5
T_GRID = numpy.exp(numpy.linspace(numpy.log(0.1), numpy.log(10.0), 400))
RUNS = {
    "maskedkd": {1: "aircraft_final_maskedkd_k29_s1_1102_132227_6542",
                 2: "aircraft_final_maskedkd_k29_s2_1102_180321_52e2",
                 3: "aircraft_final_maskedkd_k29_s3_1102_180336_347e"},
    "rollout": {1: "aircraft_final_rollout_k29_s1_1102_132227_c3d7",
                2: "aircraft_final_rollout_k29_s2_1102_180325_20b6",
                3: "aircraft_final_rollout_k29_s3_1102_180341_8606"},
    "trustmask_v1": {1: "aircraft_final_trustmask_v1_k29_s1_1103_083809_92a4",
                     2: "aircraft_final_trustmask_v1_k29_s2_1103_083818_f341",
                     3: "aircraft_final_trustmask_v1_k29_s3_1103_083822_5bcd"},
    "trustfilt": {1: "aircraft_final_trustfilt_k29_s1_1103_074527_357e",
                  2: "aircraft_final_trustfilt_k29_s2_1103_074535_db76",
                  3: "aircraft_final_trustfilt_k29_s3_1103_074540_2203"},
    "tome": {1: "aircraft_final_tome_k29_s1_1102_132227_ccc7",
             2: "aircraft_final_tome_k29_s2_1102_180330_3c69",
             3: "aircraft_final_tome_k29_s3_1102_180345_552f"},
    "random": {1: "aircraft_final_random_k29_s1_1102_132227_7454",
               2: "aircraft_final_random_k29_s2_1103_042255_db43",
               3: "aircraft_final_random_k29_s3_1103_042259_2272"},
    "full": {1: "aircraft_final_full_k196_s1_1102_123044_ed5d",
             2: "aircraft_final_full_k196_s2_1102_132227_123b",
             3: "aircraft_final_full_k196_s3_1102_132227_69cc"},
}
CONTRASTS = {"P1_mask_minus_rollout": ("maskedkd", "rollout"), "P2_mask_minus_v1": ("maskedkd", "trustmask_v1"),
             "mask_minus_trustfilt": ("maskedkd", "trustfilt"), "mask_minus_tome": ("maskedkd", "tome"),
             "mask_minus_random": ("maskedkd", "random")}
for _a in RUNS:
    if _a != "full":
        CONTRASTS[_a + "_minus_full"] = (_a, "full")


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _logp(L, T):
    Z = L / T
    Z = Z - Z.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _nll(L, y, T):
    return -_logp(L, T)[numpy.arange(len(y)), y]


def _fit_T(L, y):
    losses = [float(_nll(L, y, T).mean()) for T in T_GRID]
    return float(T_GRID[int(numpy.argmin(losses))])


def _nt_logp(logp, y):
    """log of the renormalised distribution over the non-target classes (true class set to -inf)."""
    lp = logp.copy()
    lp[numpy.arange(len(y)), y] = -numpy.inf
    m = lp.max(1, keepdims=True)
    return lp - (m + numpy.log(numpy.exp(lp - m).sum(1, keepdims=True)))


def _stats(t_logp, s_logp, y):
    """per image: non-target KL, full KL, binary target KL, top-k non-target overlap (all teacher || student)."""
    r = numpy.arange(len(y))
    tn, sn = _nt_logp(t_logp, y), _nt_logp(s_logp, y)
    pt = numpy.exp(tn)
    nt = numpy.where(pt > 0, pt * (tn - numpy.where(numpy.isfinite(sn), sn, 0.0)), 0.0).sum(1)
    kl = (numpy.exp(t_logp) * (t_logp - s_logp)).sum(1)
    a, b = numpy.exp(t_logp[r, y]).clip(1e-12, 1 - 1e-12), numpy.exp(s_logp[r, y]).clip(1e-12, 1 - 1e-12)
    tc = a * numpy.log(a / b) + (1 - a) * numpy.log((1 - a) / (1 - b))
    tk, sk = numpy.argsort(-tn, 1)[:, :TOPK], numpy.argsort(-sn, 1)[:, :TOPK]
    ov = numpy.array([len(set(tk[i]) & set(sk[i])) / TOPK for i in r])
    return {"nt": nt, "kl": kl, "tckd": tc, "top5_ov": ov}


def _verdict(ci):
    if ci[1] > 0:
        return "positive"
    if ci[2] < 0:
        return "negative"
    return "uncertain"


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    Tm = api.teacher.eval()
    batches, tl = [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    t_logp = _logp(torch.cat(tl).numpy().astype(numpy.float64), 1.0)
    n = len(y)
    folds = [numpy.arange(n) % 2 == 0, numpy.arange(n) % 2 == 1]

    temp, raw, per_run = ({a: {} for a in RUNS} for _ in range(3))
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            ls = []
            for x, _ in batches:
                with torch.no_grad():
                    ls.append(api.student_forward(S, x).logits.float().cpu())
            L = torch.cat(ls).numpy().astype(numpy.float64)
            del S
            torch.cuda.empty_cache()
            s_lp_t, Ts = numpy.zeros_like(L), []
            for fit, ev in ((folds[0], folds[1]), (folds[1], folds[0])):
                T = _fit_T(L[fit], y[fit])
                Ts.append(round(T, 4))
                s_lp_t[ev] = _logp(L[ev], T)
            temp[a][s] = _stats(t_logp, s_lp_t, y)
            raw[a][s] = _stats(t_logp, _logp(L, 1.0), y)
            per_run[a][str(s)] = {"T_fit_even_fit_odd": Ts,
                                  **{k + "_tempered": round(float(v.mean()), 4) for k, v in temp[a][s].items()},
                                  **{k + "_raw": round(float(v.mean()), 4) for k, v in raw[a][s].items()},
                                  "acc_measured": round(100 * float((L.argmax(1) == y).mean()), 2),
                                  "val_acc_last": runs[rid]["val_acc_last"]}

    seeds = sorted(RUNS["full"])
    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))
    sub, seed_t, verdict = {}, {}, {}
    for name, (A, B) in CONTRASTS.items():
        sub[name], seed_t[name] = {}, {}
        for key in ("nt", "kl", "tckd", "top5_ov"):
            sc = 100.0 if key == "top5_ov" else 1.0
            D = sc * numpy.mean([temp[A][s][key] - temp[B][s][key] for s in seeds], 0)
            R = sc * numpy.mean([raw[A][s][key] - raw[B][s][key] for s in seeds], 0)
            bm = D[idxs].mean(1)
            sub[name][key + "_tempered_boot"] = [round(float(D.mean()), 4), round(float(numpy.quantile(bm, 0.025)), 4),
                                                 round(float(numpy.quantile(bm, 0.975)), 4)]
            sub[name][key + "_raw_gap"] = round(float(R.mean()), 4)
            seed_t[name][key + "_tempered"] = _t95([sc * float((temp[A][s][key] - temp[B][s][key]).mean())
                                                    for s in seeds])
        verdict[name] = _verdict(sub[name]["nt_tempered_boot"])

    verdict["P1_maskedkd_minus_rollout_tempered_nontarget_kl"] = verdict["P1_mask_minus_rollout"]
    verdict["P2_maskedkd_minus_v1_tempered_nontarget_kl"] = verdict["P2_mask_minus_v1"]
    api.write_json("summary", {"n_val": int(n), "T_grid": [0.1, 10.0, len(T_GRID)], "topk": TOPK,
                               "subsets": sub, "per_seed_t": seed_t, "per_run": per_run, "verdict": verdict})
