"""H-A126 (round 405, analysis only, no training): Aircraft replication of H-A125 (analysis/cub_selector_tempscale.py,
code unchanged except runs, bootstrap seed and text). H-A124 found that on Aircraft at K=29 MaskedKD's student NLL
exceeds Rollout@29 by +0.122 and TrustMask_v1@29 by +0.185 at temperature 1. Every student is under-confident, so part
of that gap may be one global temperature (a cheap post-hoc fix). Does the selector gap survive held-out temperature
scaling (Guo et al. ICML 2017)? Runs from analysis/aircraft_selector_calibration.py (H-A124).

Per run: logits L on Aircraft val (un-augmented, same 21 runs as H-A124). 2-fold by image index parity: fit one temperature
T on the even fold by grid search (400 log-spaced points in [0.1, 10], minimise mean NLL), apply to the odd fold, and
vice versa. Every image thus gets a held-out tempered NLL and tempered max prob. D_i = mean_s (tNLL A_s(i) - tNLL B_s(i)).
Primary P1: mean D for MaskedKD@29 - Rollout@29 (nats), image bootstrap 2000 (temperatures fixed), numpy
default_rng(20261136), 95% quantile CI. P2: mean D for MaskedKD@29 - TrustMask_v1@29, same bootstrap.
Record: every selector@29 - Full and maskedkd - selector tempered NLL/ECE (x100), raw (T=1) gap, fraction of raw gap
left after scaling, fitted T per run and fold; per-seed t CI (df 2).
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3, K=29), does the MaskedKD - Rollout / MaskedKD - TrustMask_v1 val NLL gap "
                "survive held-out 2-fold temperature scaling?",
    "method": "per run, fit one softmax temperature on half of val (index parity), evaluate NLL and 15-bin ECE on the "
              "other half; paired MaskedKD@29 - selector@29 and selector@29 - Full for rollout/trustmask_v1/"
              "trustfilt/tome/random; image bootstrap and seed t CI",
    "paper": "Guo et al. ICML 2017 (temperature scaling, ECE); Müller et al. NeurIPS 2019; Son et al. ECCV 2024 "
             "(MaskedKD); Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
N_BINS = 15
SEED = 20261136
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


def _ece(conf, correct):
    edges = numpy.linspace(0.0, 1.0, N_BINS + 1)
    b = numpy.clip(numpy.digitize(conf, edges[1:-1]), 0, N_BINS - 1)
    sc = numpy.bincount(b, weights=conf, minlength=N_BINS)
    sa = numpy.bincount(b, weights=correct, minlength=N_BINS)
    return float(numpy.abs(sc - sa).sum() / len(conf))


def _nll_conf(L, y, T):
    Z = L / T
    Z = Z - Z.max(1, keepdims=True)
    logp = Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))
    return -logp[numpy.arange(len(y)), y], numpy.exp(logp).max(1)


def _fit_T(L, y):
    losses = [float(_nll_conf(L, y, T)[0].mean()) for T in T_GRID]
    return float(T_GRID[int(numpy.argmin(losses))])


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
    batches = [(x, y) for x, y in api.val_batches(batch=BATCH)]
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    folds = [numpy.arange(n) % 2 == 0, numpy.arange(n) % 2 == 1]

    raw, tnll, tconf, cor, per_run = ({a: {} for a in RUNS} for _ in range(5))
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            ls = []
            for x, _ in batches:
                with torch.no_grad():
                    ls.append(api.student_forward(S, x).logits.float().cpu())
            L = torch.cat(ls).numpy().astype(numpy.float64)
            del S
            raw[a][s], conf1 = _nll_conf(L, y, 1.0)
            cor[a][s] = (L.argmax(1) == y).astype(float)
            tn, tc, Ts = numpy.zeros(n), numpy.zeros(n), []
            for fit, ev in ((folds[0], folds[1]), (folds[1], folds[0])):
                T = _fit_T(L[fit], y[fit])
                Ts.append(round(T, 4))
                tn[ev], tc[ev] = _nll_conf(L[ev], y[ev], T)
            tnll[a][s], tconf[a][s] = tn, tc
            per_run[a][str(s)] = {"T_fit_even_fit_odd": Ts, "nll_raw": round(float(raw[a][s].mean()), 4),
                                  "nll_tempered": round(float(tn.mean()), 4),
                                  "ece_raw_x100": round(100 * _ece(conf1, cor[a][s]), 3),
                                  "ece_tempered_x100": round(100 * _ece(tc, cor[a][s]), 3),
                                  "conf_minus_acc_tempered_x100": round(100 * float(tc.mean() - cor[a][s].mean()), 3),
                                  "acc_measured": round(100 * float(cor[a][s].mean()), 2),
                                  "val_acc_last": runs[rid]["val_acc_last"]}

    seeds = sorted(RUNS["full"])
    rng = numpy.random.default_rng(SEED)
    sub, per_seed, seed_t, verdict = {}, {}, {}, {}
    for name, (A, B) in CONTRASTS.items():
        D = numpy.mean([tnll[A][s] - tnll[B][s] for s in seeds], 0)
        R = numpy.mean([raw[A][s] - raw[B][s] for s in seeds], 0)
        bn, be = [], []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            bn.append(D[idx].mean())
            be.append(numpy.mean([_ece(tconf[A][s][idx], cor[A][s][idx]) - _ece(tconf[B][s][idx], cor[B][s][idx])
                                  for s in seeds]))
        e0 = numpy.mean([_ece(tconf[A][s], cor[A][s]) - _ece(tconf[B][s], cor[B][s]) for s in seeds])
        sub[name] = {"tnll_boot": [round(float(D.mean()), 4), round(float(numpy.quantile(bn, 0.025)), 4),
                                   round(float(numpy.quantile(bn, 0.975)), 4)],
                     "tece_x100_boot": [round(100 * float(e0), 3), round(100 * float(numpy.quantile(be, 0.025)), 3),
                                        round(100 * float(numpy.quantile(be, 0.975)), 3)],
                     "raw_nll_gap": round(float(R.mean()), 4),
                     "frac_raw_gap_left": round(float(D.mean() / R.mean()), 4) if abs(R.mean()) > 1e-9 else None}
        ps = {"tnll": [round(float((tnll[A][s] - tnll[B][s]).mean()), 4) for s in seeds],
              "tece_x100": [round(100 * (_ece(tconf[A][s], cor[A][s]) - _ece(tconf[B][s], cor[B][s])), 3)
                            for s in seeds],
              "acc_x100": [round(100 * float((cor[A][s] - cor[B][s]).mean()), 3) for s in seeds]}
        per_seed[name] = ps
        seed_t[name] = {k: _t95(v) for k, v in ps.items()}
        verdict[name] = {"tnll": _verdict(sub[name]["tnll_boot"]), "tece": _verdict(sub[name]["tece_x100_boot"])}

    verdict["P1_maskedkd_minus_rollout_tempered_nll"] = verdict["P1_mask_minus_rollout"]["tnll"]
    verdict["P2_maskedkd_minus_v1_tempered_nll"] = verdict["P2_mask_minus_v1"]["tnll"]
    api.write_json("summary", {"n_val": int(n), "T_grid": [0.1, 10.0, len(T_GRID)], "subsets": sub,
                               "per_seed": per_seed, "per_seed_t": seed_t, "per_run": per_run, "verdict": verdict})
