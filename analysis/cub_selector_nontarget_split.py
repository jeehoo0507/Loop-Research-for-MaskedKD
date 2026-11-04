"""H-A129 (round 407, analysis only, no training): H-A127 found that on CUB at K=29, after held-out temperature
scaling, the MaskedKD student's non-target class distribution is further from the full teacher's than Rollout@29's
(+0.120 nats) and TrustMask_v1@29's (+0.152). Is that shape loss on the classes the teacher finds confusable (its
top-5 non-target classes), or spread over the remaining long tail of classes (Hinton et al. 2015 dark knowledge;
Zhao et al. CVPR 2022 NCKD; Furlanello et al. ICML 2018)? Temperature code and runs from
analysis/cub_selector_nontarget_shape.py (H-A127), unchanged.

Per run: student logits L on CUB val (un-augmented, same 21 runs as H-A127), 2-fold index-parity temperature as in
H-A127. Teacher: clean full teacher at T=1. Per image, true label removed and renormalised (teacher p, student q);
H = the teacher's top-5 non-target classes, R = the rest. Exact chain rule:
nt = KL(p || q) = mass + head + tail, with mass = binary KL(p(H) || q(H)), head = p(H) KL(p|H || q|H),
tail = p(R) KL(p|R || q|R). D_i = mean_s (A_s(i) - B_s(i)).
Primary P1: mean D of (mass + head) for MaskedKD@29 - Rollout@29 (nats), image bootstrap 2000 (temperatures fixed),
numpy default_rng(20261139), 95% quantile CI. P2: mean D of tail for MaskedKD@29 - Rollout@29, same bootstrap.
Record: the same split for MaskedKD - TrustMask_v1 / TrustFilt / Full and every selector@29 - Full, each of mass /
head / tail alone, the share (mass + head) / nt of the gap, teacher p(H), the nt total (must equal H-A127), seed t CI.
"""
import numpy
import torch

META = {
    "question": "on CUB (final, seeds 1-3, K=29), after held-out temperature scaling, is the MaskedKD - Rollout "
                "non-target shape loss on the teacher's top-5 confusable classes or on the remaining classes?",
    "method": "per run, 2-fold held-out softmax temperature; per image non-target KL(teacher || student) split exactly "
              "by the chain rule into top-5 mass, within-top-5 and within-rest terms; paired MaskedKD@29 - "
              "selector@29 and selector@29 - Full; image bootstrap and seed t CI",
    "paper": "Hinton et al. 2015 (dark knowledge); Zhao et al. CVPR 2022 (DKD, NCKD); Furlanello et al. ICML 2018 "
             "(born-again networks); Guo et al. ICML 2017 (temperature scaling); Son et al. ECCV 2024 (MaskedKD); "
             "Wah et al. 2011 (CUB)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261139
TOPK = 5
T_GRID = numpy.exp(numpy.linspace(numpy.log(0.1), numpy.log(10.0), 400))
RUNS = {
    "maskedkd": {1: "final_maskedkd_k29_s1_1101_104303_b676",
                 2: "final_maskedkd_k29_s2_1101_125958_5b02",
                 3: "final_maskedkd_k29_s3_1101_140937_b00f"},
    "rollout": {1: "final_rollout_k29_s1_1101_104303_3861",
                2: "final_rollout_k29_s2_1101_125958_2ad2",
                3: "final_rollout_k29_s3_1101_140937_8573"},
    "trustmask_v1": {1: "final_trustmask_v1_k29_s1_1103_095834_2518",
                     2: "final_trustmask_v1_k29_s2_1103_095842_a161",
                     3: "final_trustmask_v1_k29_s3_1103_095847_3c94"},
    "trustfilt": {1: "final_trustfilt_k29_s1_1103_095835_0073",
                  2: "final_trustfilt_k29_s2_1103_095842_42fd",
                  3: "final_trustfilt_k29_s3_1103_095847_a7af"},
    "tome": {1: "final_tome_k29_s1_1101_104303_ff82",
             2: "final_tome_k29_s2_1101_125958_ab06",
             3: "final_tome_k29_s3_1101_140937_cdbb"},
    "random": {1: "final_random_k29_s1_1101_104303_9664",
               2: "final_random_k29_s2_1101_125958_9495",
               3: "final_random_k29_s3_1101_140937_24cb"},
    "full": {1: "cub_final_full_k196_s1_1030_153055_d87b",
             2: "final_full_k196_s2_1101_125958_cc04",
             3: "final_full_k196_s3_1101_140937_e8fa"},
}
CONTRASTS = {"P_mask_minus_rollout": ("maskedkd", "rollout"), "mask_minus_v1": ("maskedkd", "trustmask_v1"),
             "mask_minus_trustfilt": ("maskedkd", "trustfilt"), "mask_minus_tome": ("maskedkd", "tome"),
             "mask_minus_random": ("maskedkd", "random")}
for _a in RUNS:
    if _a != "full":
        CONTRASTS[_a + "_minus_full"] = (_a, "full")
KEYS = ("nt", "mass", "head", "tail", "mass_head")


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


def _split(tn, sn, H):
    """per image: non-target KL(p || q) and its exact chain-rule split over H (teacher top-k) and the rest."""
    pt, qs = numpy.exp(tn), numpy.exp(sn)
    lr = numpy.where(pt > 0, pt * (tn - numpy.where(numpy.isfinite(sn), sn, 0.0)), 0.0)
    nt = lr.sum(1)
    pH = (pt * H).sum(1).clip(1e-12, 1 - 1e-12)
    qH = (qs * H).sum(1).clip(1e-12, 1 - 1e-12)
    mass = pH * numpy.log(pH / qH) + (1 - pH) * numpy.log((1 - pH) / (1 - qH))
    head = (lr * H).sum(1) - pH * numpy.log(pH / qH)
    tail = (lr * ~H).sum(1) - (1 - pH) * numpy.log((1 - pH) / (1 - qH))
    return {"nt": nt, "mass": mass, "head": head, "tail": tail, "mass_head": mass + head}


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
    tn = _nt_logp(t_logp, y)
    H = numpy.zeros(tn.shape, dtype=bool)
    H[numpy.arange(n)[:, None], numpy.argsort(-tn, 1)[:, :TOPK]] = True
    pH = (numpy.exp(tn) * H).sum(1)

    temp, per_run = {a: {} for a in RUNS}, {a: {} for a in RUNS}
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
            temp[a][s] = _split(tn, _nt_logp(s_lp_t, y), H)
            per_run[a][str(s)] = {"T_fit_even_fit_odd": Ts,
                                  **{k: round(float(v.mean()), 4) for k, v in temp[a][s].items()},
                                  "chain_rule_residual": round(float(numpy.abs(
                                      temp[a][s]["nt"] - temp[a][s]["mass"] - temp[a][s]["head"]
                                      - temp[a][s]["tail"]).max()), 8),
                                  "acc_measured": round(100 * float((L.argmax(1) == y).mean()), 2),
                                  "val_acc_last": runs[rid]["val_acc_last"]}

    seeds = sorted(RUNS["full"])
    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))
    sub, seed_t, verdict = {}, {}, {}
    for name, (A, B) in CONTRASTS.items():
        sub[name], seed_t[name] = {}, {}
        Ds = {}
        for key in KEYS:
            D = numpy.mean([temp[A][s][key] - temp[B][s][key] for s in seeds], 0)
            Ds[key] = D
            bm = D[idxs].mean(1)
            sub[name][key + "_boot"] = [round(float(D.mean()), 4), round(float(numpy.quantile(bm, 0.025)), 4),
                                        round(float(numpy.quantile(bm, 0.975)), 4)]
            seed_t[name][key] = _t95([float((temp[A][s][key] - temp[B][s][key]).mean()) for s in seeds])
        bs = Ds["mass_head"][idxs].mean(1) / Ds["nt"][idxs].mean(1)
        sub[name]["share_mass_head_of_nt"] = [round(float(Ds["mass_head"].mean() / Ds["nt"].mean()), 4),
                                              round(float(numpy.quantile(bs, 0.025)), 4),
                                              round(float(numpy.quantile(bs, 0.975)), 4)]
        verdict[name] = {"mass_head": _verdict(sub[name]["mass_head_boot"]),
                         "tail": _verdict(sub[name]["tail_boot"])}

    verdict["P1_maskedkd_minus_rollout_top5_part"] = verdict["P_mask_minus_rollout"]["mass_head"]
    verdict["P2_maskedkd_minus_rollout_rest_part"] = verdict["P_mask_minus_rollout"]["tail"]
    api.write_json("summary", {"n_val": int(n), "T_grid": [0.1, 10.0, len(T_GRID)], "topk": TOPK,
                               "teacher_pH_mean": round(float(pH.mean()), 4),
                               "teacher_pH_quartiles": [round(float(q), 4) for q in numpy.quantile(pH, [.25, .5, .75])],
                               "subsets": sub, "per_seed_t": seed_t, "per_run": per_run, "verdict": verdict})
