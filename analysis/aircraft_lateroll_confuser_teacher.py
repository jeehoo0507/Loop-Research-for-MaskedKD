"""H-A116 (round 400, analysis only, no training): Aircraft replication of H-A115
(analysis/cub_lateroll_confuser_teacher.py). H-A113/114 found that masking (MaskedKD@29 vs Full@196) lowers how often
a student's nearest other class in feature space is the full teacher's runner-up class (+3.52 [+1.67, +5.38] on
Aircraft). Does the same-cost better selector lateroll@29 (same teacher GFLOPs as MaskedKD@29) keep that transfer?

Per image i (Aircraft val, un-augmented): t2(i) = argmax_{c != y_i} p_T(c | x_i), the clean full teacher's runner-up
class. Per student seed s: n_s(i) = argmax_{c != y_i} cos(f_i, mean_c) on L2-normalized pre-logit features (class
means over val; other classes only). o_s(i) = 1[n_s(i) == t2(i)]. O_i = mean_s (o A_s - o B_s).
Primary P1: mean O for Full - lateroll (x100), image bootstrap 2000, numpy default_rng(20261126). Secondary P2: mean O
for lateroll - MaskedKD. Record: Full - MaskedKD (H-A114 re-measure); per-seed t CI (df 2); logit-level overlap.
Code is H-A114 (analysis/aircraft_feature_confuser_teacher.py) with the CE arm replaced by lateroll and the
masked-teacher record dropped. Images whose class has < 2 val images or whose t2 class has 0 val images are dropped.
Labels enter only the evaluation.
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3), does the same-cost lateroll@29 student keep the full teacher's "
                "runner-up class as its nearest other class in feature space (Full - lateroll, lateroll - MaskedKD@29)?",
    "method": "clean full teacher runner-up class per val image; per student nearest other class-mean (cosine) on "
              "pre-logit features; per-image overlap; paired Full - lateroll, lateroll - MaskedKD, Full - MaskedKD; "
              "image bootstrap and seed t CI",
    "paper": "Hinton, Vinyals & Dean 2015 (KD); Furlanello et al. ICML 2018 (dark knowledge); Son et al. ECCV 2024 "
             "(MaskedKD); Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261126
RUNS = {
    "lateroll": {1: "aircraft_final_lateroll_k29_s1_1103_065756_f0d9",
                 2: "aircraft_final_lateroll_k29_s2_1103_065801_38e1",
                 3: "aircraft_final_lateroll_k29_s3_1103_065805_2130"},
    "maskedkd": {1: "aircraft_final_maskedkd_k29_s1_1102_132227_6542",
                 2: "aircraft_final_maskedkd_k29_s2_1102_180321_52e2",
                 3: "aircraft_final_maskedkd_k29_s3_1102_180336_347e"},
    "full": {1: "aircraft_final_full_k196_s1_1102_123044_ed5d",
             2: "aircraft_final_full_k196_s2_1102_132227_123b",
             3: "aircraft_final_full_k196_s3_1102_132227_69cc"},
}
CONTRASTS = {"resid_loss": ("full", "lateroll"), "late_gain": ("lateroll", "maskedkd"),
             "mask_loss": ("full", "maskedkd")}


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _boot_mean(v, rng):
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
    return [round(100 * float(v.mean()), 3), round(100 * float(numpy.quantile(bs, 0.025)), 3),
            round(100 * float(numpy.quantile(bs, 0.975)), 3)]


def _runner_up(logits, y):
    z = logits.copy()
    z[numpy.arange(len(y)), y] = -numpy.inf
    return z.argmax(1)


def _nearest_other(F, y, n_cls):
    F = F / numpy.linalg.norm(F, axis=1, keepdims=True)
    S = numpy.zeros((n_cls, F.shape[1]))
    numpy.add.at(S, y, F)
    n = numpy.bincount(y, minlength=n_cls).astype(float)
    C = S / numpy.maximum(n, 1)[:, None]
    Cn = C / numpy.maximum(numpy.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    sims = F @ Cn.T
    sims[:, n == 0] = -numpy.inf
    sims[numpy.arange(len(y)), y] = -numpy.inf
    return sims.argmax(1)


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
    T = api.teacher.eval()

    batches, t_logit = [], []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            t_logit.append(api.student_forward(T, x).logits.float().cpu())
        batches.append((x, y))
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    tl = torch.cat(t_logit).numpy().astype(numpy.float64)
    n_cls = tl.shape[1]
    t2 = _runner_up(tl, y)
    t_acc = float((tl.argmax(1) == y).mean())

    near, lrun, acc = {a: {} for a in RUNS}, {a: {} for a in RUNS}, {a: {} for a in RUNS}
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            depth = len(S.blocks)
            fs, ls = [], []
            for x, _ in batches:
                with torch.no_grad():
                    so = api.student_forward(S, x, token_layers=[depth - 1])
                    f = S.forward_head(S.norm(so.tokens[depth - 1]), pre_logits=True)
                fs.append(f.float().cpu())
                ls.append(so.logits.float().cpu())
            F = torch.cat(fs).numpy().astype(numpy.float64)
            L = torch.cat(ls).numpy().astype(numpy.float64)
            near[a][s] = _nearest_other(F, y, n_cls)
            lrun[a][s] = _runner_up(L, y)
            acc[a][str(s)] = {"logit_all_measured": round(100 * float((L.argmax(1) == y).mean()), 2),
                              "val_acc_last": runs[rid]["val_acc_last"]}
            del S

    counts = numpy.bincount(y, minlength=n_cls)
    valid = (counts[y] >= 2) & (counts[t2] >= 1)
    seeds = sorted(RUNS["full"])
    ov = {a: {s: (near[a][s] == t2).astype(float) for s in seeds} for a in RUNS}
    lov = {a: {s: (lrun[a][s] == t2).astype(float) for s in seeds} for a in RUNS}

    rng = numpy.random.default_rng(SEED)
    sub, per_seed, seed_t, verdict = {}, {}, {}, {}
    for name, (A, B) in CONTRASTS.items():
        O = numpy.mean([ov[A][s] - ov[B][s] for s in seeds], 0)[valid]
        LO = numpy.mean([lov[A][s] - lov[B][s] for s in seeds], 0)[valid]
        sub[name] = {"feat_overlap_x100_boot": _boot_mean(O, rng), "logit_overlap_x100_boot": _boot_mean(LO, rng)}
        ps = {"feat_overlap_x100": [round(100 * float((ov[A][s] - ov[B][s])[valid].mean()), 3) for s in seeds],
              "logit_overlap_x100": [round(100 * float((lov[A][s] - lov[B][s])[valid].mean()), 3) for s in seeds]}
        per_seed[name] = ps
        seed_t[name] = {k: _t95(v) for k, v in ps.items()}
        verdict[name] = _verdict(sub[name]["feat_overlap_x100_boot"])

    level = {a: {str(s): {"feat_overlap_x100": round(100 * float(ov[a][s][valid].mean()), 2),
                          "logit_overlap_x100": round(100 * float(lov[a][s][valid].mean()), 2)}
                 for s in seeds} for a in RUNS}

    verdict["P1_full_minus_lateroll_overlap"] = verdict["resid_loss"]
    verdict["P2_lateroll_minus_maskedkd_overlap"] = verdict["late_gain"]
    out = {"n_val": int(len(y)), "n_valid": int(valid.sum()), "n_cls": int(n_cls), "chance_x100": round(
               100.0 / (n_cls - 1), 3), "teacher_val_acc": round(100 * t_acc, 2), "subsets": sub,
           "per_seed": per_seed, "per_seed_t": seed_t, "overlap_level": level, "acc": acc, "verdict": verdict}
    api.write_json("summary", out)
