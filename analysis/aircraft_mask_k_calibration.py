"""H-A122 (round 403, analysis only, no training): on Aircraft, does MaskedKD's student calibration cost (H-A118: NLL
+0.280 over Full at K=29) shrink as the teacher token budget K grows, and is it gone at K=98 (half the teacher cost)?
This ties the soft-label loss to the FLOPs axis: if NLL cost stays at K=98, the teacher signal is still degraded at
the cheapest budget that keeps top-1 near Full.

Per image i (Aircraft val, un-augmented), per student seed s: NLL_s(i) = -log p_S(y_i | x_i) at temperature 1.
D_i = mean_s (NLL A_s(i) - NLL B_s(i)) over the seeds both arms have. ECE: 15 equal-width bins on max softmax prob.
Primary P1: mean D for MaskedKD@98 - Full@196 (nats), image bootstrap 2000, numpy default_rng(20261132), 95% quantile
CI. P2 (dose): mean D for MaskedKD@12 - MaskedKD@49 (nats), same bootstrap.
Record: every MaskedKD@K - Full (K = 12, 29, 49, 98) NLL and ECE (x100), accuracy gap; per-seed t CI (df 2);
per run NLL, ECE, mean confidence - accuracy, val acc vs metrics; teacher NLL/ECE. Labels enter only the evaluation.
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3), does the MaskedKD student NLL cost over Full@196 shrink with token "
                "budget K, and does it remain at K=98?",
    "method": "per-image NLL of each final student on clean val, 15-bin ECE; paired MaskedKD@K - Full for K in "
              "12/29/49/98 and MaskedKD@12 - MaskedKD@49; image bootstrap and seed t CI",
    "paper": "Guo et al. ICML 2017 (calibration, ECE); Müller et al. NeurIPS 2019 (label smoothing and KD); Son et al. "
             "ECCV 2024 (MaskedKD); Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
N_BINS = 15
SEED = 20261132
RUNS = {
    "k12": {1: "aircraft_final_maskedkd_k12_s1_1102_132227_f01c",
            2: "aircraft_final_maskedkd_k12_s2_1103_024925_555b",
            3: "aircraft_final_maskedkd_k12_s3_1103_024931_458c"},
    "k29": {1: "aircraft_final_maskedkd_k29_s1_1102_132227_6542",
            2: "aircraft_final_maskedkd_k29_s2_1102_180321_52e2",
            3: "aircraft_final_maskedkd_k29_s3_1102_180336_347e"},
    "k49": {1: "aircraft_final_maskedkd_k49_s1_1102_193509_429e",
            2: "aircraft_final_maskedkd_k49_s2_1102_193516_9d71",
            3: "aircraft_final_maskedkd_k49_s3_1102_193521_573d"},
    "k98": {1: "aircraft_final_maskedkd_k98_s1_1102_123044_bd9b",
            2: "aircraft_final_maskedkd_k98_s2_1102_132227_2f10",
            3: "aircraft_final_maskedkd_k98_s3_1102_132227_c979"},
    "full": {1: "aircraft_final_full_k196_s1_1102_123044_ed5d",
             2: "aircraft_final_full_k196_s2_1102_132227_123b",
             3: "aircraft_final_full_k196_s3_1102_132227_69cc"},
}
CONTRASTS = {"k12_cost": ("k12", "full"), "k29_cost": ("k29", "full"), "k49_cost": ("k49", "full"),
             "k98_cost": ("k98", "full"), "dose_12_vs_49": ("k12", "k49")}
P1, P2 = "k98_cost", "dose_12_vs_49"


def _t95(d):
    if len(d) < 2:
        return [round(float(numpy.mean(d)), 4), None, None, len(d)]
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _ece(conf, correct):
    edges = numpy.linspace(0.0, 1.0, N_BINS + 1)
    b = numpy.clip(numpy.digitize(conf, edges[1:-1]), 0, N_BINS - 1)
    sc = numpy.bincount(b, weights=conf, minlength=N_BINS)
    sa = numpy.bincount(b, weights=correct, minlength=N_BINS)
    return float(numpy.abs(sc - sa).sum() / len(conf))


def _stats(L, y):
    L = L - L.max(1, keepdims=True)
    logp = L - numpy.log(numpy.exp(L).sum(1, keepdims=True))
    nll = -logp[numpy.arange(len(y)), y]
    p = numpy.exp(logp)
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    return nll, conf, correct


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
    t_nll, t_conf, t_cor = _stats(torch.cat(t_logit).numpy().astype(numpy.float64), y)

    nll, conf, cor, per_run = {a: {} for a in RUNS}, {a: {} for a in RUNS}, {a: {} for a in RUNS}, {a: {} for a in RUNS}
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            ls = []
            for x, _ in batches:
                with torch.no_grad():
                    ls.append(api.student_forward(S, x).logits.float().cpu())
            L = torch.cat(ls).numpy().astype(numpy.float64)
            nll[a][s], conf[a][s], cor[a][s] = _stats(L, y)
            per_run[a][str(s)] = {"nll": round(float(nll[a][s].mean()), 4),
                                  "ece_x100": round(100 * _ece(conf[a][s], cor[a][s]), 3),
                                  "conf_minus_acc_x100": round(100 * float(conf[a][s].mean() - cor[a][s].mean()), 3),
                                  "acc_measured": round(100 * float(cor[a][s].mean()), 2),
                                  "val_acc_last": runs[rid]["val_acc_last"]}
            del S

    n = len(y)
    rng = numpy.random.default_rng(SEED)
    sub, per_seed, seed_t, verdict = {}, {}, {}, {}
    for name, (A, B) in CONTRASTS.items():
        seeds = sorted(set(RUNS[A]) & set(RUNS[B]))
        D = numpy.mean([nll[A][s] - nll[B][s] for s in seeds], 0)
        bn, be = [], []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            bn.append(D[idx].mean())
            be.append(numpy.mean([_ece(conf[A][s][idx], cor[A][s][idx]) - _ece(conf[B][s][idx], cor[B][s][idx])
                                  for s in seeds]))
        e0 = numpy.mean([_ece(conf[A][s], cor[A][s]) - _ece(conf[B][s], cor[B][s]) for s in seeds])
        sub[name] = {"seeds": seeds,
                     "nll_boot": [round(float(D.mean()), 4), round(float(numpy.quantile(bn, 0.025)), 4),
                                  round(float(numpy.quantile(bn, 0.975)), 4)],
                     "ece_x100_boot": [round(100 * float(e0), 3), round(100 * float(numpy.quantile(be, 0.025)), 3),
                                       round(100 * float(numpy.quantile(be, 0.975)), 3)]}
        ps = {"nll": [round(float((nll[A][s] - nll[B][s]).mean()), 4) for s in seeds],
              "ece_x100": [round(100 * (_ece(conf[A][s], cor[A][s]) - _ece(conf[B][s], cor[B][s])), 3)
                           for s in seeds],
              "acc_x100": [round(100 * float((cor[A][s] - cor[B][s]).mean()), 3) for s in seeds]}
        per_seed[name] = ps
        seed_t[name] = {k: _t95(v) for k, v in ps.items()}
        verdict[name] = {"nll": _verdict(sub[name]["nll_boot"]), "ece": _verdict(sub[name]["ece_x100_boot"])}

    verdict["P1_k98_minus_full_nll"] = verdict[P1]["nll"]
    verdict["P2_k12_minus_k49_nll"] = verdict[P2]["nll"]
    out = {"n_val": int(n), "n_cls": int(t_logit[0].shape[1]),
           "teacher": {"nll": round(float(t_nll.mean()), 4), "ece_x100": round(100 * _ece(t_conf, t_cor), 3),
                       "acc": round(100 * float(t_cor.mean()), 2)},
           "subsets": sub, "per_seed": per_seed, "per_seed_t": seed_t, "per_run": per_run, "verdict": verdict}
    api.write_json("summary", out)
