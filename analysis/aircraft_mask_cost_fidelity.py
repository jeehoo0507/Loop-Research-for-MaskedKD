"""H-A95 (round 384, analysis only, no training): on Aircraft (primary-metric dataset), is the per-image accuracy
cost of masking (Full@196 student correct - MaskedKD@29 student correct) concentrated on images where the masked
teacher loses the full teacher's soft label (high KL)?

Per image i and seed s (final weights, seeds 1-3, val 700, un-augmented):
  kl_s   KL(p_T || q_s), p_T = clean full teacher softmax, q_s = keep-only masked teacher with the maskedkd@29 seed-s
         student's own last-block head-mean CLS attention top-29 (harness MaskedKD rule)
  cost_s full_s correct - maskedkd_s correct
f_i = mean_s kl_s, m_i = mean_s cost_s. Quintiles of f (Q1 = most faithful). Primary contrast Q5 - Q1 of m (%p),
paired image bootstrap 2000, numpy default_rng(20261106).
Record: per-quintile cost, masked teacher acc, KD gain (Full - CE), teacher confidence; Spearman(f, m) with bootstrap
CI; Spearman(f, teacher conf); per-seed Q5 - Q1 (own-seed kl quintiles, own-seed cost) with t 95% CI (df 2);
measured val acc vs val_acc_last. Labels enter only correctness; no positions or labels enter any mask.
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3), is the per-image accuracy cost of MaskedKD@29 relative to Full@196 "
                "larger on images where the masked teacher's soft label departs more from the full teacher's?",
    "method": "keep-only masked teacher with each maskedkd student's own last-block CLS top-29; per-image KL "
              "quintiles; cost Q5 - Q1 with paired image bootstrap and per-seed t CI",
    "paper": "Hinton et al. 2015 (KD); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 (keep-only); "
             "Maji et al. 2013 (FGVC-Aircraft)",
}

K = 29
BATCH = 50
N_BOOT = 2000
SEED = 20261106
RUNS = {
    "maskedkd": {1: "aircraft_final_maskedkd_k29_s1_1102_132227_6542",
                 2: "aircraft_final_maskedkd_k29_s2_1102_180321_52e2",
                 3: "aircraft_final_maskedkd_k29_s3_1102_180336_347e"},
    "full": {1: "aircraft_final_full_k196_s1_1102_123044_ed5d",
             2: "aircraft_final_full_k196_s2_1102_132227_123b",
             3: "aircraft_final_full_k196_s3_1102_132227_69cc"},
    "ce": {1: "aircraft_final_ce_k0_s1_1102_123044_6d5f",
           2: "aircraft_final_ce_k0_s2_1102_132227_80b7",
           3: "aircraft_final_ce_k0_s3_1102_132227_55cb"},
}


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _spearman(a, b):
    ra = numpy.argsort(numpy.argsort(a)).astype(float)
    rb = numpy.argsort(numpy.argsort(b)).astype(float)
    return float(numpy.corrcoef(ra, rb)[0, 1])


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _quint(v):
    edges = numpy.quantile(v, [0.2, 0.4, 0.6, 0.8])
    return numpy.searchsorted(edges, v, side="right"), edges


def _qmeans(v, q):
    return [float(v[q == j].mean()) if (q == j).any() else float("nan") for j in range(5)]


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    T = api.teacher.eval()

    batches, logp_t, conf = [], [], []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            lp = api.student_forward(T, x).logits.float().log_softmax(-1)
        logp_t.append(lp)
        conf.append(lp.exp().max(-1).values)
        batches.append((x, y))
    conf = torch.cat(conf).cpu().numpy()

    correct = {a: {} for a in RUNS}
    kl, t_acc = {}, {}
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            depth = len(S.blocks)
            cor, kls, tac = [], [], []
            for (x, y), lp in zip(batches, logp_t):
                with torch.no_grad():
                    if a == "maskedkd":
                        so = api.student_forward(S, x, attn_layers=[depth - 1])
                        keep = _topk(so.attn[depth - 1].float().mean(1)[:, 0, 1:], K)
                        lq = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                        kls.append((lp.exp() * (lp - lq)).sum(-1))
                        tac.append((lq.argmax(-1) == y).float())
                    else:
                        so = api.student_forward(S, x)
                cor.append((so.logits.float().argmax(-1) == y).float())
            correct[a][s] = torch.cat(cor).cpu().numpy()
            if a == "maskedkd":
                kl[s] = torch.cat(kls).cpu().numpy()
                t_acc[s] = torch.cat(tac).cpu().numpy()
            del S

    seeds = sorted(RUNS["maskedkd"])
    cost_seed = {s: correct["full"][s] - correct["maskedkd"][s] for s in seeds}
    m = numpy.mean([cost_seed[s] for s in seeds], 0)
    f = numpy.mean([kl[s] for s in seeds], 0)
    g = numpy.mean([correct["full"][s] - correct["ce"][s] for s in seeds], 0)
    ta = numpy.mean([t_acc[s] for s in seeds], 0)
    n = len(m)
    q, edges = _quint(f)

    qm = _qmeans(m, q)
    c51 = qm[4] - qm[0]
    rho = _spearman(f, m)
    rng = numpy.random.default_rng(SEED)
    bc, br = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        mm = _qmeans(m[idx], q[idx])
        bc.append(mm[4] - mm[0])
        br.append(_spearman(f[idx], m[idx]))
    ci = [round(100 * c51, 3), round(100 * float(numpy.quantile(bc, 0.025)), 3),
          round(100 * float(numpy.quantile(bc, 0.975)), 3)]
    ci_rho = [round(rho, 4), round(float(numpy.quantile(br, 0.025)), 4), round(float(numpy.quantile(br, 0.975)), 4)]

    per_seed = {}
    for s in seeds:
        qs, _ = _quint(kl[s])
        per_seed[s] = _qmeans(cost_seed[s], qs)
    seed_t = _t95([100 * (per_seed[s][4] - per_seed[s][0]) for s in seeds])

    lo, hi = ci[1], ci[2]
    if lo > 0:
        verdict = "cost_in_low_fidelity"
    elif hi < 0:
        verdict = "cost_in_high_fidelity"
    else:
        verdict = "uncertain"

    out = {"k": K, "n_val": n, "mask_cost_mean_pct": round(100 * float(m.mean()), 3),
           "kd_gain_mean_pct": round(100 * float(g.mean()), 3),
           "val_acc": {a: {str(s): [round(100 * float(correct[a][s].mean()), 2),
                                    runs[RUNS[a][s]]["val_acc_last"]] for s in seeds} for a in RUNS},
           "kl_mean_per_seed": {str(s): round(float(kl[s].mean()), 4) for s in seeds},
           "kl_quintile_edges": [round(float(e), 4) for e in edges],
           "quintile_kl": [round(v, 4) for v in _qmeans(f, q)],
           "quintile_mask_cost_pct": [round(100 * v, 3) for v in qm],
           "quintile_masked_teacher_acc_pct": [round(100 * v, 2) for v in _qmeans(ta, q)],
           "quintile_kd_gain_pct": [round(100 * v, 3) for v in _qmeans(g, q)],
           "quintile_teacher_conf": [round(v, 4) for v in _qmeans(conf, q)],
           "q5_minus_q1_pct_boot": ci,
           "spearman_kl_cost_boot": ci_rho,
           "spearman_kl_teacher_conf": round(_spearman(f, conf), 4),
           "per_seed_quintile_cost_pct": {str(s): [round(100 * v, 3) for v in per_seed[s]] for s in seeds},
           "per_seed_q5_minus_q1_t": seed_t,
           "verdict": {"q5_minus_q1": verdict, "P1_cost_in_low_fidelity": verdict == "cost_in_low_fidelity"}}
    api.write_json("summary", out)
