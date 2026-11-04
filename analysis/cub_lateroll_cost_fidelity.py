"""H-A100 (round 389, analysis only, no training): CUB replication of H-A96 (same code, run ids and RNG seed only).
At the same cost (K=29), is lateroll's per-image accuracy advantage over MaskedKD concentrated on the images where
MaskedKD's masked teacher loses the full teacher's soft label (high KL under the maskedkd mask)?

Per image i and seed s (final weights, seeds 1-3, val 600, un-augmented):
  kl_m_s  KL(p_T || q), q = keep-only masked teacher with the maskedkd@29 seed-s student's own last-block head-mean
          CLS attention top-29 (harness MaskedKD rule; identical to H-A97)
  kl_l_s  same with the lateroll@29 seed-s student's own late rollout (blocks 6..11, selectors/lateroll.py rule) top-29
  gain_s  lateroll_s correct - maskedkd_s correct
f_i = mean_s kl_m_s (H-A97 quintiles, Q1 = most faithful), G_i = mean_s gain_s. Primary contrast Q5 - Q1 of G (%p),
paired image bootstrap 2000, numpy default_rng(20261111).
Record: per-quintile G, lateroll cost (Full - lateroll), maskedkd cost, mean kl_l, lateroll masked-teacher acc;
Spearman(f - mean kl_l, G) with bootstrap CI; Spearman(kl_m, kl_l); per-seed Q5 - Q1 (own-seed kl_m quintiles) with
t 95% CI (df 2); lateroll cost Q5 - Q1 on its own kl_l quintiles (bootstrap); measured val acc vs val_acc_last.
Labels enter only correctness; no positions or labels enter any mask.
"""
import numpy
import torch

META = {
    "question": "on CUB (final, seeds 1-3, K=29), is lateroll's per-image accuracy advantage over MaskedKD "
                "concentrated on images where MaskedKD's masked teacher departs most from the full teacher?",
    "method": "keep-only masked teachers with each student's own selector rule (last-block CLS / late rollout) top-29; "
              "quintiles of maskedkd-mask KL; lateroll - maskedkd Q5 - Q1 with paired image bootstrap and seed t CI",
    "paper": "Hinton et al. 2015 (KD); Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); "
             "Hooker et al. NeurIPS 2019 (keep-only); Wah et al. 2011 (CUB-200-2011)",
}

K = 29
START = 6
BATCH = 50
N_BOOT = 2000
SEED = 20261111
RUNS = {
    "maskedkd": {1: "final_maskedkd_k29_s1_1101_104303_b676",
                 2: "final_maskedkd_k29_s2_1101_125958_5b02",
                 3: "final_maskedkd_k29_s3_1101_140937_b00f"},
    "lateroll": {1: "final_lateroll_k29_s1_1101_104303_ef38",
                 2: "final_lateroll_k29_s2_1101_125958_6470",
                 3: "final_lateroll_k29_s3_1101_140937_ec98"},
    "full": {1: "cub_final_full_k196_s1_1030_153055_d87b",
             2: "final_full_k196_s2_1101_125958_cc04",
             3: "final_full_k196_s3_1101_140937_e8fa"},
}


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _late_rollout(attn):
    r = None
    for a in attn[START:]:
        a = a.float().mean(dim=1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


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


def _q51_boot(v, q, rng):
    qm = _qmeans(v, q)
    n, bs = len(v), []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        mm = _qmeans(v[idx], q[idx])
        bs.append(mm[4] - mm[0])
    return [round(100 * (qm[4] - qm[0]), 3), round(100 * float(numpy.quantile(bs, 0.025)), 3),
            round(100 * float(numpy.quantile(bs, 0.975)), 3)]


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    T = api.teacher.eval()

    batches, logp_t = [], []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            logp_t.append(api.student_forward(T, x).logits.float().log_softmax(-1))
        batches.append((x, y))

    correct = {a: {} for a in RUNS}
    kl = {"maskedkd": {}, "lateroll": {}}
    t_acc = {"maskedkd": {}, "lateroll": {}}
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
                    elif a == "lateroll":
                        so = api.student_forward(S, x, attn_layers=list(range(START, depth)))
                        keep = _topk(_late_rollout(so.attn), K)
                    else:
                        so = api.student_forward(S, x)
                        keep = None
                    if keep is not None:
                        lq = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                        kls.append((lp.exp() * (lp - lq)).sum(-1))
                        tac.append((lq.argmax(-1) == y).float())
                cor.append((so.logits.float().argmax(-1) == y).float())
            correct[a][s] = torch.cat(cor).cpu().numpy()
            if a in kl:
                kl[a][s] = torch.cat(kls).cpu().numpy()
                t_acc[a][s] = torch.cat(tac).cpu().numpy()
            del S

    seeds = sorted(RUNS["maskedkd"])
    gain_seed = {s: correct["lateroll"][s] - correct["maskedkd"][s] for s in seeds}
    G = numpy.mean([gain_seed[s] for s in seeds], 0)
    cost_l = numpy.mean([correct["full"][s] - correct["lateroll"][s] for s in seeds], 0)
    cost_m = numpy.mean([correct["full"][s] - correct["maskedkd"][s] for s in seeds], 0)
    f = numpy.mean([kl["maskedkd"][s] for s in seeds], 0)
    fl = numpy.mean([kl["lateroll"][s] for s in seeds], 0)
    ta_l = numpy.mean([t_acc["lateroll"][s] for s in seeds], 0)
    ta_m = numpy.mean([t_acc["maskedkd"][s] for s in seeds], 0)
    n = len(G)
    q, edges = _quint(f)
    ql, _ = _quint(fl)

    rng = numpy.random.default_rng(SEED)
    ci = _q51_boot(G, q, rng)
    dkl = f - fl
    rho = _spearman(dkl, G)
    br = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        br.append(_spearman(dkl[idx], G[idx]))
    ci_rho = [round(rho, 4), round(float(numpy.quantile(br, 0.025)), 4), round(float(numpy.quantile(br, 0.975)), 4)]
    ci_lcost_own = _q51_boot(cost_l, ql, rng)

    per_seed = {}
    for s in seeds:
        qs, _ = _quint(kl["maskedkd"][s])
        per_seed[s] = _qmeans(gain_seed[s], qs)
    seed_t = _t95([100 * (per_seed[s][4] - per_seed[s][0]) for s in seeds])

    lo, hi = ci[1], ci[2]
    if lo > 0:
        verdict = "gain_in_low_fidelity"
    elif hi < 0:
        verdict = "gain_in_high_fidelity"
    else:
        verdict = "uncertain"

    out = {"k": K, "n_val": n, "gain_mean_pct": round(100 * float(G.mean()), 3),
           "val_acc": {a: {str(s): [round(100 * float(correct[a][s].mean()), 2),
                                    runs[RUNS[a][s]]["val_acc_last"]] for s in seeds} for a in RUNS},
           "kl_mean_per_seed": {a: {str(s): round(float(kl[a][s].mean()), 4) for s in seeds} for a in kl},
           "kl_m_quintile_edges": [round(float(e), 4) for e in edges],
           "quintile_kl_maskedkd": [round(v, 4) for v in _qmeans(f, q)],
           "quintile_kl_lateroll": [round(v, 4) for v in _qmeans(fl, q)],
           "quintile_gain_lateroll_minus_maskedkd_pct": [round(100 * v, 3) for v in _qmeans(G, q)],
           "quintile_cost_maskedkd_pct": [round(100 * v, 3) for v in _qmeans(cost_m, q)],
           "quintile_cost_lateroll_pct": [round(100 * v, 3) for v in _qmeans(cost_l, q)],
           "quintile_masked_teacher_acc_maskedkd_pct": [round(100 * v, 2) for v in _qmeans(ta_m, q)],
           "quintile_masked_teacher_acc_lateroll_pct": [round(100 * v, 2) for v in _qmeans(ta_l, q)],
           "q5_minus_q1_gain_pct_boot": ci,
           "spearman_klreduction_gain_boot": ci_rho,
           "spearman_kl_m_kl_l": round(_spearman(f, fl), 4),
           "lateroll_cost_own_kl_q5_minus_q1_pct_boot": ci_lcost_own,
           "lateroll_cost_own_kl_quintiles_pct": [round(100 * v, 3) for v in _qmeans(cost_l, ql)],
           "per_seed_quintile_gain_pct": {str(s): [round(100 * v, 3) for v in per_seed[s]] for s in seeds},
           "per_seed_q5_minus_q1_t": seed_t,
           "verdict": {"q5_minus_q1": verdict, "P1_gain_in_low_fidelity": verdict == "gain_in_low_fidelity"}}
    api.write_json("summary", out)
