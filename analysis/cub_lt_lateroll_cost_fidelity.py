"""H-A101 (round 390, analysis only, no training): cub_lt tail-group version of H-A100 (same code, run ids, RNG seed,
and quintiles taken inside the tail group). At the same cost (K=29), lateroll@29 beats maskedkd@29 on tail accuracy
(H-A67: +5.78 [+1.52, +10.04]). Is that per-image tail advantage concentrated on tail images where MaskedKD's masked
teacher loses the full teacher's soft label (high KL under the maskedkd mask)?

Per image i and seed s (final weights, seeds 1-3, val, un-augmented):
  kl_m_s  KL(p_T || q), q = keep-only masked teacher with the maskedkd@29 seed-s student's own last-block head-mean
          CLS attention top-29 (harness MaskedKD rule; identical to H-A100)
  kl_l_s  same with the lateroll@29 seed-s student's own late rollout (blocks 6..11, selectors/lateroll.py rule) top-29
  gain_s  lateroll_s correct - maskedkd_s correct
f_i = mean_s kl_m_s, G_i = mean_s gain_s. Primary: tail images only (val group 1), quintiles of f inside the tail,
contrast Q5 - Q1 of G (%p), paired image bootstrap 2000, numpy default_rng(20261112).
Record: same contrast on head images (group 0) and on all images; tail - head mean gain (bootstrap); per-quintile G,
kl, masked-teacher acc inside the tail; per-seed tail Q5 - Q1 t 95% CI (df 2); measured val acc vs val_acc_last.
Groups and labels enter only the evaluation; no positions, groups or labels enter any mask.
"""
import numpy
import torch

META = {
    "question": "on cub_lt (final, seeds 1-3, K=29), is lateroll's per-image tail accuracy advantage over MaskedKD "
                "concentrated on tail images where MaskedKD's masked teacher departs most from the full teacher?",
    "method": "keep-only masked teachers with each student's own selector rule (last-block CLS / late rollout) top-29; "
              "quintiles of maskedkd-mask KL inside the tail group; lateroll - maskedkd Q5 - Q1 with paired image "
              "bootstrap and seed t CI",
    "paper": "Hinton et al. 2015 (KD); Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); "
             "Hooker et al. NeurIPS 2019 (keep-only); Wah et al. 2011 (CUB-200-2011); Liu et al. CVPR 2019 (long-tail)",
}

K = 29
START = 6
BATCH = 50
N_BOOT = 2000
SEED = 20261112
RUNS = {
    "maskedkd": {1: "cub_lt_final_maskedkd_k29_s1_1103_234609_76d1",
                 2: "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 3: "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"},
    "lateroll": {1: "cub_lt_final_lateroll_k29_s1_1104_084157_2b15",
                 2: "cub_lt_final_lateroll_k29_s2_1104_084204_9f09",
                 3: "cub_lt_final_lateroll_k29_s3_1104_084209_a3f8"},
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
                                             runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid]["val_wga_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    T = api.teacher.eval()
    groups = api.val_groups().cpu().numpy()

    batches, logp_t = [], []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            logp_t.append(api.student_forward(T, x).logits.float().log_softmax(-1))
        batches.append((x, y))

    correct = {a: {} for a in RUNS}
    kl = {a: {} for a in RUNS}
    t_acc = {a: {} for a in RUNS}
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
                    else:
                        so = api.student_forward(S, x, attn_layers=list(range(START, depth)))
                        keep = _topk(_late_rollout(so.attn), K)
                    lq = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    kls.append((lp.exp() * (lp - lq)).sum(-1))
                    tac.append((lq.argmax(-1) == y).float())
                cor.append((so.logits.float().argmax(-1) == y).float())
            correct[a][s] = torch.cat(cor).cpu().numpy()
            kl[a][s] = torch.cat(kls).cpu().numpy()
            t_acc[a][s] = torch.cat(tac).cpu().numpy()
            del S

    seeds = sorted(RUNS["maskedkd"])
    gain_seed = {s: correct["lateroll"][s] - correct["maskedkd"][s] for s in seeds}
    G = numpy.mean([gain_seed[s] for s in seeds], 0)
    f = numpy.mean([kl["maskedkd"][s] for s in seeds], 0)
    fl = numpy.mean([kl["lateroll"][s] for s in seeds], 0)
    ta_l = numpy.mean([t_acc["lateroll"][s] for s in seeds], 0)
    ta_m = numpy.mean([t_acc["maskedkd"][s] for s in seeds], 0)
    tail, head = groups == 1, groups == 0

    rng = numpy.random.default_rng(SEED)
    sub = {}
    for name, m in (("tail", tail), ("head", head), ("all", numpy.ones_like(tail))):
        q, edges = _quint(f[m])
        sub[name] = {"n": int(m.sum()), "gain_mean_pct": round(100 * float(G[m].mean()), 3),
                     "kl_m_quintile_edges": [round(float(e), 4) for e in edges],
                     "quintile_kl_maskedkd": [round(v, 4) for v in _qmeans(f[m], q)],
                     "quintile_kl_lateroll": [round(v, 4) for v in _qmeans(fl[m], q)],
                     "quintile_gain_pct": [round(100 * v, 3) for v in _qmeans(G[m], q)],
                     "quintile_masked_teacher_acc_maskedkd_pct": [round(100 * v, 2) for v in _qmeans(ta_m[m], q)],
                     "quintile_masked_teacher_acc_lateroll_pct": [round(100 * v, 2) for v in _qmeans(ta_l[m], q)],
                     "q5_minus_q1_gain_pct_boot": _q51_boot(G[m], q, rng)}

    gt, gh = G[tail], G[head]
    bd = [gt[rng.integers(0, len(gt), len(gt))].mean() - gh[rng.integers(0, len(gh), len(gh))].mean()
          for _ in range(N_BOOT)]
    tail_minus_head = [round(100 * float(gt.mean() - gh.mean()), 3), round(100 * float(numpy.quantile(bd, 0.025)), 3),
                       round(100 * float(numpy.quantile(bd, 0.975)), 3)]

    per_seed = {}
    for s in seeds:
        qs, _ = _quint(kl["maskedkd"][s][tail])
        per_seed[s] = _qmeans(gain_seed[s][tail], qs)
    seed_t = _t95([100 * (per_seed[s][4] - per_seed[s][0]) for s in seeds])

    ci = sub["tail"]["q5_minus_q1_gain_pct_boot"]
    if ci[1] > 0:
        verdict = "tail_gain_in_low_fidelity"
    elif ci[2] < 0:
        verdict = "tail_gain_in_high_fidelity"
    else:
        verdict = "uncertain"

    out = {"k": K, "n_val": int(len(G)), "subsets": sub, "tail_minus_head_gain_pct_boot": tail_minus_head,
           "val_acc": {a: {str(s): [round(100 * float(correct[a][s].mean()), 2), runs[RUNS[a][s]]["val_acc_last"],
                                    round(100 * float(correct[a][s][tail].mean()), 2),
                                    runs[RUNS[a][s]]["val_wga_last"]] for s in seeds} for a in RUNS},
           "kl_mean_per_seed": {a: {str(s): round(float(kl[a][s].mean()), 4) for s in seeds} for a in kl},
           "per_seed_tail_quintile_gain_pct": {str(s): [round(100 * v, 3) for v in per_seed[s]] for s in seeds},
           "per_seed_tail_q5_minus_q1_t": seed_t,
           "verdict": {"tail_q5_minus_q1": verdict, "P1_tail_gain_in_low_fidelity": verdict == "tail_gain_in_low_fidelity"}}
    api.write_json("summary", out)
