"""H-A93 (round 382, analysis only, no training): Aircraft replication of the mentor's inverted-U test (CUB H4,
analysis_cub_budget_gonogo_1101_104255_4a16: not inverted-U, quintile KD gain [0.83, 0.00, 2.29, 2.08, 1.67]%p).
Aircraft is the primary-metric dataset with a larger KD gain (Full - CE val acc +2.3%p at seed 1).

Per image i: KD gain g_i = mean over seeds 1-3 of (Full@196 student correct - CE student correct), final weights.
Agreement a_i = mean over the 3 CE students of Spearman(CE student rollout over all blocks, clean teacher rollout).
Quintiles of a; contrasts mid (Q2-Q4) - Q1 and mid - Q5; paired image bootstrap 2000, numpy default_rng(20261104).
Record: per-seed contrasts with t 95% CI (df 2), Spearman(a, g). No labels or positions enter any score; labels
enter only correctness.
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3), is per-image KD gain (Full - CE) highest at middle CE-student / "
                "teacher attribution agreement (mentor's inverted U), unlike CUB H4?",
    "method": "per-image Spearman of CE student rollout vs teacher rollout; quintiles of agreement; KD gain "
              "contrasts with paired image bootstrap and per-seed t CI",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Hinton et al. 2015 (KD); Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261104
RUNS = {
    "ce": {1: "aircraft_final_ce_k0_s1_1102_123044_6d5f",
           2: "aircraft_final_ce_k0_s2_1102_132227_80b7",
           3: "aircraft_final_ce_k0_s3_1102_132227_55cb"},
    "full": {1: "aircraft_final_full_k196_s1_1102_123044_ed5d",
             2: "aircraft_final_full_k196_s2_1102_132227_123b",
             3: "aircraft_final_full_k196_s3_1102_132227_69cc"},
}
CUB_H4 = [0.83, 0.00, 2.29, 2.08, 1.67]


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    m = r[:, 0, 1:]
    return m / m.sum(1, keepdim=True)


def _rank(x):
    return x.argsort(1).argsort(1).float()


def _spearman_rows(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(1, keepdim=True)
    rb = rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1))


def _spearman(a, b):
    ra = numpy.argsort(numpy.argsort(a)).astype(float)
    rb = numpy.argsort(numpy.argsort(b)).astype(float)
    return float(numpy.corrcoef(ra, rb)[0, 1])


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _contrasts(g, q):
    qm = [float(g[q == j].mean()) if (q == j).any() else float("nan") for j in range(5)]
    mid = float(g[(q >= 1) & (q <= 3)].mean())
    return qm, mid - qm[0], mid - qm[4]


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    batches, tmaps = [], []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        tmaps.append(_rollout(out.attn))
        batches.append((x, y))

    correct = {a: {} for a in RUNS}
    agree = []
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            depth = len(S.blocks)
            cor, ag = [], []
            for (x, y), tm in zip(batches, tmaps):
                with torch.no_grad():
                    so = api.student_forward(S, x, attn_layers=list(range(depth)))
                cor.append((so.logits.argmax(-1) == y).float())
                if a == "ce":
                    ag.append(_spearman_rows(_rollout(so.attn), tm))
            correct[a][s] = torch.cat(cor).float().cpu().numpy()
            if a == "ce":
                agree.append(torch.cat(ag).float().cpu().numpy())
            del S

    seeds = sorted(RUNS["ce"])
    agr = numpy.mean(agree, 0)
    gain_seed = {s: correct["full"][s] - correct["ce"][s] for s in seeds}
    g = numpy.mean([gain_seed[s] for s in seeds], 0)
    n = len(g)
    edges = numpy.quantile(agr, [0.2, 0.4, 0.6, 0.8])
    q = numpy.searchsorted(edges, agr, side="right")

    qm, c1, c5 = _contrasts(g, q)
    rng = numpy.random.default_rng(SEED)
    b1, b5 = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        _, u, v = _contrasts(g[idx], q[idx])
        b1.append(u)
        b5.append(v)
    ci1 = [round(100 * c1, 3), round(100 * float(numpy.quantile(b1, 0.025)), 3),
           round(100 * float(numpy.quantile(b1, 0.975)), 3)]
    ci5 = [round(100 * c5, 3), round(100 * float(numpy.quantile(b5, 0.025)), 3),
           round(100 * float(numpy.quantile(b5, 0.975)), 3)]

    per_seed = {str(s): _contrasts(gain_seed[s], q) for s in seeds}
    seed_t = {"mid_minus_q1": _t95([100 * per_seed[str(s)][1] for s in seeds]),
              "mid_minus_q5": _t95([100 * per_seed[str(s)][2] for s in seeds])}

    lo1, hi1, lo5, hi5 = ci1[1], ci1[2], ci5[1], ci5[2]
    if lo1 > 0 and lo5 > 0:
        shape = "inverted_u"
    elif hi1 < 0 and hi5 < 0:
        shape = "u"
    elif (lo1 > 0 and hi5 < 0) or (hi1 < 0 and lo5 > 0):
        shape = "monotone"
    else:
        shape = "not_inverted_u_uncertain"

    out = {"n_val": n, "kd_gain_mean_pct": round(100 * float(g.mean()), 3),
           "val_acc": {a: {str(s): round(100 * float(correct[a][s].mean()), 2) for s in seeds} for a in RUNS},
           "agreement_mean": round(float(agr.mean()), 4), "agreement_quintile_edges": [round(float(e), 4)
                                                                                      for e in edges],
           "quintile_kd_gain_pct": [round(100 * v, 3) for v in qm],
           "mid_minus_q1_pct_boot": ci1, "mid_minus_q5_pct_boot": ci5,
           "per_seed_quintile_gain_pct": {s: [round(100 * v, 3) for v in per_seed[s][0]] for s in per_seed},
           "per_seed_t": seed_t,
           "spearman_agreement_gain": round(_spearman(agr, g), 4),
           "cub_h4_quintile_gain_pct": CUB_H4,
           "verdict": {"shape": shape,
                       "P1_inverted_u": shape == "inverted_u"}}
    api.write_json("summary", out)
