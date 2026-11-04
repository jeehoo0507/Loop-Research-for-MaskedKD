"""H-A94 (round 383, analysis only, no training): on Aircraft (primary-metric dataset), where does the per-image
KD gain come from in terms of the clean teacher's confidence? Same 6 students as H-A93 (CE and Full@196 final,
seeds 1-3), val 700.

Per image i: KD gain g_i = mean over seeds 1-3 of (Full@196 student correct - CE student correct), final weights.
Teacher confidence c_i = clean teacher max softmax probability (no labels). Quintiles of c (Q1 = least confident).
Primary contrast Q1 - Q5 of g (%p); paired image bootstrap 2000, numpy default_rng(20261105).
Record: per-quintile gain, CE accuracy (headroom), teacher accuracy, teacher non-target entropy; Spearman(c, g)
with bootstrap CI; gain on teacher-correct vs teacher-wrong images; per-seed Q1 - Q5 with t 95% CI (df 2).
Labels enter only correctness.
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final, seeds 1-3), is the per-image KD gain (Full - CE) larger on images where the "
                "clean teacher is less confident (more soft-label mass off the top class)?",
    "method": "clean teacher max-prob quintiles; KD gain Q1 - Q5 with paired image bootstrap and per-seed t CI",
    "paper": "Hinton et al. 2015 (KD); Zhao et al. CVPR 2022 (DKD, non-target classes); Maji et al. 2013",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261105
RUNS = {
    "ce": {1: "aircraft_final_ce_k0_s1_1102_123044_6d5f",
           2: "aircraft_final_ce_k0_s2_1102_132227_80b7",
           3: "aircraft_final_ce_k0_s3_1102_132227_55cb"},
    "full": {1: "aircraft_final_full_k196_s1_1102_123044_ed5d",
             2: "aircraft_final_full_k196_s2_1102_132227_123b",
             3: "aircraft_final_full_k196_s3_1102_132227_69cc"},
}


def _spearman(a, b):
    ra = numpy.argsort(numpy.argsort(a)).astype(float)
    rb = numpy.argsort(numpy.argsort(b)).astype(float)
    return float(numpy.corrcoef(ra, rb)[0, 1])


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _qmeans(v, q):
    return [float(v[q == j].mean()) if (q == j).any() else float("nan") for j in range(5)]


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in RUNS.items()})
    T = api.teacher.eval()

    batches, conf, tcor, nt_ent = [], [], [], []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            p = api.student_forward(T, x).logits.float().softmax(-1)
        pm, am = p.max(-1)
        conf.append(pm)
        tcor.append((am == y).float())
        rest = p.scatter(1, am[:, None], 0.0)
        rest = rest / rest.sum(-1, keepdim=True).clamp_min(1e-12)
        nt_ent.append(-(rest * rest.clamp_min(1e-12).log()).sum(-1))
        batches.append((x, y))
    conf = torch.cat(conf).cpu().numpy()
    tcor = torch.cat(tcor).cpu().numpy()
    nt_ent = torch.cat(nt_ent).cpu().numpy()

    correct = {a: {} for a in RUNS}
    for a, ids in RUNS.items():
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            cor = []
            for x, y in batches:
                with torch.no_grad():
                    cor.append((api.student_forward(S, x).logits.argmax(-1) == y).float())
            correct[a][s] = torch.cat(cor).float().cpu().numpy()
            del S

    seeds = sorted(RUNS["ce"])
    gain_seed = {s: correct["full"][s] - correct["ce"][s] for s in seeds}
    g = numpy.mean([gain_seed[s] for s in seeds], 0)
    ce_acc = numpy.mean([correct["ce"][s] for s in seeds], 0)
    n = len(g)
    edges = numpy.quantile(conf, [0.2, 0.4, 0.6, 0.8])
    q = numpy.searchsorted(edges, conf, side="right")

    qm = _qmeans(g, q)
    c15 = qm[0] - qm[4]
    rho = _spearman(conf, g)
    rng = numpy.random.default_rng(SEED)
    bc, br = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = _qmeans(g[idx], q[idx])
        bc.append(m[0] - m[4])
        br.append(_spearman(conf[idx], g[idx]))
    ci = [round(100 * c15, 3), round(100 * float(numpy.quantile(bc, 0.025)), 3),
          round(100 * float(numpy.quantile(bc, 0.975)), 3)]
    ci_rho = [round(rho, 4), round(float(numpy.quantile(br, 0.025)), 4), round(float(numpy.quantile(br, 0.975)), 4)]

    per_seed = {s: _qmeans(gain_seed[s], q) for s in seeds}
    seed_t = _t95([100 * (per_seed[s][0] - per_seed[s][4]) for s in seeds])

    lo, hi = ci[1], ci[2]
    if lo > 0:
        verdict = "gain_in_low_confidence"
    elif hi < 0:
        verdict = "gain_in_high_confidence"
    else:
        verdict = "uncertain"

    tw = tcor < 0.5
    out = {"n_val": n, "kd_gain_mean_pct": round(100 * float(g.mean()), 3),
           "val_acc": {a: {str(s): round(100 * float(correct[a][s].mean()), 2) for s in seeds} for a in RUNS},
           "teacher_acc_pct": round(100 * float(tcor.mean()), 2),
           "conf_quintile_edges": [round(float(e), 4) for e in edges],
           "quintile_kd_gain_pct": [round(100 * v, 3) for v in qm],
           "quintile_ce_acc_pct": [round(100 * v, 2) for v in _qmeans(ce_acc, q)],
           "quintile_teacher_acc_pct": [round(100 * v, 2) for v in _qmeans(tcor, q)],
           "quintile_teacher_conf": [round(v, 4) for v in _qmeans(conf, q)],
           "quintile_nontarget_entropy": [round(v, 4) for v in _qmeans(nt_ent, q)],
           "q1_minus_q5_pct_boot": ci,
           "spearman_conf_gain_boot": ci_rho,
           "per_seed_quintile_gain_pct": {str(s): [round(100 * v, 3) for v in per_seed[s]] for s in seeds},
           "per_seed_q1_minus_q5_t": seed_t,
           "gain_teacher_correct_pct": round(100 * float(g[~tw].mean()), 3),
           "gain_teacher_wrong_pct": round(100 * float(g[tw].mean()), 3) if tw.any() else None,
           "n_teacher_wrong": int(tw.sum()),
           "verdict": {"q1_minus_q5": verdict, "P1_low_conf_gain": verdict == "gain_in_low_confidence"}}
    api.write_json("summary", out)
