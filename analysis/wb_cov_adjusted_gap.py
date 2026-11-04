"""H-A84 (round 373, analysis only, no training): does teacher-evidence coverage explain the masked-teacher collapse
of the self-selecting maskedkd@29 student on Waterbirds group 2? H-A83 found a small coverage gap (maskedkd - frozen
cov group 2 -0.0051) next to a large masked-teacher accuracy gap (t_acc group 2 -0.198). If the q_y gap remains after
reweighting maskedkd images to the frozen run's coverage distribution, rollout mass is not what the loop removes;
the loop then changes *which* tokens are kept beyond their teacher mass.

Same token rule for both students (last-block head-mean CLS attention, top-K, K = 29), final snapshot, un-augmented
val image. Per image: cov (teacher rollout mass inside the top-K), q_y / t_acc of the keep-only masked teacher, and
the Jaccard overlap of the two students' kept sets (same image, same seed).
Coverage adjustment, per seed and group: 5 bins on the frozen run's cov quantiles within the group; adjusted gap =
sum over bins of w_b * (mean maskedkd q_y in bin - mean frozen q_y in bin), w_b = frozen share of the bin, renormalised
over bins where both runs have images. Paired same-seed t 95% CI (df 2). Within-run Spearman(cov, q_y) per run.
No labels, groups or positions enter any mask; labels and groups enter only the statistics.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds (final, seeds 1-3), does the maskedkd@29 - maskedkd_frozen@29 masked-teacher q_y gap "
                "in group 2 remain after reweighting to equal teacher-evidence coverage, and does per-image coverage "
                "rank masked-teacher q_y within a run?",
    "method": "attention rollout (teacher all blocks); last-block CLS attention top-K; keep-only masked teacher; "
              "coverage-bin reweighting (direct standardisation); paired t CI over seeds; Spearman",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 "
             "(keep-only); Sagawa et al. ICLR 2020 (Waterbirds)",
}

K = 29
BATCH = 50
NBIN = 5
GROUPS = ("all", "0", "1", "2", "3")
ARMS = {
    "maskedkd": {1: "waterbirds_final_maskedkd_k29_s1_1031_201502_4be8",
                 2: "waterbirds_final_maskedkd_k29_s2_1103_170931_92bd",
                 3: "waterbirds_final_maskedkd_k29_s3_1103_170935_907a"},
    "frozen": {1: "waterbirds_final_maskedkd_frozen_k29_s1_1103_171021_f885",
               2: "waterbirds_final_maskedkd_frozen_k29_s2_1103_171026_2b13",
               3: "waterbirds_final_maskedkd_frozen_k29_s3_1103_171031_b65e"},
}


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    m = r[:, 0, 1:]
    return m / m.sum(1, keepdim=True)


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _rank(v):
    r = numpy.empty(len(v))
    r[numpy.argsort(v, kind="stable")] = numpy.arange(len(v))
    return r


def _spearman(a, b):
    return float(numpy.corrcoef(_rank(a), _rank(b))[0, 1])


def _adjusted(cm, qm, cf, qf):
    edges = numpy.quantile(cf, numpy.linspace(0, 1, NBIN + 1))
    edges[0], edges[-1] = -numpy.inf, numpy.inf
    bm, bf = numpy.digitize(cm, edges[1:-1]), numpy.digitize(cf, edges[1:-1])
    num, den = 0.0, 0.0
    for b in range(NBIN):
        if (bm == b).any() and (bf == b).any():
            w = float((bf == b).mean())
            num += w * (qm[bm == b].mean() - qf[bf == b].mean())
            den += w
    return num / den, den


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in ARMS.items()})
    T = api.teacher.eval()
    depth_t = len(T.blocks)
    groups = api.val_groups().numpy()
    masks = {"all": numpy.ones(len(groups), dtype=bool)}
    masks.update({str(g): groups == g for g in range(4)})

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        batches.append((x, y, _rollout(out.attn)))

    per = {a: {} for a in ARMS}
    for a, ids in ARMS.items():
        for seed, rid in sorted(ids.items()):
            S = api.student(rid)
            depth = len(S.blocks)
            acc = {"cov": [], "q_y": [], "t_acc": [], "keep": []}
            for x, y, tm in batches:
                with torch.no_grad():
                    so = api.student_forward(S, x, attn_layers=[depth - 1])
                    keep = _topk(so.attn[depth - 1].float().mean(1)[:, 0, 1:], K)
                    lq = api.teacher_forward_keepmask(T, x, keep).float()
                acc["cov"].append((tm * keep).sum(1).float())
                acc["q_y"].append(lq.softmax(-1).gather(1, y[:, None])[:, 0])
                acc["t_acc"].append((lq.argmax(-1) == y).float())
                acc["keep"].append(keep)
            per[a][seed] = {s: torch.cat(v).cpu() for s, v in acc.items()}

    res = {}
    for g in GROUPS:
        m = masks[g]
        rows = {"raw_q": [], "adj_q": [], "raw_t": [], "adj_t": [], "raw_cov": [], "w_used": [],
                "jaccard": [], "spear_mk": [], "spear_fr": []}
        for s in (1, 2, 3):
            pm, pf = per["maskedkd"][s], per["frozen"][s]
            cm, cf = pm["cov"].numpy()[m], pf["cov"].numpy()[m]
            qm, qf = pm["q_y"].numpy()[m], pf["q_y"].numpy()[m]
            tm_, tf = pm["t_acc"].numpy()[m], pf["t_acc"].numpy()[m]
            aq, w = _adjusted(cm, qm, cf, qf)
            at, _ = _adjusted(cm, tm_, cf, tf)
            km, kf = pm["keep"][torch.from_numpy(m)], pf["keep"][torch.from_numpy(m)]
            jac = ((km & kf).sum(1).float() / (km | kf).sum(1).float()).mean().item()
            rows["raw_q"].append(float(qm.mean() - qf.mean()))
            rows["adj_q"].append(float(aq))
            rows["raw_t"].append(float(tm_.mean() - tf.mean()))
            rows["adj_t"].append(float(at))
            rows["raw_cov"].append(float(cm.mean() - cf.mean()))
            rows["w_used"].append(round(w, 4))
            rows["jaccard"].append(round(jac, 4))
            rows["spear_mk"].append(_spearman(cm, qm))
            rows["spear_fr"].append(_spearman(cf, qf))
        res[g] = {"n": int(m.sum()), "per_seed": {k: [round(v, 4) for v in vs] for k, vs in rows.items()},
                  "t95": {k: _t95(vs) for k, vs in rows.items()},
                  "adj_minus_raw_q": _t95([a - r for a, r in zip(rows["adj_q"], rows["raw_q"])])}
    p1 = res["2"]["t95"]["adj_q"]
    p2 = res["2"]["t95"]["spear_mk"]
    out = {"k": K, "nbin": NBIN, "n_val": int(len(groups)), "by_group": res,
           "verdict": {"P1_group2_cov_adjusted_q_gap_ci_high_lt0": bool(p1[2] < 0),
                       "P2_group2_maskedkd_within_run_spearman_cov_q_ci_low_gt0": bool(p2[1] > 0)}}
    api.write_json("summary", out)
