"""H-A85 (round 374, analysis only, no training): H-A84 found that the maskedkd@29 vs maskedkd_frozen@29 masked-teacher
q_y gap on Waterbirds group 2 is not explained by teacher-evidence coverage, so the loop changes *which* tokens are
kept. Do the tokens only the online student keeps (M minus F) actively push the masked teacher away from the label,
or does the online student merely miss the helpful tokens the frozen student keeps (F minus M)?

Same token rule for both students (last-block head-mean CLS attention, top-K, K = 29), final snapshot, un-augmented
val image, same seed. Per image: I = M & F; keep-only masked teacher q_y on I, M (= I + M\\F), F (= I + F\\M), and on
all 196 patches. Marginals d_M = q_y(M) - q_y(I), d_F = q_y(F) - q_y(I); both add the same number of tokens to I.
Per seed and group: mean over images; paired same-seed t 95% CI (df 2).
No labels, groups or positions enter any mask; labels and groups enter only the statistics.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds group 2 (final, seeds 1-3), do the tokens kept only by the online maskedkd@29 student "
                "lower the masked teacher's q_y relative to the shared tokens, and do the tokens kept only by the "
                "frozen student raise it?",
    "method": "last-block CLS attention top-K; keep-only masked teacher on intersection / union pieces; marginal "
              "q_y of the non-shared tokens; paired t CI over seeds",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 (keep-only); Sagawa et al. ICLR 2020 "
             "(Waterbirds)",
}

K = 29
BATCH = 50
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


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _qy(T, api, x, y, keep):
    with torch.no_grad():
        lq = api.teacher_forward_keepmask(T, x, keep).float()
    return lq.softmax(-1).gather(1, y[:, None])[:, 0]


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                    for s, rid in sorted(ids.items())} for a, ids in ARMS.items()})
    T = api.teacher.eval()
    groups = api.val_groups().numpy()
    masks = {"all": numpy.ones(len(groups), dtype=bool)}
    masks.update({str(g): groups == g for g in range(4)})

    per = {}
    for s in (1, 2, 3):
        Sm, Sf = api.student(ARMS["maskedkd"][s]), api.student(ARMS["frozen"][s])
        acc = {"q_I": [], "q_M": [], "q_F": [], "q_full": [], "n_I": []}
        for x, y in api.val_batches(batch=BATCH):
            with torch.no_grad():
                om = api.student_forward(Sm, x, attn_layers=[len(Sm.blocks) - 1])
                of = api.student_forward(Sf, x, attn_layers=[len(Sf.blocks) - 1])
            km = _topk(om.attn[len(Sm.blocks) - 1].float().mean(1)[:, 0, 1:], K)
            kf = _topk(of.attn[len(Sf.blocks) - 1].float().mean(1)[:, 0, 1:], K)
            ki = km & kf
            acc["q_I"].append(_qy(T, api, x, y, ki))
            acc["q_M"].append(_qy(T, api, x, y, km))
            acc["q_F"].append(_qy(T, api, x, y, kf))
            acc["q_full"].append(_qy(T, api, x, y, torch.ones_like(km)))
            acc["n_I"].append(ki.sum(1).float())
        per[s] = {k: torch.cat(v).cpu().numpy() for k, v in acc.items()}

    res = {}
    for g in GROUPS:
        m = masks[g]
        rows = {"d_M": [], "d_F": [], "d_F_minus_d_M": [], "q_I": [], "q_M": [], "q_F": [], "q_full": [],
                "n_I": [], "frac_dM_neg": []}
        for s in (1, 2, 3):
            p = {k: v[m] for k, v in per[s].items()}
            dM, dF = p["q_M"] - p["q_I"], p["q_F"] - p["q_I"]
            rows["d_M"].append(float(dM.mean()))
            rows["d_F"].append(float(dF.mean()))
            rows["d_F_minus_d_M"].append(float((dF - dM).mean()))
            for k in ("q_I", "q_M", "q_F", "q_full", "n_I"):
                rows[k].append(float(p[k].mean()))
            rows["frac_dM_neg"].append(float((dM < 0).mean()))
        res[g] = {"n": int(m.sum()), "per_seed": {k: [round(v, 4) for v in vs] for k, vs in rows.items()},
                  "t95": {k: _t95(vs) for k, vs in rows.items()}}
    p1, p2 = res["2"]["t95"]["d_M"], res["2"]["t95"]["d_F"]
    out = {"k": K, "n_val": int(len(groups)), "by_group": res,
           "verdict": {"P1_group2_dM_ci_high_lt0": bool(p1[2] < 0),
                       "P2_group2_dF_ci_low_gt0": bool(p2[1] > 0)}}
    api.write_json("summary", out)
