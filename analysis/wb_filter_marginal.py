"""H-A89 (round 378, analysis only, no training): H-A88 found that v0 (trustfilt) and lateroll keep tokens at the
same mean grid centre distance on Waterbirds group 2 (+0.003), yet v0 - lateroll WGA is +8.77, so the teacher-region
filter's share is not position. What does the filter remove? On the same student, apply both rules: L = lateroll
top-K (student rollout over blocks 6..11), V = v0 rule (same scores restricted to the clean teacher rollout top
1.5K region). I = V & L. Keep-only masked teacher q_y on I, V, L and all 196 patches; marginals d_V = q_y(V) - q_y(I),
d_L = q_y(L) - q_y(I) (both add the same number of tokens to I). Do the tokens the filter removes (L minus V) actively
lower q_y on group 2, and is the filter's margin d_V - d_L larger on group 2 than on group 0?

Students: v0@29 final seeds 1-3 (main) and lateroll@29 final seeds 1-3 (record), final snapshot, clean val images.
Per seed and group: mean over images; paired same-seed t 95% CI (df 2).
No labels, groups or positions enter any mask; labels and groups enter only the statistics.
"""
import math

import numpy
import torch

META = {
    "question": "on Waterbirds group 2 (final K=29, seeds 1-3), do the tokens the teacher-region filter removes from "
                "the lateroll choice lower the masked teacher's q_y relative to the shared tokens, and is the "
                "filter's margin larger on group 2 than on group 0?",
    "method": "student rollout blocks 6..11 top-K with / without teacher rollout top-1.5K region; keep-only masked "
              "teacher on intersection / union pieces; marginal q_y; paired t CI over seeds",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Hooker et al. NeurIPS 2019 "
             "(keep-only); Sagawa et al. ICLR 2020 (Waterbirds)",
}

K = 29
START = 6
WIDEN = 1.5
BATCH = 50
GROUPS = ("all", "0", "1", "2", "3")
ARMS = {
    "trustfilt": {1: "waterbirds_final_trustfilt_k29_s1_1103_171121_24be",
                  2: "waterbirds_final_trustfilt_k29_s2_1103_171126_b33d",
                  3: "waterbirds_final_trustfilt_k29_s3_1103_171130_db65"},
    "lateroll": {1: "waterbirds_final_lateroll_k29_s1_1104_025003_b396",
                 2: "waterbirds_final_lateroll_k29_s2_1104_025008_c94e",
                 3: "waterbirds_final_lateroll_k29_s3_1104_025012_629b"},
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


def _qy(T, api, x, y, keep):
    with torch.no_grad():
        lq = api.teacher_forward_keepmask(T, x, keep).float()
    return lq.softmax(-1).gather(1, y[:, None])[:, 0]


def run(api):
    runs = api.runs()
    api.write_json("students", {a: {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                             runs[rid]["seed"], runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]]
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
        tm = _rollout(out.attn)
        allowed = _topk(tm, min(int(math.ceil(WIDEN * K)), tm.shape[1]))
        q_full = _qy(T, api, x, y, torch.ones_like(allowed))
        batches.append((x, y, allowed, q_full))

    res = {}
    for a, ids in ARMS.items():
        per = {}
        for s, rid in sorted(ids.items()):
            S = api.student(rid)
            depth = len(S.blocks)
            acc = {"q_I": [], "q_V": [], "q_L": [], "q_full": [], "n_I": []}
            for x, y, allowed, q_full in batches:
                with torch.no_grad():
                    so = api.student_forward(S, x, attn_layers=list(range(START, depth)))
                late = _rollout([so.attn[i] for i in range(START, depth)])
                kl = _topk(late, K)
                kv = _topk(late.masked_fill(~allowed, -1.0), K)
                ki = kl & kv
                acc["q_I"].append(_qy(T, api, x, y, ki))
                acc["q_V"].append(_qy(T, api, x, y, kv))
                acc["q_L"].append(_qy(T, api, x, y, kl))
                acc["q_full"].append(q_full)
                acc["n_I"].append(ki.sum(1).float())
            per[s] = {k: torch.cat(v).float().cpu().numpy() for k, v in acc.items()}
        res[a] = {}
        for g in GROUPS:
            m = masks[g]
            rows = {"d_V": [], "d_L": [], "d_V_minus_d_L": [], "q_I": [], "q_V": [], "q_L": [], "q_full": [],
                    "n_I": [], "frac_dL_neg": []}
            for s in sorted(ids):
                p = {k: v[m] for k, v in per[s].items()}
                dV, dL = p["q_V"] - p["q_I"], p["q_L"] - p["q_I"]
                rows["d_V"].append(float(dV.mean()))
                rows["d_L"].append(float(dL.mean()))
                rows["d_V_minus_d_L"].append(float((dV - dL).mean()))
                for k in ("q_I", "q_V", "q_L", "q_full", "n_I"):
                    rows[k].append(float(p[k].mean()))
                rows["frac_dL_neg"].append(float((dL < 0).mean()))
            res[a][g] = {"n": int(m.sum()), "per_seed": {k: [round(v, 4) for v in vs] for k, vs in rows.items()},
                         "t95": {k: _t95(vs) for k, vs in rows.items()}}
        g2, g0 = res[a]["2"]["per_seed"]["d_V_minus_d_L"], res[a]["0"]["per_seed"]["d_V_minus_d_L"]
        res[a]["margin_2_minus_0"] = _t95([u - v for u, v in zip(g2, g0)])
    p1 = res["trustfilt"]["2"]["t95"]["d_L"]
    p2 = res["trustfilt"]["margin_2_minus_0"]
    out = {"k": K, "n_val": int(len(groups)), "group_counts": [int((groups == g).sum()) for g in range(4)],
           "by_arm": res,
           "verdict": {"P1_trustfilt_group2_dL_ci_high_lt0": bool(p1[2] < 0),
                       "P2_trustfilt_margin_2_minus_0_ci_low_gt0": bool(p2[1] > 0)}}
    api.write_json("summary", out)
