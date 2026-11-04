"""H-A92 (round 381, analysis only, no training): Aircraft contrast for H-A90/H-A91. On Waterbirds, the lateroll rule
applied to the v0 (trustfilt) student gives a higher group 2 masked teacher q_y than on the lateroll student (final
+0.175, H-A90); on cub_lt tail the gap is -0.011 (H-A91). Aircraft is the unbiased dataset with the primary metric
(mean accuracy); its v0 - lateroll val accuracy is -0.71 (seeds 1-3, uncertain). Is there a whole-set gap in masked
teacher q_y under the lateroll rule on Aircraft? Same rule and measurement as H-A90/91: L = student rollout over
blocks 6..11 top-K, keep-only masked teacher q_y on L; V = same scores restricted to the clean teacher rollout top
1.5K region; student accuracy at each snapshot (record).

Students: Aircraft v0@29 and lateroll@29 final seeds 1-3, snapshots e25, e50, final, clean val images.
Per seed and epoch: mean over images; paired same-seed t 95% CI (df 2) of v0 student - lateroll student.
No labels or positions enter any mask; labels enter only the statistics.
"""
import math

import numpy
import torch

META = {
    "question": "on Aircraft (final K=29, seeds 1-3), is there a v0 student - lateroll student gap in masked "
                "teacher q_y under the lateroll rule, as on Waterbirds group 2 (H-A90) but not cub_lt tail (H-A91)?",
    "method": "student rollout blocks 6..11 top-K on e25/e50/final snapshots; keep-only masked teacher q_y; "
              "paired t CI over seeds",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Hooker et al. NeurIPS 2019 "
             "(keep-only); Maji et al. 2013 (FGVC-Aircraft)",
}

K = 29
START = 6
WIDEN = 1.5
BATCH = 50
EPOCHS = (25, 50, "final")
STATS = ("q_L", "q_V", "s_acc")
ARMS = {
    "trustfilt": {1: "aircraft_final_trustfilt_k29_s1_1103_074527_357e",
                  2: "aircraft_final_trustfilt_k29_s2_1103_074535_db76",
                  3: "aircraft_final_trustfilt_k29_s3_1103_074540_2203"},
    "lateroll": {1: "aircraft_final_lateroll_k29_s1_1103_065756_f0d9",
                 2: "aircraft_final_lateroll_k29_s2_1103_065801_38e1",
                 3: "aircraft_final_lateroll_k29_s3_1103_065805_2130"},
}
WB_GROUP2_FINAL_GAP = 0.175  # H-A90, analysis_wb_filter_snapshots_1104_155628_6d71
HALF = WB_GROUP2_FINAL_GAP / 2


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
                                             runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid]["snapshots"]]
                                    for s, rid in sorted(ids.items())} for a, ids in ARMS.items()})
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        tm = _rollout(out.attn)
        allowed = _topk(tm, min(int(math.ceil(WIDEN * K)), tm.shape[1]))
        batches.append((x, y, allowed))

    per = {a: {str(e): {} for e in EPOCHS} for a in ARMS}
    for a, ids in ARMS.items():
        for s, rid in sorted(ids.items()):
            for ep in EPOCHS:
                S = api.student(rid, epoch=None if ep == "final" else ep)
                depth = len(S.blocks)
                acc = {k: [] for k in STATS}
                for x, y, allowed in batches:
                    with torch.no_grad():
                        so = api.student_forward(S, x, attn_layers=list(range(START, depth)))
                    late = _rollout([so.attn[i] for i in range(START, depth)])
                    acc["q_L"].append(_qy(T, api, x, y, _topk(late, K)))
                    acc["q_V"].append(_qy(T, api, x, y, _topk(late.masked_fill(~allowed, -1.0), K)))
                    acc["s_acc"].append((so.logits.argmax(-1) == y).float())
                per[a][str(ep)][s] = {k: torch.cat(v).float().cpu().numpy() for k, v in acc.items()}
                del S

    seeds = sorted(ARMS["trustfilt"])
    mean = {a: {e: {k: [float(per[a][e][s][k].mean()) for s in seeds] for k in STATS} for e in per[a]} for a in ARMS}
    res = {"per_seed_means": {a: {e: {k: [round(v, 4) for v in vs] for k, vs in d.items()} for e, d in ae.items()}
                              for a, ae in mean.items()},
           "gap_v0_minus_lateroll": {}, "gap_final_minus_e25": {}}
    for e in map(str, EPOCHS):
        res["gap_v0_minus_lateroll"][e] = {k: _t95([u - v for u, v in zip(mean["trustfilt"][e][k],
                                                                         mean["lateroll"][e][k])]) for k in STATS}
    for k in STATS:
        gf = [u - v for u, v in zip(mean["trustfilt"]["final"][k], mean["lateroll"]["final"][k])]
        g25 = [u - v for u, v in zip(mean["trustfilt"]["25"][k], mean["lateroll"]["25"][k])]
        res["gap_final_minus_e25"][k] = _t95([u - v for u, v in zip(gf, g25)])
    p1 = res["gap_v0_minus_lateroll"]["final"]["q_L"]
    n_val = int(len(per["trustfilt"]["final"][seeds[0]]["q_L"]))
    out = {"k": K, "n_val": n_val, **res,
           "verdict": {"P1_all_qL_gap_final_ci_contains0": bool(p1[1] <= 0 <= p1[2]),
                       "P1_all_qL_gap_final_ci_low_gt0": bool(p1[1] > 0),
                       "P1_all_qL_gap_final_ci_high_lt0": bool(p1[2] < 0),
                       "P2_all_qL_gap_final_mean_lt_half_wb_group2": bool(p1[0] < HALF)}}
    api.write_json("summary", out)
