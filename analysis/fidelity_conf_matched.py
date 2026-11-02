"""H-A7: is the lower MaskedKD masked-teacher faithfulness on FGVC-Aircraft than on CUB at the same K (H-A6) only
a consequence of the Aircraft teacher being less confident? Run once per dataset (--dataset cub, --dataset aircraft).

Why: H-A6 found maskedkd@98 KL 0.182 on Aircraft vs 0.071 on CUB, and preregistered the caveat that the Aircraft
teacher is less sure (val 77.43 vs 84.67), so part of the gap could come from flatter full-teacher soft labels.
Here every val image is put in a fixed bin of the full teacher's max probability (bins chosen before any result),
and per-bin means and standard errors of the H-A6 statistics are written, so the two datasets can be compared bin by
bin and with Aircraft reweighted to the CUB confidence mix. Descriptive only: nothing is selected or tuned, no
selector is changed (QUEUE 2026-11-02 08:50). Same student choice (final seed-1 maskedkd@98 of that dataset), same
scores and statistics as analysis/aircraft_maskedkd_fidelity_k.py.
"""
import math

import torch

META = {
    "question": "within fixed bins of full-teacher confidence, is the MaskedKD masked teacher at K=98/49 less "
                "faithful (KL, top-1 agreement) on FGVC-Aircraft than on CUB?",
    "method": "keep-only masked-teacher faithfulness stratified by full-teacher max probability (fixed bins), "
              "last-block CLS attention and attention rollout token keeps",
    "paper": "Hinton et al. 2015 (KD); Hooker et al. NeurIPS 2019 (keep-only evaluation); "
             "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD)",
}

KS = (98, 49)
SCORES = ("maskedkd", "rollout")
EDGES = (0.0, 0.5, 0.8, 0.95, 1.0001)  # full-teacher max-prob bins, fixed before results (PROPOSALS H-A7)


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _attn_scores(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad():
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
    A = [a.float().mean(1) for a in out.attn]
    B, N, _ = A[0].shape
    eye = torch.eye(N, device=x.device).expand(B, N, N)
    R = None
    for a in A:
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        R = a if R is None else a @ R
    return {"maskedkd": A[-1][:, 0, 1:], "rollout": R[:, 0, 1:]}


def _mean_se(v):
    n = len(v)
    if n == 0:
        return None, None
    m = sum(v) / n
    if n < 2:
        return m, None
    var = sum((a - m) ** 2 for a in v) / (n - 1)
    return m, math.sqrt(var / n)


def run(api):
    ds = api.dataset
    runs = api.runs()
    rid = None
    for r_id, r in sorted(runs.items()):
        if (r.get("dataset") == ds and r["seed"] == 1 and r["tier"] == "final"
                and r["method"] == "maskedkd" and int(r.get("keep") or 0) == 98):
            rid = r_id
            break
    api.write_json("student", {"dataset": ds, "run": rid, "val_acc_last": runs[rid]["val_acc_last"] if rid else None})
    if rid is None:
        return

    T = api.teacher.eval()
    student = api.student(rid)
    pmax, t_ok = [], []
    per = {(m, k): {"kl": [], "top1_agree": []} for m in SCORES for k in KS}
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        pf = full.exp()
        pmax += pf.max(-1).values.tolist()
        t_ok += (pf.argmax(-1) == y).float().tolist()
        sc = _attn_scores(api, student, x)
        for k in KS:
            for m in SCORES:
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, _topk(sc[m], k)).float().log_softmax(-1)
                per[(m, k)]["kl"] += (pf * (full - lm)).sum(-1).tolist()
                per[(m, k)]["top1_agree"] += (pf.argmax(-1) == lm.argmax(-1)).float().tolist()

    N = len(pmax)
    bins = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        idx = [i for i, p in enumerate(pmax) if lo <= p < hi]
        b = {"lo": lo, "hi": min(hi, 1.0), "n": len(idx), "frac": round(len(idx) / N, 4),
             "teacher_acc": round(100 * sum(t_ok[i] for i in idx) / max(len(idx), 1), 2)}
        for (m, k), d in per.items():
            for s, v in d.items():
                mu, se = _mean_se([v[i] for i in idx])
                b[f"{m}@{k}/{s}"] = None if mu is None else round(mu, 4)
                b[f"{m}@{k}/{s}_se"] = None if se is None else round(se, 4)
        bins.append(b)
    overall = {}
    for (m, k), d in per.items():
        for s, v in d.items():
            mu, se = _mean_se(v)
            overall[f"{m}@{k}/{s}"] = round(mu, 4)
            overall[f"{m}@{k}/{s}_se"] = round(se, 4)
    api.write_json("bins", bins)
    api.write_json("per_image", {"pmax": [round(p, 5) for p in pmax],
                                 "maskedkd@98_kl": [round(v, 5) for v in per[("maskedkd", 98)]["kl"]],
                                 "maskedkd@49_kl": [round(v, 5) for v in per[("maskedkd", 49)]["kl"]]})
    api.write_json("summary", {"dataset": ds, "student": rid, "n_val": N, "edges": list(EDGES),
                               "mean_pmax": round(sum(pmax) / N, 4), "overall": overall,
                               "bins": [{k: b[k] for k in ("lo", "hi", "n", "frac", "maskedkd@98/kl",
                                                             "maskedkd@98/kl_se")} for b in bins]})
