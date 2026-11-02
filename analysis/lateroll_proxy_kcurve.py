"""H-A13: does the lateroll (roll6) proxy gap over the maskedkd ranking (roll11) and full rollout (roll0) shrink as
the budget grows? Same six students as H-A12 (final maskedkd@29, seeds 1, 2, 3, per dataset), masks at K = 12, 29, 49.
CUB is the anchor: lateroll beats maskedkd / rollout in accuracy at K=29 (CI above 0) but not at K=49 (CI covers 0).
Run once with --dataset cub and once with --dataset aircraft. Descriptive proxy only: nothing is selected or tuned,
lateroll is not changed (QUEUE 2026-11-02 08:50). Rollout code copied from analysis/lateroll_proxy_k29students.py
(analyses may not import each other); bootstrap uses numpy default_rng.
"""
import numpy as np
import torch

META = {
    "question": "on CUB and FGVC-Aircraft final maskedkd@29 students of seeds 1, 2, 3, does the masked-teacher KL gap "
                "roll11 - roll6 and roll0 - roll6 shrink from K=29 to K=49 (and how large is it at K=12)?",
    "method": "keep-only masked-teacher faithfulness (KL, DKD TCKD) at several budgets, attention rollout start-block "
              "variants, paired bootstrap over val images of the K=49 minus K=29 gap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Zhao et al. CVPR 2022 (DKD); Son et al. ECCV 2024 (MaskedKD); "
             "Hooker et al. NeurIPS 2019 (keep-only evaluation)",
}

SEEDS = (1, 2, 3)
KS = (12, 29, 49)
STUDENT_K = 29
EPS = 1e-8
N_BOOT = 2000
SCORES = ("roll6", "roll0", "roll11")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _rollouts(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad():
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
        A = [a.float().mean(1) for a in out.attn]
        B, N, _ = A[0].shape
        eye = torch.eye(N, device=A[0].device).expand(B, N, N)
        res, R = {}, None
        for s in range(depth - 1, -1, -1):
            a = 0.5 * A[s] + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else R @ a
            if f"roll{s}" in SCORES:
                res[f"roll{s}"] = R[:, 0, 1:].clone()
    return res


def _stats(full_lp, mask_lp, y):
    pf, pm = full_lp.exp(), mask_lp.exp()
    kl = (pf * (full_lp - mask_lp)).sum(-1)
    pt_f = pf.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    pt_m = pm.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()
    return kl, tckd


def _ci(d, idx):
    d = np.asarray(d, dtype=np.float64)
    m = d[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return [round(float(d.mean()), 5), round(float(lo), 5), round(float(hi), 5)]


def run(api):
    ds = api.dataset
    found = {}
    for rid, r in sorted(api.runs().items()):
        if (r.get("dataset") == ds and r["seed"] in SEEDS and r["tier"] == "final"
                and r["method"] == "maskedkd" and int(r.get("keep") or 0) == STUDENT_K and r["seed"] not in found):
            found[r["seed"]] = rid
    api.write_json("students", {"dataset": ds, "runs": {str(k): v for k, v in found.items()}})
    T = api.teacher.eval()
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        batches.append((x, y, full))
    N = sum(b[0].shape[0] for b in batches)
    idx = np.random.default_rng(0).integers(0, N, size=(N_BOOT, N))

    summary = {"dataset": ds, "Ks": list(KS), "n_val": N, "n_boot": N_BOOT}
    for seed in SEEDS:
        if seed not in found:
            continue
        student = api.student(found[seed])
        per = {(m, k): ([], []) for m in SCORES for k in KS}
        for x, y, full in batches:
            sc = _rollouts(api, student, x)
            for m in SCORES:
                for k in KS:
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, _topk(sc[m], k)).float().log_softmax(-1)
                    kl, tc = _stats(full, lm, y)
                    per[(m, k)][0].append(kl)
                    per[(m, k)][1].append(tc)
        per = {key: (torch.cat(v[0]).cpu().numpy(), torch.cat(v[1]).cpu().numpy()) for key, v in per.items()}
        s = {"student": found[seed]}
        for k in KS:
            for m in SCORES:
                s[f"k{k}_{m}_kl_mean"] = round(float(per[(m, k)][0].mean()), 5)
                s[f"k{k}_{m}_tckd_mean"] = round(float(per[(m, k)][1].mean()), 5)
            for ref in ("roll11", "roll0"):
                s[f"k{k}_gap_kl_{ref}_minus_roll6"] = _ci(per[(ref, k)][0] - per[("roll6", k)][0], idx)
                s[f"k{k}_gap_tckd_{ref}_minus_roll6"] = _ci(per[(ref, k)][1] - per[("roll6", k)][1], idx)
        for ref in ("roll11", "roll0"):
            g29 = per[(ref, 29)][0] - per[("roll6", 29)][0]
            g49 = per[(ref, 49)][0] - per[("roll6", 49)][0]
            s[f"shrink_kl_{ref}_k49_minus_k29"] = _ci(g49 - g29, idx)
            s[f"ratio_kl_{ref}_over_roll6_k29"] = round(float(per[(ref, 29)][0].mean() / per[("roll6", 29)][0].mean()), 5)
            s[f"ratio_kl_{ref}_over_roll6_k49"] = round(float(per[(ref, 49)][0].mean() / per[("roll6", 49)][0].mean()), 5)
        summary[f"seed{seed}"] = s
    api.write_json("summary", summary)
