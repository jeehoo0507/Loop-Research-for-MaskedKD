"""H-A12: before any Aircraft lateroll run, does roll6 (late-block rollout = selectors/lateroll.py, unchanged) keep the
masked teacher's soft label (KL, DKD TCKD) better than the maskedkd ranking (roll11) and full rollout (roll0) at the
training budget K=29, on three students per dataset? H-A1 saw this on Aircraft with seed-1 gate students only
(ce, full, maskedkd@98). Here the students are the final maskedkd@29 runs of seeds 1, 2, 3 on both datasets (CUB is
the anchor where lateroll@29 accuracy is known). Run once with --dataset cub and once with --dataset aircraft.
Descriptive proxy prediction only: nothing is selected or tuned, lateroll is not changed (QUEUE 2026-11-02 08:50).
Rollout code path copied from analysis/aircraft_lateroll_proxy.py (analyses may not import each other); the random
reference and the bootstrap use numpy default_rng.
"""
import numpy as np
import torch

META = {
    "question": "on CUB and FGVC-Aircraft final maskedkd@29 students of seeds 1, 2, 3, at K=29, does roll6 (lateroll) "
                "give lower masked-teacher KL / TCKD than roll11 (maskedkd ranking), roll0 (full rollout) and "
                "random, with paired bootstrap CIs?",
    "method": "keep-only masked-teacher faithfulness (KL, DKD TCKD), attention rollout start-block variants, paired "
              "bootstrap over val images",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Zhao et al. CVPR 2022 (DKD); Son et al. ECCV 2024 (MaskedKD); "
             "Hooker et al. NeurIPS 2019 (keep-only evaluation)",
}

SEEDS = (1, 2, 3)
K = 29
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
                and r["method"] == "maskedkd" and int(r.get("keep") or 0) == K and r["seed"] not in found):
            found[r["seed"]] = rid
    api.write_json("students", {"dataset": ds, "runs": {str(k): v for k, v in found.items()}})
    T = api.teacher.eval()
    n = api.num_patches
    rng_mask = np.random.default_rng(0)
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        r_sc = torch.from_numpy(rng_mask.random((x.shape[0], n))).to(x.device)
        with torch.no_grad():
            lm = api.teacher_forward_keepmask(T, x, _topk(r_sc, K)).float().log_softmax(-1)
        batches.append((x, y, full, _stats(full, lm, y)))
    N = sum(b[0].shape[0] for b in batches)
    rand_kl = torch.cat([b[3][0] for b in batches]).cpu().numpy()
    rand_tc = torch.cat([b[3][1] for b in batches]).cpu().numpy()
    idx = np.random.default_rng(0).integers(0, N, size=(N_BOOT, N))

    summary = {"dataset": ds, "K": K, "n_val": N, "n_boot": N_BOOT,
               "random": {"kl_mean": round(float(rand_kl.mean()), 5), "tckd_mean": round(float(rand_tc.mean()), 5)}}
    for seed in SEEDS:
        if seed not in found:
            continue
        student = api.student(found[seed])
        per = {m: ([], []) for m in SCORES}
        for x, y, full, _r in batches:
            sc = _rollouts(api, student, x)
            for m in SCORES:
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, _topk(sc[m], K)).float().log_softmax(-1)
                kl, tc = _stats(full, lm, y)
                per[m][0].append(kl)
                per[m][1].append(tc)
        per = {m: (torch.cat(v[0]).cpu().numpy(), torch.cat(v[1]).cpu().numpy()) for m, v in per.items()}
        s = {"student": found[seed]}
        for m in SCORES:
            s[f"{m}_kl_mean"] = round(float(per[m][0].mean()), 5)
            s[f"{m}_tckd_mean"] = round(float(per[m][1].mean()), 5)
        r6kl, r6tc = per["roll6"]
        refs = {"roll11": per["roll11"], "roll0": per["roll0"], "random": (rand_kl, rand_tc)}
        for ref, (kl, tc) in refs.items():
            s[f"d_kl_roll6_minus_{ref}"] = _ci(r6kl - kl, idx)
            s[f"d_tckd_roll6_minus_{ref}"] = _ci(r6tc - tc, idx)
        summary[f"seed{seed}"] = s
        api.write_json(f"per_image_seed{seed}", {f"{m}_{q}": [round(float(v), 5) for v in per[m][i]]
                                                  for m in SCORES for i, q in enumerate(("kl", "tckd"))})
    api.write_json("summary", summary)
