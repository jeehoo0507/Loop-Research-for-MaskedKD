"""H-A10: does the H-A9 reading replicate on the other Aircraft MaskedKD students? H-A9 (seed-1 student only) found
that on Aircraft the student-score keep mask at K=98 loses more teacher soft label than the teacher's own last-block
CLS attention mask (KL student - teacher +0.099 [+0.071, +0.131]), while on CUB the two are the same (-0.016).
Every H-A6..H-A9 reading carries the caveat "seed 1 student". This runs the identical per-image computation of
analysis/score_source_fidelity.py on the Aircraft final maskedkd@98 students of seeds 2 and 3 (CUB has no final
maskedkd@98 seed 2/3, so only Aircraft). Run once with --dataset aircraft.
Descriptive only: nothing is selected or tuned, no selector is changed (QUEUE 2026-11-02 08:50).
"""
import math

import numpy as np
import torch

META = {
    "question": "on FGVC-Aircraft final maskedkd@98 students of seeds 2 and 3, does the student-score top-98 keep mask "
                "still lose more masked-teacher KL than the teacher's own last-block CLS attention top-98 mask, as for "
                "seed 1 (H-A9)?",
    "method": "last-block CLS attention (student and teacher), top-K Jaccard overlap, keep-only masked-teacher KL "
              "for student / teacher / random scores, bootstrap CI; same code path as score_source_fidelity.py",
    "paper": "Hinton et al. 2015 (KD); Hooker et al. NeurIPS 2019 (keep-only evaluation); "
             "Son et al. ECCV 2024 (MaskedKD)",
}

SEEDS = (2, 3)
K = 98
N_BOOT = 2000


# helpers copied unchanged from analysis/score_source_fidelity.py (analyses may not import each other)
def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _last_cls_attn(api, model, x):
    last = len(model.blocks) - 1
    with torch.no_grad():
        out = api.student_forward(model, x, attn_layers=[last])
    return out.attn[last].float().mean(1)[:, 0, 1:]


def _rank(v):
    r = np.empty(len(v))
    r[np.argsort(v, kind="mergesort")] = np.arange(len(v))
    return r


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def _boot_spearman(a, b, rng):
    n = len(a)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        vals.append(_spearman(a[i], b[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _boot_mean(v, rng):
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    vals = [v[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _mean_se(v):
    v = np.asarray(v, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def _one(api, rid):
    T = api.teacher.eval()
    student = api.student(rid)
    rng_scores = np.random.default_rng(0)
    kl = {"student": [], "teacher": [], "random": []}
    jac = []
    for x, _y in api.val_batches(batch=50):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        pf = full.exp()
        masks = {
            "student": _topk(_last_cls_attn(api, student, x), K),
            "teacher": _topk(_last_cls_attn(api, T, x), K),
            "random": _topk(torch.from_numpy(rng_scores.random((x.shape[0], api.num_patches))).to(x.device), K),
        }
        inter = (masks["student"] & masks["teacher"]).sum(1).float()
        union = (masks["student"] | masks["teacher"]).sum(1).float()
        jac += (inter / union).tolist()
        for name, m in masks.items():
            with torch.no_grad():
                lm = api.teacher_forward_keepmask(T, x, m).float().log_softmax(-1)
            kl[name] += (pf * (full - lm)).sum(-1).tolist()

    rng = np.random.default_rng(0)
    s = {"student": rid, "n_val": len(jac), "n_boot": N_BOOT, "K": K}
    m, se = _mean_se(jac)
    s["jaccard_mean"], s["jaccard_se"] = round(m, 5), round(se, 5)
    for name, v in kl.items():
        m, se = _mean_se(v)
        s[f"kl_{name}_mean"], s[f"kl_{name}_se"] = round(m, 5), round(se, 5)
    d = np.asarray(kl["student"]) - np.asarray(kl["teacher"])
    m, se = _mean_se(d)
    lo, hi = _boot_mean(d, rng)
    s["kl_student_minus_teacher_mean"], s["kl_student_minus_teacher_se"] = round(m, 5), round(se, 5)
    s["kl_student_minus_teacher_ci"] = [round(lo, 5), round(hi, 5)]
    rho = _spearman(-np.asarray(jac), np.asarray(kl["student"]))
    lo, hi = _boot_spearman(-np.asarray(jac), np.asarray(kl["student"]), rng)
    s["spearman_disagree_kl_student"] = round(rho, 4)
    s["spearman_disagree_kl_student_ci"] = [round(lo, 4), round(hi, 4)]
    return s, {"jaccard": [round(v, 5) for v in jac], **{f"kl_{n}": [round(v, 5) for v in kl[n]] for n in kl}}


def run(api):
    ds = api.dataset
    runs = api.runs()
    found = {}
    for r_id, r in sorted(runs.items()):
        if (r.get("dataset") == ds and r["seed"] in SEEDS and r["tier"] == "final"
                and r["method"] == "maskedkd" and int(r.get("keep") or 0) == 98 and r["seed"] not in found):
            found[r["seed"]] = r_id
    api.write_json("students", {"dataset": ds, "runs": {str(k): v for k, v in found.items()}})
    summary = {"dataset": ds}
    for seed in SEEDS:
        if seed not in found:
            continue
        s, per = _one(api, found[seed])
        summary[f"seed{seed}"] = s
        api.write_json(f"per_image_seed{seed}", per)
    api.write_json("summary", summary)
