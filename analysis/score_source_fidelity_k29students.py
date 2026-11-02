"""H-A11: is the H-A9/H-A10 dataset contrast (Aircraft student-score mask loses more masked-teacher KL than the
teacher's own last-block CLS attention mask; CUB does not) seen with three students on BOTH datasets? CUB has only a
seed-1 final maskedkd@98 student, but both datasets have final maskedkd@29 students for seeds 1, 2, 3. This runs the
per-image computation of analysis/score_source_fidelity.py on those six students, at K=98 (as H-A9) and K=29 (their
training budget). Run once with --dataset cub and once with --dataset aircraft.
Descriptive only: nothing is selected or tuned, no selector is changed (QUEUE 2026-11-02 08:50).
"""
import math

import numpy as np
import torch

META = {
    "question": "on CUB and FGVC-Aircraft final maskedkd@29 students of seeds 1, 2, 3, does the student-score top-K "
                "keep mask lose more masked-teacher KL than the teacher's own last-block CLS attention top-K mask "
                "on Aircraft but not on CUB (H-A9/H-A10 contrast), at K=98 and K=29?",
    "method": "last-block CLS attention (student and teacher), top-K Jaccard overlap, keep-only masked-teacher KL "
              "for student / teacher / random scores, bootstrap CI; same code path as score_source_fidelity.py",
    "paper": "Hinton et al. 2015 (KD); Hooker et al. NeurIPS 2019 (keep-only evaluation); "
             "Son et al. ECCV 2024 (MaskedKD)",
}

SEEDS = (1, 2, 3)
TRAIN_K = 29
KS = (98, 29)
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
    kl = {k: {"student": [], "teacher": [], "random": []} for k in KS}
    jac = {k: [] for k in KS}
    for x, _y in api.val_batches(batch=50):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        pf = full.exp()
        s_sc = _last_cls_attn(api, student, x)
        t_sc = _last_cls_attn(api, T, x)
        r_sc = torch.from_numpy(rng_scores.random((x.shape[0], api.num_patches))).to(x.device)
        for k in KS:
            masks = {"student": _topk(s_sc, k), "teacher": _topk(t_sc, k), "random": _topk(r_sc, k)}
            inter = (masks["student"] & masks["teacher"]).sum(1).float()
            union = (masks["student"] | masks["teacher"]).sum(1).float()
            jac[k] += (inter / union).tolist()
            for name, m in masks.items():
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, m).float().log_softmax(-1)
                kl[k][name] += (pf * (full - lm)).sum(-1).tolist()
    out, per = {"student": rid}, {}
    for k in KS:
        s, p = _stats(rid, k, jac[k], kl[k])
        out[f"k{k}"] = s
        per.update({f"k{k}_{n}": v for n, v in p.items()})
    return out, per


def _stats(rid, K, jac, kl):
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
                and r["method"] == "maskedkd" and int(r.get("keep") or 0) == TRAIN_K and r["seed"] not in found):
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
