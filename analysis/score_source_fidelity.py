"""H-A9: is the Aircraft-vs-CUB loss of MaskedKD masked-teacher faithfulness (H-A6) about the student's score, or
about the dataset/teacher? At K=98, compare keep masks picked by (s) the student's last-block CLS attention
(= maskedkd), (t) the full teacher's own last-block CLS attention, and (r) random scores; and measure how much the
student's top-98 agrees with the teacher's top-98. Run once per dataset (--dataset cub, --dataset aircraft).

Why: H-A6 found maskedkd@98 KL 0.182 on Aircraft vs 0.071 on CUB; H-A7 ruled out teacher confidence; H-A8 found the
student score a bit more spread on Aircraft, but that explains only part of the gap. MaskedKD's premise is that the
student's attention points where the teacher needs tokens. If the student and teacher agree less on Aircraft, and a
teacher-attention mask keeps the Aircraft teacher about as faithful as on CUB, the gap is a student-score gap.
Descriptive only: nothing is selected or tuned, no selector is changed (QUEUE 2026-11-02 08:50). Same student choice
(final seed-1 maskedkd@98 of that dataset) and same KL as analysis/attn_mass_fidelity.py.
"""
import math

import numpy as np
import torch

META = {
    "question": "at K=98, does the student's last-block CLS attention top-K agree less with the teacher's own "
                "last-block CLS attention top-K on FGVC-Aircraft than on CUB, and does a teacher-attention keep mask "
                "close the Aircraft-vs-CUB gap in masked-teacher KL that the student-score mask shows?",
    "method": "last-block CLS attention (student and teacher), top-K Jaccard overlap, keep-only masked-teacher KL "
              "for student / teacher / random scores, bootstrap CI",
    "paper": "Hinton et al. 2015 (KD); Hooker et al. NeurIPS 2019 (keep-only evaluation); "
             "Son et al. ECCV 2024 (MaskedKD)",
}

K = 98
N_BOOT = 2000


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
    N = len(jac)
    summary = {"dataset": ds, "student": rid, "n_val": N, "n_boot": N_BOOT, "K": K}
    m, se = _mean_se(jac)
    summary["jaccard_mean"], summary["jaccard_se"] = round(m, 5), round(se, 5)
    for name, v in kl.items():
        m, se = _mean_se(v)
        summary[f"kl_{name}_mean"], summary[f"kl_{name}_se"] = round(m, 5), round(se, 5)
    d = np.asarray(kl["student"]) - np.asarray(kl["teacher"])
    m, se = _mean_se(d)
    lo, hi = _boot_mean(d, rng)
    summary["kl_student_minus_teacher_mean"], summary["kl_student_minus_teacher_se"] = round(m, 5), round(se, 5)
    summary["kl_student_minus_teacher_ci"] = [round(lo, 5), round(hi, 5)]
    rho = _spearman(-np.asarray(jac), np.asarray(kl["student"]))
    lo, hi = _boot_spearman(-np.asarray(jac), np.asarray(kl["student"]), rng)
    summary["spearman_disagree_kl_student"] = round(rho, 4)
    summary["spearman_disagree_kl_student_ci"] = [round(lo, 4), round(hi, 4)]
    api.write_json("per_image", {"jaccard": [round(v, 5) for v in jac],
                                 **{f"kl_{n}": [round(v, 5) for v in kl[n]] for n in kl}})
    api.write_json("summary", summary)
