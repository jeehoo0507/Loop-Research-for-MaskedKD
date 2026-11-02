"""H-A8: is the student's last-block CLS attention more spread on FGVC-Aircraft than on CUB, so that MaskedKD's top-K
keep leaves more attention mass outside the kept tokens, and does that left-out mass track the per-image loss of
masked-teacher faithfulness (KL to the full teacher)? Run once per dataset (--dataset cub, --dataset aircraft).

Why: H-A6 found the MaskedKD masked teacher less faithful on Aircraft at the same K (maskedkd@98 KL 0.182 vs 0.071),
and H-A7 showed teacher confidence does not explain it. A plain candidate is attention spread: if the score mass
outside the top K is larger, a fixed K drops more of what the score itself calls relevant. This is a per-image,
position-free quantity (it only uses the score), and it is the descriptive side of "how many tokens per image".
Descriptive only: nothing is selected or tuned, no selector is changed (QUEUE 2026-11-02 08:50). Same student choice
(final seed-1 maskedkd@98 of that dataset), same maskedkd score and KL as analysis/fidelity_conf_matched.py.
"""
import math

import numpy as np
import torch

META = {
    "question": "is MaskedKD's last-block CLS attention more spread on FGVC-Aircraft than on CUB (more mass outside "
                "the top K), and does the left-out mass track per-image masked-teacher KL within each dataset?",
    "method": "per-image attention mass outside the top-K maskedkd keep and attention effective token count, "
              "Spearman with keep-only masked-teacher KL, bootstrap CI",
    "paper": "Hinton et al. 2015 (KD); Hooker et al. NeurIPS 2019 (keep-only evaluation); "
             "Son et al. ECCV 2024 (MaskedKD)",
}

KS = (98, 49)
N_BOOT = 2000


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _last_cls_attn(api, student, x):
    last = len(student.blocks) - 1
    with torch.no_grad():
        out = api.student_forward(student, x, attn_layers=[last])
    a = out.attn[last] if len(out.attn) > last else out.attn[-1]
    return a.float().mean(1)[:, 0, 1:]


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
    out_mass = {k: [] for k in KS}
    kl = {k: [] for k in KS}
    eff_n = []
    for x, _y in api.val_batches(batch=50):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        pf = full.exp()
        s = _last_cls_attn(api, student, x)
        p = s / s.sum(-1, keepdim=True)
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1)
        eff_n += ent.exp().tolist()
        for k in KS:
            m = _topk(s, k)
            out_mass[k] += (1.0 - (p * m).sum(-1)).tolist()
            with torch.no_grad():
                lm = api.teacher_forward_keepmask(T, x, m).float().log_softmax(-1)
            kl[k] += (pf * (full - lm)).sum(-1).tolist()

    rng = np.random.default_rng(0)
    N = len(eff_n)
    summary = {"dataset": ds, "student": rid, "n_val": N, "n_boot": N_BOOT}
    m, se = _mean_se(eff_n)
    summary["eff_tokens_mean"], summary["eff_tokens_se"] = round(m, 3), round(se, 3)
    for k in KS:
        a, b = np.asarray(out_mass[k]), np.asarray(kl[k])
        m, se = _mean_se(a)
        summary[f"out_mass@{k}_mean"], summary[f"out_mass@{k}_se"] = round(m, 5), round(se, 5)
        m, se = _mean_se(b)
        summary[f"kl@{k}_mean"], summary[f"kl@{k}_se"] = round(m, 5), round(se, 5)
        rho = _spearman(a, b)
        lo, hi = _boot_spearman(a, b, rng)
        summary[f"spearman_outmass_kl@{k}"] = round(rho, 4)
        summary[f"spearman_outmass_kl@{k}_ci"] = [round(lo, 4), round(hi, 4)]
    e = np.asarray(eff_n)
    for k in KS:
        rho = _spearman(e, np.asarray(kl[k]))
        lo, hi = _boot_spearman(e, np.asarray(kl[k]), rng)
        summary[f"spearman_efftok_kl@{k}"] = round(rho, 4)
        summary[f"spearman_efftok_kl@{k}_ci"] = [round(lo, 4), round(hi, 4)]
    api.write_json("per_image", {"eff_tokens": [round(v, 3) for v in eff_n],
                                 **{f"out_mass@{k}": [round(v, 5) for v in out_mass[k]] for k in KS},
                                 **{f"kl@{k}": [round(v, 5) for v in kl[k]] for k in KS}})
    api.write_json("summary", summary)
