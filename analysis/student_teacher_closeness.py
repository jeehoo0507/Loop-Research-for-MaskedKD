"""H-A14: do the trained CUB lateroll@29 students end closer to the full teacher than the maskedkd@29 and rollout@29
students of the same seed? H-A1/H-A12/H-A13 measured the masked teacher (the training signal); this measures the
outcome of training: KL(full teacher || student) on val, its target (TCKD) and non-target (NCKD) parts, and top-1
agreement. Final students of seeds 1, 2, 3 for lateroll@29, maskedkd@29, rollout@29, full, ce (whichever exist).
Descriptive only: nothing is selected or tuned, lateroll is not changed (QUEUE 2026-11-02 08:50). Bootstrap uses
numpy default_rng.
"""
import numpy as np
import torch

META = {
    "question": "on CUB final students of seeds 1, 2, 3, is KL(full teacher || student) on val lower for lateroll@29 "
                "than for maskedkd@29 and rollout@29 of the same seed, and does closeness to the teacher track "
                "val accuracy across methods?",
    "method": "teacher-student output divergence on held-out images, decoupled into target (TCKD) and non-target "
              "(NCKD) parts, paired bootstrap over val images",
    "paper": "Hinton et al. 2015 (KD); Zhao et al. CVPR 2022 (DKD); Stanton et al. NeurIPS 2021 (does KD really "
             "work? student-teacher fidelity)",
}

SEEDS = (1, 2, 3)
ARMS = (("lateroll", 29), ("maskedkd", 29), ("rollout", 29), ("full", None), ("ce", None))
EPS = 1e-8
N_BOOT = 2000


def _per_image(t_lp, s_lp, y):
    pt, ps = t_lp.exp(), s_lp.exp()
    kl = (pt * (t_lp - s_lp)).sum(-1)
    a = pt.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    b = ps.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = a * (a / b).log() + (1 - a) * ((1 - a) / (1 - b)).log()
    nt = torch.ones_like(pt, dtype=torch.bool).scatter_(1, y[:, None], False)
    qt = (pt * nt) / (pt * nt).sum(-1, keepdim=True).clamp_min(EPS)
    qs = (ps * nt) / (ps * nt).sum(-1, keepdim=True).clamp_min(EPS)
    nckd = (qt * (qt.clamp_min(EPS).log() - qs.clamp_min(EPS).log())).sum(-1)
    agree = (pt.argmax(-1) == ps.argmax(-1)).float()
    correct = (ps.argmax(-1) == y).float()
    return {"kl": kl, "tckd": tckd, "nckd": nckd, "agree": agree, "correct": correct}


def _ci(d, idx):
    d = np.asarray(d, dtype=np.float64)
    m = d[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return [round(float(d.mean()), 5), round(float(lo), 5), round(float(hi), 5)]


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def run(api):
    ds = api.dataset
    found = {}
    for rid, r in sorted(api.runs().items()):
        if r.get("dataset") != ds or r["tier"] != "final" or r["seed"] not in SEEDS:
            continue
        for m, k in ARMS:
            if r["method"] == m and (k is None or int(r.get("keep") or 0) == k):
                found.setdefault((m, r["seed"]), (rid, r.get("val_acc_last")))
    api.write_json("students", {"dataset": ds, "runs": {f"{m}_s{s}": v for (m, s), v in found.items()}})
    T = api.teacher.eval()
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            batches.append((x, y, T(x).float().log_softmax(-1)))
    N = sum(b[0].shape[0] for b in batches)
    idx = np.random.default_rng(0).integers(0, N, size=(N_BOOT, N))

    per = {}
    summary = {"dataset": ds, "n_val": N, "n_boot": N_BOOT, "students": {}}
    for (m, s), (rid, acc_last) in sorted(found.items()):
        student = api.student(rid)
        acc = {key: [] for key in ("kl", "tckd", "nckd", "agree", "correct")}
        for x, y, t_lp in batches:
            with torch.no_grad():
                s_lp = api.student_forward(student, x).logits.float().log_softmax(-1)
            for key, v in _per_image(t_lp, s_lp, y).items():
                acc[key].append(v)
        acc = {key: torch.cat(v).cpu().numpy().astype(np.float64) for key, v in acc.items()}
        per[(m, s)] = acc
        summary["students"][f"{m}_s{s}"] = {
            "run": rid, "val_acc_last": acc_last, "val_acc_here": round(100 * float(acc["correct"].mean()), 3),
            "kl": round(float(acc["kl"].mean()), 5), "tckd": round(float(acc["tckd"].mean()), 5),
            "nckd": round(float(acc["nckd"].mean()), 5), "agree": round(100 * float(acc["agree"].mean()), 3)}
        del student
        torch.cuda.empty_cache()

    for s in SEEDS:
        if ("lateroll", s) not in per:
            continue
        L = per[("lateroll", s)]
        g = {}
        for ref in ("maskedkd", "rollout", "full", "ce"):
            if (ref, s) not in per:
                continue
            R = per[(ref, s)]
            for key in ("kl", "tckd", "nckd"):
                g[f"{key}_{ref}_minus_lateroll"] = _ci(R[key] - L[key], idx)
            g[f"agree_lateroll_minus_{ref}"] = _ci(100 * (L["agree"] - R["agree"]), idx)
        summary[f"seed{s}_gaps"] = g

    keys = sorted(summary["students"])
    accs = [summary["students"][k]["val_acc_here"] for k in keys]
    kls = [summary["students"][k]["kl"] for k in keys]
    kd = [k for k in keys if not k.startswith("ce_")]
    summary["spearman_kl_vs_acc_all"] = round(_spearman(kls, accs), 4) if len(keys) > 2 else None
    summary["spearman_kl_vs_acc_kd_only"] = round(_spearman([summary["students"][k]["kl"] for k in kd],
                                                            [summary["students"][k]["val_acc_here"] for k in kd]),
                                                  4) if len(kd) > 2 else None
    summary["n_students"] = len(keys)
    api.write_json("summary", summary)
