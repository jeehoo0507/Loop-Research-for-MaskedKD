"""H-A19: can CUB final val (600 images) resolve the KD gain, and does Full KD transfer the teacher even where the
accuracy gain is uncertain? Students: CUB final CE (k0) and Full KD (k196), seeds 1, 2, 3 (H-C33 verdict:
Full - CE +0.50 [-3.04, +4.04] over seeds). Per val image:
  (1) seed-paired Full vs CE discordance (b = Full right & CE wrong, c = the reverse), exact McNemar p;
  (2) same-method different-seed discordance (CE vs CE, Full vs Full) as the seed-noise reference;
  (3) image bootstrap of the seed-averaged Full - CE accuracy (val-sampling noise only, these 6 students fixed);
  (4) agreement with the teacher argmax and KL(T || S), Full - CE, image bootstrap per seed;
  (5) Full - CE accuracy split by teacher correct / teacher wrong.
Descriptive only: nothing is selected or tuned, H-C33 is not re-judged, lateroll is not changed (QUEUE 2026-11-02
08:50). Helpers copied from analysis/signal_outcome_null.py (analyses may not import each other). Bootstrap uses
numpy default_rng.
"""
import math

import numpy as np
import torch

META = {
    "question": "on CUB final seeds 1, 2, 3, is the Full KD - CE accuracy difference resolvable on 600 val images "
                "(per-image discordance vs seed noise), and do Full KD students agree with the teacher more than CE?",
    "method": "per-image paired correctness (McNemar), same-method seed-pair discordance as noise reference, "
              "image bootstrap of seed-averaged difference, teacher-argmax agreement and KL(T||S)",
    "paper": "Hinton et al. 2015 (KD); Stanton et al. NeurIPS 2021 (student-teacher fidelity); "
             "Dietterich 1998 (McNemar for classifier comparison); Bouthillier et al. MLSys 2021 (variance sources)",
}

SEEDS = (1, 2, 3)
METHODS = {"ce": 0, "full": 196}
PAIRS = ((1, 2), (1, 3), (2, 3))
N_BOOT = 2000


def _kl(p_lp, q_lp):
    return (p_lp.exp() * (p_lp - q_lp)).sum(-1)


def _mcnemar_p(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return round(min(1.0, 2.0 * tail), 5)


def _boot_mean(v, idx):
    vals = v[idx].mean(1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(v.mean()), 5), round(float(lo), 5), round(float(hi), 5)]


def _disc(a, b):
    return {"a_right_b_wrong": int(((a == 1) & (b == 0)).sum()), "a_wrong_b_right": int(((a == 0) & (b == 1)).sum()),
            "discordant_frac": round(float((a != b).mean()), 5),
            "mcnemar_p": _mcnemar_p(int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()))}


def run(api):
    ds = api.dataset
    found = {}
    for rid, r in sorted(api.runs().items()):
        if r.get("dataset") != ds or r["tier"] != "final" or r["seed"] not in SEEDS:
            continue
        if r["method"] in METHODS and int(r.get("keep") or 0) == METHODS[r["method"]]:
            found.setdefault((r["method"], r["seed"]), (rid, r.get("val_acc_last")))
    api.write_json("students", {"dataset": ds, "runs": {f"{m}_s{s}": v for (m, s), v in found.items()}})
    T = api.teacher.eval()
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            batches.append((x, y, T(x).float().log_softmax(-1)))
    t_lp_all = torch.cat([lp for _x, _y, lp in batches])
    y_all = torch.cat([y for _x, y, _lp in batches])
    t_arg = t_lp_all.argmax(-1)
    t_cor = (t_arg == y_all).cpu().numpy().astype(np.int64)
    N = len(t_cor)
    idx = np.random.default_rng(0).integers(0, N, size=(N_BOOT, N))

    summary = {"dataset": ds, "n_val": N, "n_boot": N_BOOT, "teacher_val_acc": round(100 * float(t_cor.mean()), 3),
               "students": {}}
    cols = {}
    for (m, s), (rid, acc_last) in sorted(found.items()):
        student = api.student(rid)
        lps = []
        for x, _y, _t in batches:
            with torch.no_grad():
                lps.append(api.student_forward(student, x).logits.float().log_softmax(-1))
        s_lp = torch.cat(lps)
        c = {"cor": (s_lp.argmax(-1) == y_all).cpu().numpy().astype(np.int64),
             "agree": (s_lp.argmax(-1) == t_arg).cpu().numpy().astype(np.int64),
             "kl": _kl(t_lp_all, s_lp).cpu().numpy().astype(np.float64)}
        cols[(m, s)] = c
        summary["students"][f"{m}_s{s}"] = {
            "run": rid, "val_acc_last": acc_last, "val_acc_here": round(100 * float(c["cor"].mean()), 3),
            "teacher_agree": round(100 * float(c["agree"].mean()), 3), "kl_t_s": round(float(c["kl"].mean()), 5)}
        del student
        torch.cuda.empty_cache()

    have = [s for s in SEEDS if ("ce", s) in cols and ("full", s) in cols]
    summary["full_vs_ce_same_seed"] = {f"seed{s}": _disc(cols[("full", s)]["cor"], cols[("ce", s)]["cor"])
                                       for s in have}
    summary["same_method_seed_pairs"] = {
        f"{m}_s{a}_vs_s{b}": _disc(cols[(m, a)]["cor"], cols[(m, b)]["cor"])
        for m in METHODS for a, b in PAIRS if (m, a) in cols and (m, b) in cols}
    fc = [v["discordant_frac"] for v in summary["full_vs_ce_same_seed"].values()]
    sm = [v["discordant_frac"] for v in summary["same_method_seed_pairs"].values()]
    summary["discordance_ratio_fullce_over_samemethod"] = (
        round(float(np.mean(fc) / np.mean(sm)), 4) if fc and sm else None)

    if have:
        d = np.mean([cols[("full", s)]["cor"] - cols[("ce", s)]["cor"] for s in have], axis=0).astype(np.float64)
        summary["seedavg_full_minus_ce_acc_image_boot"] = [round(100 * v, 3) for v in _boot_mean(d, idx)]
        summary["by_teacher"] = {
            name: {"n": int(mask.sum()),
                   "full_minus_ce_acc_pp": round(100 * float(d[mask].mean()), 3) if mask.any() else None}
            for name, mask in (("teacher_right", t_cor == 1), ("teacher_wrong", t_cor == 0))}
        summary["by_teacher_diff_right_minus_wrong_image_boot"] = None
        if (t_cor == 1).any() and (t_cor == 0).any():
            vals = []
            for i in idx:
                dd, tc = d[i], t_cor[i]
                if (tc == 1).any() and (tc == 0).any():
                    vals.append(dd[tc == 1].mean() - dd[tc == 0].mean())
            lo, hi = np.percentile(vals, [2.5, 97.5])
            pt = d[t_cor == 1].mean() - d[t_cor == 0].mean()
            summary["by_teacher_diff_right_minus_wrong_image_boot"] = [
                round(100 * float(pt), 3), round(100 * float(lo), 3), round(100 * float(hi), 3)]
    summary["teacher_transfer_full_minus_ce"] = {
        f"seed{s}": {
            "agree_pp": [round(100 * v, 3) for v in _boot_mean(
                (cols[("full", s)]["agree"] - cols[("ce", s)]["agree"]).astype(np.float64), idx)],
            "kl_t_s": _boot_mean(cols[("full", s)]["kl"] - cols[("ce", s)]["kl"], idx)}
        for s in have}
    api.write_json("summary", summary)
