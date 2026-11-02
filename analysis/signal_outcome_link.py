"""H-A17: is lateroll's training-signal advantage (its K=29 mask keeps the teacher's soft label better than the maskedkd
ranking, H-A12) found on the same val images where the trained lateroll@29 student ends closer to the full teacher than
the maskedkd@29 student of the same seed (H-A14)? Per val image, seeds 1, 2, 3 on CUB:
  g = KL(T || T masked by the maskedkd student's own last-block ranking) - KL(T || T masked by the lateroll student's
      own roll6 mask)                                   (signal advantage, each student scores its own mask)
  o = KL(T || maskedkd student) - KL(T || lateroll student)   (outcome advantage)
Spearman(g, o), and a partial Spearman controlling for image difficulty (teacher entropy, mean signal KL, mean
outcome KL), because both differences grow with difficulty even without any link. Val images were never trained on,
so this is an image-type link, not a causal one. Descriptive only: nothing is selected or tuned, lateroll is not
changed (QUEUE 2026-11-02 08:50). Rollout and KL helpers copied from analysis/lateroll_proxy_k29students.py and
analysis/student_teacher_closeness.py (analyses may not import each other). Bootstrap uses numpy default_rng.
"""
import numpy as np
import torch

META = {
    "question": "on CUB final lateroll@29 / maskedkd@29 students of seeds 1, 2, 3, does the per-image masked-teacher "
                "KL advantage of the lateroll mask rank-correlate with the per-image student-teacher KL advantage of "
                "the lateroll student, also after controlling for image difficulty?",
    "method": "keep-only masked-teacher faithfulness (KL) linked to student-teacher output divergence per image; "
              "Spearman and rank partial correlation with paired bootstrap over val images",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Stanton et al. NeurIPS 2021 "
             "(student-teacher fidelity); Hooker et al. NeurIPS 2019 (keep-only evaluation)",
}

SEEDS = (1, 2, 3)
K = 29
START = 6
N_BOOT = 2000


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _rollout(api, student, x, start):
    depth = len(student.blocks)
    with torch.no_grad():
        out = api.student_forward(student, x, attn_layers=list(range(start, depth)))
        if start == depth - 1:                       # maskedkd baseline exactly: last-block head-mean CLS row
            return out.attn[start].float().mean(1)[:, 0, 1:].clone()
        R = None
        for s in range(depth - 1, start - 1, -1):
            a = out.attn[s].float().mean(1)
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else R @ a
    return R[:, 0, 1:].clone()


def _kl(p_lp, q_lp):
    return (p_lp.exp() * (p_lp - q_lp)).sum(-1)


def _rank(v):
    return np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(np.float64)


def _spearman(a, b):
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _partial(a, b, ctrl):
    Z = np.column_stack([np.ones(len(a))] + [_rank(c) for c in ctrl])
    ra, rb = _rank(a), _rank(b)
    ea = ra - Z @ np.linalg.lstsq(Z, ra, rcond=None)[0]
    eb = rb - Z @ np.linalg.lstsq(Z, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])


def _boot(fn, arrays, idx):
    vals = np.array([fn(*[a[i] for a in arrays]) for i in idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(fn(*arrays), 4), round(float(lo), 4), round(float(hi), 4)]


def run(api):
    ds = api.dataset
    found = {}
    for rid, r in sorted(api.runs().items()):
        if r.get("dataset") != ds or r["tier"] != "final" or r["seed"] not in SEEDS:
            continue
        for m in ("lateroll", "maskedkd"):
            if r["method"] == m and int(r.get("keep") or 0) == K:
                found.setdefault((m, r["seed"]), (rid, r.get("val_acc_last")))
    api.write_json("students", {"dataset": ds, "runs": {f"{m}_s{s}": v for (m, s), v in found.items()}})
    T = api.teacher.eval()
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            batches.append((x, y, T(x).float().log_softmax(-1)))
    t_ent = torch.cat([-(lp.exp() * lp).sum(-1) for _x, _y, lp in batches]).cpu().numpy().astype(np.float64)
    N = len(t_ent)
    idx = np.random.default_rng(0).integers(0, N, size=(N_BOOT, N))

    summary = {"dataset": ds, "K": K, "n_val": N, "n_boot": N_BOOT, "students": {}}
    per = {}
    for s in SEEDS:
        if ("lateroll", s) not in found or ("maskedkd", s) not in found:
            continue
        cols = {}
        for m, start in (("lateroll", START), ("maskedkd", None)):
            rid, acc_last = found[(m, s)]
            student = api.student(rid)
            sig, out, cor = [], [], []
            for x, y, t_lp in batches:
                with torch.no_grad():
                    s_lp = api.student_forward(student, x).logits.float().log_softmax(-1)
                sc = _rollout(api, student, x, len(student.blocks) - 1 if start is None else start)
                with torch.no_grad():
                    m_lp = api.teacher_forward_keepmask(T, x, _topk(sc, K)).float().log_softmax(-1)
                sig.append(_kl(t_lp, m_lp))
                out.append(_kl(t_lp, s_lp))
                cor.append((s_lp.argmax(-1) == y).float())
            cols[m] = {k: torch.cat(v).cpu().numpy().astype(np.float64)
                       for k, v in (("sig", sig), ("out", out), ("cor", cor))}
            summary["students"][f"{m}_s{s}"] = {
                "run": rid, "val_acc_last": acc_last, "val_acc_here": round(100 * float(cols[m]["cor"].mean()), 3),
                "signal_kl": round(float(cols[m]["sig"].mean()), 5), "outcome_kl": round(float(cols[m]["out"].mean()), 5)}
            del student
            torch.cuda.empty_cache()
        L, M = cols["lateroll"], cols["maskedkd"]
        g = M["sig"] - L["sig"]
        o = M["out"] - L["out"]
        c = L["cor"] - M["cor"]
        ctrl = [t_ent, 0.5 * (M["sig"] + L["sig"]), 0.5 * (M["out"] + L["out"])]
        per[s] = (g, o, ctrl)
        q = np.quantile(g, [1 / 3, 2 / 3])
        lo_t, hi_t = g <= q[0], g >= q[1]
        summary[f"seed{s}"] = {
            "g_mean": round(float(g.mean()), 5), "o_mean": round(float(o.mean()), 5),
            "spearman_g_o": _boot(_spearman, [g, o], idx),
            "partial_g_o_given_difficulty": _boot(lambda a, b, c0, c1, c2: _partial(a, b, [c0, c1, c2]),
                                                  [g, o] + ctrl, idx),
            "spearman_g_correct_diff": round(_spearman(g, c), 4),
            "o_mean_top_third_g": round(float(o[hi_t].mean()), 5),
            "o_mean_bottom_third_g": round(float(o[lo_t].mean()), 5),
            "acc_diff_top_third_g": round(100 * float(c[hi_t].mean()), 3),
            "acc_diff_bottom_third_g": round(100 * float(c[lo_t].mean()), 3),
        }
    cross = {}
    for a in per:
        for b in per:
            if a != b:
                cross[f"g{a}_o{b}"] = round(_spearman(per[a][0], per[b][1]), 4)
                cross[f"partial_g{a}_o{b}"] = round(_partial(per[a][0], per[b][1], per[b][2]), 4)
    summary["cross_seed_report_only"] = cross
    api.write_json("summary", summary)
