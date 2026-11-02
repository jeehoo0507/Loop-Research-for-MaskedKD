"""H-A18: two checks on the H-A17 image-level link between masked-teacher signal and student outcome, CUB final
seeds 1, 2, 3 at K=29, each student scoring its own mask:
  (A) rollout comparator: g = sig(rollout@29 student, full rollout mask) - sig(lateroll@29 student, roll6 mask),
      o = out(rollout student) - out(lateroll student), same seed. Does the H-A17 link hold against rollout too?
  (N) null control: two students of the SAME method (maskedkd or lateroll) from different seeds, pairs (1,2), (1,3),
      (2,3): g = sig(a) - sig(b), o = out(a) - out(b). If this also correlates, the H-A17 link is a generic
      student-level coupling (a student whose own mask keeps the teacher better on an image is also closer to the
      teacher there), not something that needs the lateroll-vs-maskedkd method difference.
sig = KL(T || T masked by the student's own top-K mask), out = KL(T || student). Spearman and partial Spearman
(teacher entropy, mean signal KL, mean outcome KL) with paired bootstrap over val images. Descriptive only: nothing
is selected or tuned, lateroll is not changed (QUEUE 2026-11-02 08:50). Helpers copied from
analysis/signal_outcome_link.py (analyses may not import each other). Bootstrap uses numpy default_rng.
"""
import numpy as np
import torch

META = {
    "question": "on CUB final K=29 students of seeds 1, 2, 3, does the H-A17 per-image signal/outcome link also hold "
                "for lateroll vs rollout, and does a same-method different-seed null pair show the same link?",
    "method": "keep-only masked-teacher faithfulness (KL) linked to student-teacher output divergence per image; "
              "Spearman and rank partial correlation with paired bootstrap over val images; same-method null pairs",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Stanton et al. NeurIPS 2021 "
             "(student-teacher fidelity); Hooker et al. NeurIPS 2019 (keep-only evaluation)",
}

SEEDS = (1, 2, 3)
K = 29
STARTS = {"lateroll": 6, "maskedkd": 11, "rollout": 0}
PAIRS = ((1, 2), (1, 3), (2, 3))
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
        for s in range(depth - 1, start - 1, -1):    # start=0 is the harness rollout baseline (A11 ... A0)
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


def _link(A, B, t_ent, idx):
    """g, o = B minus A (A is the side expected to be better); Spearman and difficulty-controlled partial."""
    g = B["sig"] - A["sig"]
    o = B["out"] - A["out"]
    ctrl = [t_ent, 0.5 * (A["sig"] + B["sig"]), 0.5 * (A["out"] + B["out"])]
    return {
        "g_mean": round(float(g.mean()), 5), "o_mean": round(float(o.mean()), 5),
        "spearman_g_o": _boot(_spearman, [g, o], idx),
        "partial_g_o_given_difficulty": _boot(lambda a, b, c0, c1, c2: _partial(a, b, [c0, c1, c2]),
                                              [g, o] + ctrl, idx),
    }


def run(api):
    ds = api.dataset
    found = {}
    for rid, r in sorted(api.runs().items()):
        if r.get("dataset") != ds or r["tier"] != "final" or r["seed"] not in SEEDS:
            continue
        if r["method"] in STARTS and int(r.get("keep") or 0) == K:
            found.setdefault((r["method"], r["seed"]), (rid, r.get("val_acc_last")))
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
    cols = {}
    for (m, s), (rid, acc_last) in sorted(found.items()):
        student = api.student(rid)
        sig, out, cor = [], [], []
        for x, y, t_lp in batches:
            with torch.no_grad():
                s_lp = api.student_forward(student, x).logits.float().log_softmax(-1)
            sc = _rollout(api, student, x, STARTS[m])
            with torch.no_grad():
                m_lp = api.teacher_forward_keepmask(T, x, _topk(sc, K)).float().log_softmax(-1)
            sig.append(_kl(t_lp, m_lp))
            out.append(_kl(t_lp, s_lp))
            cor.append((s_lp.argmax(-1) == y).float())
        cols[(m, s)] = {k: torch.cat(v).cpu().numpy().astype(np.float64)
                        for k, v in (("sig", sig), ("out", out), ("cor", cor))}
        summary["students"][f"{m}_s{s}"] = {
            "run": rid, "val_acc_last": acc_last, "val_acc_here": round(100 * float(cols[(m, s)]["cor"].mean()), 3),
            "signal_kl": round(float(cols[(m, s)]["sig"].mean()), 5),
            "outcome_kl": round(float(cols[(m, s)]["out"].mean()), 5)}
        del student
        torch.cuda.empty_cache()

    summary["A_rollout_vs_lateroll"] = {
        f"seed{s}": _link(cols[("lateroll", s)], cols[("rollout", s)], t_ent, idx)
        for s in SEEDS if ("lateroll", s) in cols and ("rollout", s) in cols}
    summary["N_same_method_pairs"] = {
        f"{m}_s{a}_vs_s{b}": _link(cols[(m, a)], cols[(m, b)], t_ent, idx)
        for m in ("maskedkd", "lateroll") for a, b in PAIRS if (m, a) in cols and (m, b) in cols}
    summary["H_A17_repro_maskedkd_vs_lateroll"] = {
        f"seed{s}": _link(cols[("lateroll", s)], cols[("maskedkd", s)], t_ent, idx)
        for s in SEEDS if ("lateroll", s) in cols and ("maskedkd", s) in cols}
    api.write_json("summary", summary)
