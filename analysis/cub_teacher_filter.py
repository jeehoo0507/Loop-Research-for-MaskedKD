"""H-C14 (QUEUE 2026-10-31 20:40, TrustMask): does the cached teacher map help as a *filter* (allowed region in which
the student still picks) rather than as a *picker* (teacher top-K), and only at small K / early in training?

Students: CUB final full-KD s1 at epoch 25, 50 and final (a selector runs on a student that is still training).
Teacher map: the teacher's attention rollout of the whole val image (same rule as harness/teacher_cache.py and the
teacher_topk baseline: head mean, 0.5 A + 0.5 I, rows renormalised, product over all blocks, CLS row).
Student scores: roll0 = student rollout over blocks 0..11 (harness `rollout`), roll6 = blocks 6..11 (`lateroll`).
Conditions per K in (98, 49, 29, 12):
  s_roll0, s_roll6        student top-K
  t_pick                  teacher-map top-K (= teacher_topk)
  f_roll0, f_roll6        allowed = teacher top-ceil(1.5 K) (<= 196); student top-K inside it
  fa_roll6                per-image count k_i: the batch's B*K highest (image, token) teacher-map values counted per
                          image (sum-normalised maps, so a concentrated teacher map gets fewer); allowed = teacher
                          top-ceil(1.5 k_i); roll6 top-k_i inside it; batch mean = K exactly
  random                  fixed draw (numpy default_rng on the analysis side only, never a selector)
Masked-teacher KL(full || masked) and DKD TCKD per image; paired diffs with image bootstrap 95% CI (2000 draws,
numpy default_rng 20261031); top-K overlap |student & teacher| / K.
No position prior: every score is a per-image map.
"""
import math

import numpy
import torch

META = {
    "question": "on CUB, is the cached teacher rollout map better used to pick (teacher top-K) or to filter "
                "(student picks inside the teacher's allowed region), and does a filter gain appear only at small K "
                "or early in training?",
    "method": "attention rollout (student and teacher); keep-only masked teacher faithfulness (KL, DKD TCKD); "
              "image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Zhao et al. CVPR 2022 (DKD); Son et al. ECCV 2024 (MaskedKD)",
}

KS = (98, 49, 29, 12)
SPLIT = 6
EPS = 1e-8
NBOOT = 2000
BATCH = 50
CONDS = ("s_roll0", "s_roll6", "t_pick", "f_roll0", "f_roll6", "fa_roll6", "random")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _rollouts(api, M, x):
    depth = len(M.blocks)
    with torch.no_grad():
        out = api.student_forward(M, x, attn_layers=list(range(depth)))
        A = [a.float().mean(1) for a in out.attn]
        B, N, _ = A[0].shape
        eye = torch.eye(N, device=A[0].device).expand(B, N, N)
        R, r6 = None, None
        for s in range(depth - 1, -1, -1):
            a = 0.5 * A[s] + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else R @ a
            if s == SPLIT:
                r6 = R[:, 0, 1:].clone()
    r0 = R[:, 0, 1:].float()
    return r0 / r0.sum(1, keepdim=True), r6.float() / r6.float().sum(1, keepdim=True)


def _filter(student, teacher, k):
    """student top-k inside teacher top-ceil(1.5 k); k: int or long [B]."""
    B, n = student.shape
    k = torch.as_tensor(k, device=student.device).expand(B).long()
    m = torch.clamp(torch.ceil(1.5 * k.float()).long(), max=n)
    rank_t = teacher.argsort(1, descending=True).argsort(1)
    allowed = rank_t < m[:, None]
    s = student.masked_fill(~allowed, -1.0)
    rank_s = s.argsort(1, descending=True).argsort(1)
    return rank_s < k[:, None]


def _adaptive_counts(teacher, k):
    B, n = teacher.shape
    flat = teacher.reshape(-1)
    idx = flat.topk(B * k).indices
    cnt = torch.zeros(B * n, device=teacher.device)
    cnt[idx] = 1
    return cnt.view(B, n).sum(1).long().clamp(min=1)


def _stats(full_lp, mask_lp, y):
    pf, pm = full_lp.exp(), mask_lp.exp()
    kl = (pf * (full_lp - mask_lp)).sum(-1)
    pt_f = pf.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    pt_m = pm.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()
    return kl, tckd


def _ci(v, gen):
    n = len(v)
    bs = numpy.array([v[gen.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return [round(float(v.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4)]


def run(api):
    runs = api.runs()
    rid = None
    for r_id, r in sorted(runs.items()):
        if r.get("dataset") == "cub" and r["tier"] == "final" and str(r["method"]).split("@")[0] == "full" \
                and r["seed"] == 1:
            rid = r_id
    snaps = [e for e in (25, 50) if e in (runs[rid].get("snapshots") or [])]
    api.write_json("students", {"run": rid, "epochs": snaps + ["final"]})
    T = api.teacher.eval()
    n = api.num_patches
    gen = numpy.random.default_rng(20261031)

    # teacher maps and full teacher log-probs, once
    tmaps, fulls, ys, xs_count = [], [], [], 0
    for x, y in api.val_batches(batch=BATCH):
        tr0, _ = _rollouts(api, T, x)
        tmaps.append(tr0)
        with torch.no_grad():
            fulls.append(T(x).float().log_softmax(-1))
        ys.append(y)
        xs_count += x.shape[0]

    summary, verdict_src = {}, {}
    for ep in snaps + ["final"]:
        S = api.student(rid, epoch=None if ep == "final" else ep)
        g = numpy.random.default_rng(20261031)
        per = {k: {c: {"kl": [], "tckd": []} for c in CONDS} for k in KS}
        ov = {k: {"roll0": [], "roll6": []} for k in KS}
        kstat = {k: [] for k in KS}
        for b, (x, y) in enumerate(api.val_batches(batch=BATCH)):
            B = x.shape[0]
            r0, r6 = _rollouts(api, S, x)
            tm, full = tmaps[b], fulls[b]
            rnd = torch.from_numpy(g.random((B, n))).float().to(x.device)
            for k in KS:
                tk = _topk(tm, k)
                ka = _adaptive_counts(tm, k)
                masks = {"s_roll0": _topk(r0, k), "s_roll6": _topk(r6, k), "t_pick": tk,
                         "f_roll0": _filter(r0, tm, k), "f_roll6": _filter(r6, tm, k),
                         "fa_roll6": _filter(r6, tm, ka), "random": _topk(rnd, k)}
                for c, keep in masks.items():
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    kl, tckd = _stats(full, lm, y)
                    per[k][c]["kl"].append(kl.cpu())
                    per[k][c]["tckd"].append(tckd.cpu())
                ov[k]["roll0"].append(((masks["s_roll0"] & tk).sum(1).float() / k).cpu())
                ov[k]["roll6"].append(((masks["s_roll6"] & tk).sum(1).float() / k).cpu())
                kstat[k].append(ka.float().cpu())
        res = {}
        for k in KS:
            t = {c: torch.cat(per[k][c]["tckd"]).numpy().astype(numpy.float64) for c in CONDS}
            kl = {c: torch.cat(per[k][c]["kl"]).numpy().astype(numpy.float64) for c in CONDS}
            ka = torch.cat(kstat[k]).numpy()
            res[str(k)] = {
                "tckd": {c: round(float(v.mean()), 4) for c, v in t.items()},
                "kl": {c: round(float(v.mean()), 4) for c, v in kl.items()},
                "d_tckd_f_roll6_minus_s_roll6": _ci(t["f_roll6"] - t["s_roll6"], gen),
                "d_tckd_f_roll0_minus_s_roll0": _ci(t["f_roll0"] - t["s_roll0"], gen),
                "d_tckd_f_roll6_minus_t_pick": _ci(t["f_roll6"] - t["t_pick"], gen),
                "d_tckd_s_roll6_minus_t_pick": _ci(t["s_roll6"] - t["t_pick"], gen),
                "d_tckd_fa_roll6_minus_f_roll6": _ci(t["fa_roll6"] - t["f_roll6"], gen),
                "overlap_student_teacher": {s: round(float(torch.cat(v).mean()), 4) for s, v in ov[k].items()},
                "adaptive_k": {"mean": round(float(ka.mean()), 2), "p10": float(numpy.quantile(ka, 0.1)),
                               "p90": float(numpy.quantile(ka, 0.9))},
            }
        summary[str(ep)] = res
        api.write_json(f"fidelity_e{ep}", res)

    fin = summary["final"]
    verdict = {
        "P1_f_roll6_beats_s_roll6_ci_hi_lt_0_at_12_and_29":
            all(fin[str(k)]["d_tckd_f_roll6_minus_s_roll6"][2] < 0 for k in (12, 29)),
        "P1_gain_at_12_larger_than_at_98":
            fin["12"]["d_tckd_f_roll6_minus_s_roll6"][0] < fin["98"]["d_tckd_f_roll6_minus_s_roll6"][0],
        "P2_e25_gain_larger_than_final_at_29": ("25" in summary) and
            summary["25"]["29"]["d_tckd_f_roll6_minus_s_roll6"][0] < fin["29"]["d_tckd_f_roll6_minus_s_roll6"][0],
        "P3_fa_roll6_beats_f_roll6_ci_hi_lt_0_at_29":
            fin["29"]["d_tckd_fa_roll6_minus_f_roll6"][2] < 0,
        "D_t_pick_best_tckd_at_each_k":
            {str(k): min(fin[str(k)]["tckd"], key=fin[str(k)]["tckd"].get) for k in KS},
    }
    api.write_json("summary", {"students": {"run": rid, "epochs": snaps + ["final"]}, "n_images": xs_count,
                               "fidelity": summary, "verdict": verdict})
