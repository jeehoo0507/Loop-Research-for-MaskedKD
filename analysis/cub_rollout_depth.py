"""H-C2b: on CUB, which blocks should attention rollout use to keep the masked teacher's soft label, does the answer
hold for students seen mid-training, and is the gain explained by a centre bias?

Basis: analysis_cub_tckd_scores_1031_095253_a64e (H-C2a). Rollout over blocks 6..11 only (roll6) kept TCKD / KL
lower than full rollout (blocks 0..11) at K=49, 25, 12 for all three CUB final students (e.g. K=12 TCKD 1.67 vs
2.06 on full196_s1). That was a side result (not the preregistered score), so it is checked here before any use:
  1. start-block sweep s in (0, 3, 6, 8, 9, 10, 11): rollout_s = rollout over blocks s..11 of the student's own
     attention (s=0 is the harness rollout; s=11 gives the same ranking as maskedkd, a sanity check).
  2. the same on the epoch-25 / epoch-50 snapshots, because a selector is used on a student that is still training.
  3. position check (validity rule: no position prior): for each score the mean normalised distance of kept tokens
     from the image centre and the share of kept tokens inside centre@K; centre@K and random@K fidelity as refs.
No labels, no teacher, no position enter any score. Cost difference between start blocks is at most 12 products
of [197 x 197] matrices per image (< 0.2 GFLOPs), small next to the 7.64 GFLOPs student step.
"""
import torch

META = {
    "question": "on CUB, does rollout over late student blocks only keep the masked teacher's TCKD/KL better than "
                "full rollout at the same K, on final and mid-training students, without a centre bias?",
    "method": "keep-only masked-teacher faithfulness (KL to full teacher, DKD TCKD/NCKD split, p_true, top-1 "
              "agreement) for rollout start-block sweep, plus centre-distance / centre-overlap of kept tokens",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Chefer et al. CVPR 2021 (late-layer relevance); "
             "Zhao et al. CVPR 2022 (DKD); Son et al. ECCV 2024 (MaskedKD)",
}

KS = (98, 49, 25, 12)
STARTS = (0, 3, 6, 8, 9, 10, 11)
SIDE = 14
EPS = 1e-8


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _dist(n, dev):
    r = torch.arange(n, device=dev)
    c = (SIDE - 1) / 2
    return ((r // SIDE - c) ** 2 + (r % SIDE - c) ** 2).float().sqrt() / (c * 2 ** 0.5)   # 0 centre .. 1 corner


def _center(B, n, k, dev):
    return _topk((-(_dist(n, dev) + torch.arange(n, device=dev).float() * 1e-6)).expand(B, n), k)


def _random(B, n, k, dev, off):
    i = torch.arange(off, off + B, device=dev, dtype=torch.float64)[:, None]
    j = torch.arange(n, device=dev, dtype=torch.float64)[None, :]
    u = torch.frac(torch.sin(i * 12.9898 + j * 78.233) * 43758.5453)
    return _topk(u.float(), k)


def _rollouts(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad():
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
        A = [a.float().mean(1) for a in out.attn]
        B, N, _ = A[0].shape
        eye = torch.eye(N, device=A[0].device).expand(B, N, N)
        res, R = {}, None
        for s in range(depth - 1, -1, -1):           # R_s = A_11 ... A_s, built from the last block backwards
            a = 0.5 * A[s] + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else R @ a
            if s in STARTS:
                res[f"roll{s}"] = R[:, 0, 1:].clone()
    return res


def _stats(full_lp, mask_lp, y):
    pf, pm = full_lp.exp(), mask_lp.exp()
    kl = (pf * (full_lp - mask_lp)).sum(-1)
    pt_f = pf.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    pt_m = pm.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()
    return {"kl": kl, "tckd": tckd, "nckd": kl - tckd, "p_true": pt_m,
            "top1_agree": (pf.argmax(-1) == pm.argmax(-1)).float()}


def run(api):
    runs = api.runs()
    studs = {}
    for rid, r in sorted(runs.items()):
        if r.get("dataset") != "cub" or r["seed"] != 1 or r["tier"] != "final":
            continue
        name = f"{r['method']}{int(r.get('keep') or 0)}_s1"
        studs[f"{name}_final"] = (rid, None)
        for e in r.get("snapshots") or []:
            studs[f"{name}_e{e}"] = (rid, e)
    api.write_json("students", {k: [v[0], v[1]] for k, v in studs.items()})

    T = api.teacher.eval()
    n = api.num_patches
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        batches.append((x, y, full))
    N = sum(b[0].shape[0] for b in batches)
    names = [f"roll{s}" for s in STARTS]

    # fixed references (student-independent)
    ref = {}
    for k in KS:
        for m in ("center", "random"):
            acc, off = {}, 0
            for x, y, full in batches:
                B = x.shape[0]
                keep = _center(B, n, k, x.device) if m == "center" else _random(B, n, k, x.device, off)
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                for s, v in _stats(full, lm, y).items():
                    acc[s] = acc.get(s, 0.0) + v.sum().item()
                off += B
            ref[f"{m}@{k}"] = {s: round(v / N, 4) for s, v in acc.items()}
    api.write_json("reference", ref)

    results, verdict = {}, {}
    for sname, (rid, ep) in studs.items():
        student = api.student(rid, ep)
        acc = {(m, k): {} for m in names for k in KS}
        pos = {(m, k): [0.0, 0.0] for m in names for k in KS}      # mean centre distance, overlap with centre@K
        for x, y, full in batches:
            B = x.shape[0]
            sc = _rollouts(api, student, x)
            d = _dist(n, x.device)
            for k in KS:
                cen = _center(B, n, k, x.device)
                for m in names:
                    keep = _topk(sc[m], k)
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    for s, v in _stats(full, lm, y).items():
                        acc[(m, k)][s] = acc[(m, k)].get(s, 0.0) + v.sum().item()
                    pos[(m, k)][0] += (keep.float() * d).sum().item() / k
                    pos[(m, k)][1] += (keep & cen).sum().item() / k
        res = {}
        for (m, k), dd in acc.items():
            o = {s: round(v / N, 4) for s, v in dd.items()}
            o["top1_agree"] = round(100 * o["top1_agree"], 2)
            o["centre_dist"] = round(pos[(m, k)][0] / N, 4)
            o["centre_overlap"] = round(pos[(m, k)][1] / N, 4)
            res[f"{m}@{k}"] = o
        results[sname] = res
        api.write_json(f"fidelity_{sname}", res)
        # preregistered reading (PROPOSALS H-C2b)
        verdict[sname] = {str(k): {
            "P1_tckd_roll6_lt_roll0": res[f"roll6@{k}"]["tckd"] < res[f"roll0@{k}"]["tckd"],
            "P1_kl_roll6_le_roll0": res[f"roll6@{k}"]["kl"] <= res[f"roll0@{k}"]["kl"],
            "P2_kl_roll6_lt_center": res[f"roll6@{k}"]["kl"] < ref[f"center@{k}"]["kl"],
            "centre_overlap_roll6_minus_roll0": round(res[f"roll6@{k}"]["centre_overlap"]
                                                       - res[f"roll0@{k}"]["centre_overlap"], 4),
            "best_start_by_tckd": min(names, key=lambda m: res[f"{m}@{k}"]["tckd"]),
        } for k in (25, 12)}
    api.write_json("verdict", verdict)
    api.write_json("summary", {"students": {k: list(v) for k, v in studs.items()}, "n_val": N, "ks": list(KS),
                               "starts": list(STARTS), "verdict": verdict})
