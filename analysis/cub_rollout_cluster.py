"""H-C15: why does rollout over student blocks 6..11 (lateroll) keep the masked teacher's soft label better than full
rollout (blocks 0..11)? H-C13 ruled out an image-independent fixed pattern. New hypothesis (spatial smoothing): the
early blocks mix mostly with neighbours, so multiplying them in blurs the rollout map; its top-K then forms fewer,
larger contiguous blobs and covers fewer separate object parts. Under the multi-view view of KD (Allen-Zhu & Li,
ICLR 2023) the teacher's soft label rests on several parts' evidence, so a clustered keep set loses evidence.

Common student (CUB final full-KD s1), final weights (verdict) and epoch 50 (descriptive).
Scores: roll0 = rollout blocks 0..11 (harness `rollout`), roll6 = blocks 6..11 (`lateroll`), same rule as H-C13.
Per image and K in (12, 29, 49), on the 14x14 grid (relative geometry only, no fixed location):
  comps     number of 4-connected components of the keep set
  spread    mean pairwise Euclidean distance between kept tokens (grid units)
  autocorr  per-image Pearson between a token's score and the mean score of its 4-neighbours (map smoothness)
  tmass     teacher rollout mass inside the keep set (teacher map as in H-C14 / harness teacher_cache)
Causal control (analysis side only, never a selector): roll0_nms / roll6_nms = greedy top-K with 4-neighbour
suppression (a picked token blocks its 4 neighbours; when nothing is left the suppression is lifted, picked tokens
stay excluded). If dispersing roll0 closes the roll0 -> roll6 TCKD gap, clustering is the mechanism.
Masked-teacher KL / DKD TCKD per image; paired diffs with image bootstrap 95% CI (2000 draws, numpy default_rng
20261031); Spearman(d_tckd, d_comps) with bootstrap CI.
"""
import numpy
import torch

META = {
    "question": "on CUB, does full rollout keep fewer, more clustered token blobs than late-block rollout because the "
                "early blocks smooth the map spatially, and does that clustering cost masked-teacher fidelity (TCKD)?",
    "method": "attention rollout split by block range; spatial statistics of the keep set (connected components, "
              "spread, neighbour autocorrelation); neighbour-suppressed top-K as a causal control; keep-only masked "
              "teacher faithfulness (KL, DKD TCKD); image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Zhao et al. CVPR 2022 (DKD); Allen-Zhu & Li ICLR 2023 (multi-view "
             "KD); Son et al. ECCV 2024 (MaskedKD)",
}

KS = (12, 29, 49)
SPLIT = 6
EPS = 1e-8
NBOOT = 2000
BATCH = 50
G = 14
MASKS = ("roll0", "roll6", "roll0_nms", "roll6_nms")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _neighbours(device):
    nb = torch.zeros(G * G, G * G, dtype=torch.bool, device=device)
    for r in range(G):
        for c in range(G):
            i = r * G + c
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < G and 0 <= cc < G:
                    nb[i, rr * G + cc] = True
    return nb


def _nms_topk(scores, k, nb):
    B, n = scores.shape
    keep = torch.zeros(B, n, dtype=torch.bool, device=scores.device)
    blocked = torch.zeros_like(keep)
    ar = torch.arange(B, device=scores.device)
    for _ in range(k):
        free = ~(keep | blocked)
        empty = ~free.any(1)
        blocked[empty] = False
        free = ~(keep | blocked)
        i = scores.masked_fill(~free, -1e9).argmax(1)
        keep[ar, i] = True
        blocked |= nb[i]
    return keep


def _student_maps(api, S, x):
    depth = len(S.blocks)
    with torch.no_grad():
        out = api.student_forward(S, x, attn_layers=list(range(depth)))
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
    return {"roll0": R[:, 0, 1:].float(), "roll6": r6.float()}


def _teacher_map(api, T, x):
    """teacher rollout over all blocks (same rule as H-C14 / harness teacher_cache), sum-normalised."""
    t = _student_maps(api, T, x)["roll0"]
    return t / t.sum(1, keepdim=True)


def _comps(keep):
    out = []
    for m in keep.view(-1, G, G).cpu().numpy():
        seen = numpy.zeros_like(m)
        c = 0
        for r in range(G):
            for q in range(G):
                if m[r, q] and not seen[r, q]:
                    c += 1
                    st = [(r, q)]
                    seen[r, q] = True
                    while st:
                        a, b = st.pop()
                        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            aa, bb = a + da, b + db
                            if 0 <= aa < G and 0 <= bb < G and m[aa, bb] and not seen[aa, bb]:
                                seen[aa, bb] = True
                                st.append((aa, bb))
        out.append(c)
    return torch.tensor(out, dtype=torch.float32)


def _spread(keep, coords):
    out = []
    for m in keep.cpu():
        p = coords[m]
        d = torch.cdist(p, p)
        k = p.shape[0]
        out.append(d.sum() / max(k * (k - 1), 1))
    return torch.stack(out)


def _autocorr(score, nb):
    nbf = nb.float()
    nmean = (score @ nbf.T) / nbf.sum(1)
    a = score - score.mean(1, keepdim=True)
    b = nmean - nmean.mean(1, keepdim=True)
    return (a * b).sum(1) / ((a ** 2).sum(1).sqrt() * (b ** 2).sum(1).sqrt() + EPS)


def _stats(full_lp, mask_lp, y):
    pf, pm = full_lp.exp(), mask_lp.exp()
    kl = (pf * (full_lp - mask_lp)).sum(-1)
    pt_f = pf.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    pt_m = pm.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()
    return kl, tckd


def _rank(v):
    return numpy.argsort(numpy.argsort(v, axis=-1), axis=-1).astype(numpy.float64)


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(-1, keepdims=True)
    rb -= rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / numpy.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))


def _ci(v, gen):
    n = len(v)
    bs = numpy.array([v[gen.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return [round(float(v.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4)]


def _rho_ci(a, b, gen):
    rho = float(_spearman(a[None], b[None])[0])
    idx = [gen.integers(0, len(a), len(a)) for _ in range(NBOOT)]
    bs = numpy.array([_spearman(a[None, i], b[None, i])[0] for i in idx])
    bs = bs[numpy.isfinite(bs)]
    return [round(rho, 4), round(float(numpy.quantile(bs, 0.025)), 4), round(float(numpy.quantile(bs, 0.975)), 4)]


def _measure(api, S, T, nb, coords, gen):
    per = {k: {m: {q: [] for q in ("kl", "tckd", "comps", "spread", "tmass")} for m in MASKS} for k in KS}
    ac = {"roll0": [], "roll6": []}
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        sc = _student_maps(api, S, x)
        tmap = _teacher_map(api, T, x)
        for s in ac:
            ac[s].append(_autocorr(sc[s], nb).cpu())
        for k in KS:
            masks = {"roll0": _topk(sc["roll0"], k), "roll6": _topk(sc["roll6"], k),
                     "roll0_nms": _nms_topk(sc["roll0"], k, nb), "roll6_nms": _nms_topk(sc["roll6"], k, nb)}
            for m, keep in masks.items():
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                kl, tckd = _stats(full, lm, y)
                d = per[k][m]
                d["kl"].append(kl.cpu())
                d["tckd"].append(tckd.cpu())
                d["comps"].append(_comps(keep))
                d["spread"].append(_spread(keep, coords))
                d["tmass"].append((tmap * keep.float()).sum(1).cpu())
    arr = {k: {m: {q: torch.cat(v).numpy().astype(numpy.float64) for q, v in per[k][m].items()} for m in MASKS}
           for k in KS}
    acs = {s: torch.cat(v).numpy().astype(numpy.float64) for s, v in ac.items()}
    res = {"autocorr": {s: _ci(v, gen) for s, v in acs.items()},
           "autocorr_roll0_minus_roll6": _ci(acs["roll0"] - acs["roll6"], gen)}
    for k in KS:
        a = arr[k]
        d_t = a["roll0"]["tckd"] - a["roll6"]["tckd"]
        d_c = a["roll6"]["comps"] - a["roll0"]["comps"]
        r = {q: {m: round(float(a[m][q].mean()), 4) for m in MASKS} for q in ("tckd", "kl", "comps", "spread", "tmass")}
        r["d_tckd_roll0_minus_roll6"] = _ci(d_t, gen)
        r["d_comps_roll6_minus_roll0"] = _ci(d_c, gen)
        r["d_spread_roll6_minus_roll0"] = _ci(a["roll6"]["spread"] - a["roll0"]["spread"], gen)
        r["d_tmass_roll6_minus_roll0"] = _ci(a["roll6"]["tmass"] - a["roll0"]["tmass"], gen)
        r["d_tckd_roll0nms_minus_roll0"] = _ci(a["roll0_nms"]["tckd"] - a["roll0"]["tckd"], gen)
        r["d_tckd_roll6nms_minus_roll6"] = _ci(a["roll6_nms"]["tckd"] - a["roll6"]["tckd"], gen)
        r["d_tckd_roll0nms_minus_roll6"] = _ci(a["roll0_nms"]["tckd"] - a["roll6"]["tckd"], gen)
        gap = float(d_t.mean())
        rec = float((a["roll0"]["tckd"] - a["roll0_nms"]["tckd"]).mean())
        r["nms_gap_recovery_fraction"] = round(rec / gap, 4) if abs(gap) > EPS else None
        r["spearman_dtckd_dcomps"] = _rho_ci(d_t, d_c, gen)
        r["spearman_dtckd_dtmass"] = _rho_ci(d_t, a["roll6"]["tmass"] - a["roll0"]["tmass"], gen)
        res[str(k)] = r
    return res


def run(api):
    runs = api.runs()
    rid = None
    for i, r in sorted(runs.items()):
        if r.get("dataset") == "cub" and r["tier"] == "final" and str(r["method"]).split("@")[0] == "full" \
                and r["seed"] == 1:
            rid = i
    snaps = runs[rid]["snapshots"]
    mid = 50 if 50 in snaps else (snaps[len(snaps) // 2] if snaps else None)
    api.write_json("students", {"common_full196_s1": rid, "mid_epoch": mid, "snapshots": snaps})
    T = api.teacher.eval()
    nb = _neighbours(api.device)
    coords = torch.tensor([[r, c] for r in range(G) for c in range(G)], dtype=torch.float32)
    out = {}
    for tag, ep in (("final", None), (f"e{mid}", mid)):
        if tag != "final" and mid is None:
            continue
        gen = numpy.random.default_rng(20261031)
        out[tag] = _measure(api, api.student(rid, ep), T, nb, coords, gen)
        api.write_json(f"measure_{tag}", out[tag])
    f = out["final"]
    verdict = {
        "P1_autocorr_roll0_gt_roll6_ci_lo_gt_0": f["autocorr_roll0_minus_roll6"][1] > 0,
    }
    for k in (29, 49):
        r = f[str(k)]
        verdict[f"P1_comps_roll6_gt_roll0_ci_lo_gt_0@{k}"] = r["d_comps_roll6_minus_roll0"][1] > 0
        verdict[f"P2_spearman_dtckd_dcomps_rho_ge_0.1_ci_lo_gt_0@{k}"] = bool(
            r["spearman_dtckd_dcomps"][0] >= 0.1 and r["spearman_dtckd_dcomps"][1] > 0)
        verdict[f"P3_roll0nms_better_ci_hi_lt_0_and_recovery_ge_0.5@{k}"] = bool(
            r["d_tckd_roll0nms_minus_roll0"][2] < 0 and (r["nms_gap_recovery_fraction"] or 0) >= 0.5)
    api.write_json("summary", {"students": rid, "measure": out, "verdict": verdict})
