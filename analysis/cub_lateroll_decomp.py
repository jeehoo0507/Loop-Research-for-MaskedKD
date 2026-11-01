"""H-C18: decomposition of lateroll on CUB (AGENT_PROMPT "승격 전 분해", AUDIT_PROMPT 15), measured on lateroll's own
trained screen students (K=29 and K=49, seeds 0, 4, 5, final weights; K=29 seed 0 also at epochs 25 and 50).

Per student, at the K it was trained with, on the 600 clean val images (eval mode):
  overlap   |L & M| / K, |L & R| / K, |L & C| / K   (L lateroll, M maskedkd, R rollout, C center@K)
  spatial   mean distance (patch units) of the kept tokens from the image centre, for L, R, M
  fidelity  keep-only masked-teacher TCKD / KL vs the full teacher for
            L, R, M, C, G (grid), U (uniform random),
            swapM_rand  L with its tokens not in M replaced by random tokens from outside L
            swapM_grid  L with its tokens not in M replaced by grid-order tokens from outside L
            swapR_rand  L with its tokens not in R replaced by random tokens from outside L
Paired diffs vs L with image bootstrap 95% CI (2000 draws). All randomness: numpy default_rng on the analysis side.
Scores follow harness/baselines.py (MaskedKD, Rollout, Grid, Center) and selectors/lateroll.py (blocks 6..11).
"""
import math

import numpy
import torch

META = {
    "question": "on CUB, where does lateroll's masked-teacher fidelity come from: how much do its tokens overlap "
                "MaskedKD's and rollout's, do its non-overlapping tokens matter (swap with random/grid), and is it "
                "more centred than rollout?",
    "method": "attention rollout (blocks 6..11 vs 0..11), MaskedKD last-block CLS attention; token overlap, swap "
              "test, spatial spread; keep-only masked teacher faithfulness (DKD TCKD, KL); image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Zhao et al. CVPR 2022 (DKD); "
             "DeYoung et al. ACL 2020 (ERASER sufficiency, keep-only)",
}

SPLIT = 6
EPS = 1e-8
NBOOT = 2000
BATCH = 50
CONDS = ("L", "R", "M", "C", "G", "U", "swapM_rand", "swapM_grid", "swapR_rand")


def _topk(s, k):
    idx = s.topk(k, dim=1).indices
    return torch.zeros_like(s, dtype=torch.bool).scatter_(1, idx, True)


def _rollout(attn, start):
    r = None
    for a in attn[start:]:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _fixed_masks(n, k, B, device):
    side = int(round(n ** 0.5))
    r = torch.arange(n, device=device)
    c = (side - 1) / 2
    dist = ((r // side - c) ** 2 + (r % side - c) ** 2).float() + r.float() * 1e-6
    C = _topk((-dist).expand(B, n), k)
    gidx = torch.linspace(0, n - 1, k, device=device).round().long()
    G = torch.zeros(B, n, dtype=torch.bool, device=device).index_fill_(1, gidx, True)
    return C, G, dist.sqrt()


def _swap(L, other, fill_scores):
    """Replace L's tokens that are not in `other` by the same number of tokens from outside L, by fill_scores."""
    drop = L & ~other
    m = drop.sum(1)
    s = fill_scores.clone()
    s[L] = -1e9
    order = s.argsort(1, descending=True)
    ranks = torch.empty_like(order).scatter_(1, order, torch.arange(s.shape[1], device=s.device).expand_as(order).contiguous())
    add = ranks < m[:, None]
    return (L & ~drop) | add


def _kl_tckd(full_lp, logits, y):
    pf = full_lp.exp()
    lp = logits.float().log_softmax(-1)
    kl = (pf * (full_lp - lp)).sum(-1)
    pm = lp.exp()
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
    students = []
    for r_id, r in sorted(runs.items()):
        if r.get("dataset") == "cub" and r["tier"] == "screen" and str(r["method"]).split("@")[0] == "lateroll" \
                and int(float(r["keep"])) in (29, 49) and r["seed"] in (0, 4, 5):
            k = int(float(r["keep"]))
            students.append((f"k{k}_s{r['seed']}_final", r_id, None, k))
            if k == 29 and r["seed"] == 0:
                for e in (25, 50):
                    if e in (r.get("snapshots") or []):
                        students.append((f"k{k}_s0_e{e}", r_id, e, k))
    api.write_json("students", students)
    T = api.teacher.eval()
    n = api.num_patches
    gen = numpy.random.default_rng(20261101)

    fulls = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            fulls.append(T(x).float().log_softmax(-1))

    summary = {}
    for name, rid, ep, K in students:
        S = api.student(rid, epoch=ep)
        g = numpy.random.default_rng(20261101)
        acc = {c: {"kl": [], "tckd": []} for c in CONDS}
        ov = {"LM": [], "LR": [], "LC": []}
        dist = {"L": [], "R": [], "M": []}
        for b, (x, y) in enumerate(api.val_batches(batch=BATCH)):
            B = x.shape[0]
            with torch.no_grad():
                out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
            attn = [a.detach() for a in out.attn]
            L = _topk(_rollout(attn, SPLIT), K)
            R = _topk(_rollout(attn, 0), K)
            M = _topk(attn[-1].float().mean(1)[:, 0, 1:], K)
            C, G, d = _fixed_masks(n, K, B, x.device)
            U = _topk(torch.from_numpy(g.random((B, n))).float().to(x.device), K)
            # grid tokens first, then the rest in raster order
            gfill = G.float() * 2 + torch.linspace(1, 0, n, device=x.device)[None, :]
            masks = {"L": L, "R": R, "M": M, "C": C, "G": G, "U": U,
                     "swapM_rand": _swap(L, M, torch.from_numpy(g.random((B, n))).float().to(x.device)),
                     "swapM_grid": _swap(L, M, gfill),
                     "swapR_rand": _swap(L, R, torch.from_numpy(g.random((B, n))).float().to(x.device))}
            for c, mk in masks.items():
                assert int(mk.sum(1).min()) == K and int(mk.sum(1).max()) == K
                with torch.no_grad():
                    logits = api.teacher_forward_keepmask(T, x, mk)
                kl, tc = _kl_tckd(fulls[b], logits, y)
                acc[c]["kl"].append(kl.cpu()); acc[c]["tckd"].append(tc.cpu())
            ov["LM"].append(((L & M).sum(1).float() / K).cpu())
            ov["LR"].append(((L & R).sum(1).float() / K).cpu())
            ov["LC"].append(((L & C).sum(1).float() / K).cpu())
            for key, mk in (("L", L), ("R", R), ("M", M)):
                dist[key].append(((mk.float() * d[None, :]).sum(1) / K).cpu())
        v = {c: {m: torch.cat(a).numpy().astype(numpy.float64) for m, a in acc[c].items()} for c in CONDS}
        o = {kk: torch.cat(a).numpy().astype(numpy.float64) for kk, a in ov.items()}
        dd = {kk: torch.cat(a).numpy().astype(numpy.float64) for kk, a in dist.items()}
        res = {"run": rid, "epoch": ep or "final", "K": K,
               "mean": {c: {m: round(float(a.mean()), 4) for m, a in v[c].items()} for c in CONDS},
               "overlap": {kk: round(float(a.mean()), 4) for kk, a in o.items()},
               "centre_dist": {kk: round(float(a.mean()), 4) for kk, a in dd.items()},
               "d_centre_dist_L_minus_R": _ci(dd["L"] - dd["R"], gen)}
        for c in CONDS[1:]:
            res[f"d_tckd_{c}_minus_L"] = _ci(v[c]["tckd"] - v["L"]["tckd"], gen)
            res[f"d_kl_{c}_minus_L"] = _ci(v[c]["kl"] - v["L"]["kl"], gen)
        summary[name] = res
        api.write_json(f"decomp_{name}", res)

    k29 = [s for s in summary if s.startswith("k29_") and s.endswith("_final")]
    verdict = {
        "P1_overlap_LM_le_0.6_k29_all": all(summary[s]["overlap"]["LM"] <= 0.6 for s in k29),
        "P2_swapM_rand_tckd_ci_lo_gt_0_k29_all": all(summary[s]["d_tckd_swapM_rand_minus_L"][1] > 0 for s in k29),
        "P3_swapR_rand_and_R_tckd_ci_lo_gt_0_k29_all":
            all(summary[s]["d_tckd_swapR_rand_minus_L"][1] > 0 and summary[s]["d_tckd_R_minus_L"][1] > 0
                for s in k29),
        "P4_centre_dist_L_ge_R_k29_all": all(summary[s]["d_centre_dist_L_minus_R"][0] >= 0 for s in k29),
        "n_k29_students": len(k29),
    }
    api.write_json("summary", {"students": students, "decomp": summary, "verdict": verdict})
