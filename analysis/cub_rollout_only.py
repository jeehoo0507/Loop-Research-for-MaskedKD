"""H-C19: are the tokens that rollout picks but lateroll does not (R minus L) worse than random tokens for the
keep-only masked teacher? Opposite direction of H-C18's swapR_rand (analysis_cub_lateroll_decomp_1101_070348_b793).

Same students as H-C18: lateroll's own screen students (K=29 and K=49, seeds 0, 4, 5, final weights), 600 clean val
images, eval mode. Per student, at its training K:
  L, R            lateroll (rollout over blocks 6..11) and rollout (blocks 0..11) top-K
  swapRL_rand     R with its tokens not in L replaced by random tokens from outside R | L   (primary)
  swapLR_rand     L with its tokens not in R replaced by random tokens from outside R | L   (control, descriptive)
  early rank      mean rank (0 = top) of R-minus-L tokens under the early rollout (blocks 0..5) and the late one
Paired image-bootstrap 95% CI (2000 draws), numpy default_rng 20261101 for every random draw.
"""
import numpy
import torch

META = {
    "question": "on CUB, are the tokens chosen by full-depth rollout but not by lateroll (blocks 6..11) worse than "
                "random unselected tokens for the keep-only masked teacher's soft label?",
    "method": "attention rollout (blocks 0..11, 0..5, 6..11), token swap test, keep-only masked teacher faithfulness "
              "(DKD TCKD, KL); image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Zhao et al. CVPR 2022 (DKD); DeYoung et al. ACL 2020 (ERASER "
             "sufficiency, keep-only)",
}

SPLIT = 6
EPS = 1e-8
NBOOT = 2000
BATCH = 50
CONDS = ("L", "R", "swapRL_rand", "swapLR_rand")


def _topk(s, k):
    idx = s.topk(k, dim=1).indices
    return torch.zeros_like(s, dtype=torch.bool).scatter_(1, idx, True)


def _rollout(attn, start, stop=None):
    r = None
    for a in attn[start:stop]:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _swap(base, other, fill_scores, excl):
    """Replace base's tokens not in `other` by the same number of tokens outside `excl`, ordered by fill_scores."""
    drop = base & ~other
    m = drop.sum(1)
    s = fill_scores.clone()
    s[excl] = -1e9
    order = s.argsort(1, descending=True)
    ranks = torch.empty_like(order).scatter_(1, order, torch.arange(s.shape[1], device=s.device).expand_as(order).contiguous())
    add = ranks < m[:, None]
    return (base & ~drop) | add


def _ranks(s):
    order = s.argsort(1, descending=True)
    return torch.empty_like(order).scatter_(1, order, torch.arange(s.shape[1], device=s.device).expand_as(order).contiguous())


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
            students.append((f"k{k}_s{r['seed']}_final", r_id, k))
    api.write_json("students", students)
    T = api.teacher.eval()
    gen = numpy.random.default_rng(20261101)

    fulls = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            fulls.append(T(x).float().log_softmax(-1))

    summary = {}
    for name, rid, K in students:
        S = api.student(rid)
        g = numpy.random.default_rng(20261101)
        acc = {c: {"kl": [], "tckd": []} for c in CONDS}
        n_rl, rk_early, rk_late = [], [], []
        for b, (x, y) in enumerate(api.val_batches(batch=BATCH)):
            B, n = x.shape[0], api.num_patches
            with torch.no_grad():
                out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
            attn = [a.detach() for a in out.attn]
            late = _rollout(attn, SPLIT)
            early = _rollout(attn, 0, SPLIT)
            L = _topk(late, K)
            R = _topk(_rollout(attn, 0), K)
            excl = R | L
            masks = {"L": L, "R": R,
                     "swapRL_rand": _swap(R, L, torch.from_numpy(g.random((B, n))).float().to(x.device), excl),
                     "swapLR_rand": _swap(L, R, torch.from_numpy(g.random((B, n))).float().to(x.device), excl)}
            for c, mk in masks.items():
                assert int(mk.sum(1).min()) == K and int(mk.sum(1).max()) == K
                with torch.no_grad():
                    logits = api.teacher_forward_keepmask(T, x, mk)
                kl, tc = _kl_tckd(fulls[b], logits, y)
                acc[c]["kl"].append(kl.cpu()); acc[c]["tckd"].append(tc.cpu())
            rl = R & ~L
            cnt = rl.sum(1).float()
            n_rl.append(cnt.cpu())
            ok = cnt > 0
            re, rla = _ranks(early).float(), _ranks(late).float()
            rk_early.append(((re * rl).sum(1)[ok] / cnt[ok]).cpu())
            rk_late.append(((rla * rl).sum(1)[ok] / cnt[ok]).cpu())
        v = {c: {m: torch.cat(a).numpy().astype(numpy.float64) for m, a in acc[c].items()} for c in CONDS}
        res = {"run": rid, "K": K,
               "mean": {c: {m: round(float(a.mean()), 4) for m, a in v[c].items()} for c in CONDS},
               "n_R_minus_L": round(float(torch.cat(n_rl).mean()), 3),
               "R_minus_L_rank_early": round(float(torch.cat(rk_early).mean()), 2),
               "R_minus_L_rank_late": round(float(torch.cat(rk_late).mean()), 2),
               "d_tckd_swapRL_rand_minus_R": _ci(v["swapRL_rand"]["tckd"] - v["R"]["tckd"], gen),
               "d_kl_swapRL_rand_minus_R": _ci(v["swapRL_rand"]["kl"] - v["R"]["kl"], gen),
               "d_tckd_swapLR_rand_minus_L": _ci(v["swapLR_rand"]["tckd"] - v["L"]["tckd"], gen),
               "d_kl_swapLR_rand_minus_L": _ci(v["swapLR_rand"]["kl"] - v["L"]["kl"], gen),
               "d_tckd_R_minus_L": _ci(v["R"]["tckd"] - v["L"]["tckd"], gen)}
        summary[name] = res
        api.write_json(f"swap_{name}", res)

    k29 = [s for s in summary if s.startswith("k29_")]
    verdict = {
        "P1_swapRL_rand_minus_R_tckd_ci_hi_lt_0_k29_all":
            all(summary[s]["d_tckd_swapRL_rand_minus_R"][2] < 0 for s in k29),
        "P2_rank_early_lt_rank_late_k29_all":
            all(summary[s]["R_minus_L_rank_early"] < summary[s]["R_minus_L_rank_late"] for s in k29),
        "n_k29_students": len(k29),
    }
    api.write_json("summary", {"students": students, "swap": summary, "verdict": verdict})
