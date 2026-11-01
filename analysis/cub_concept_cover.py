"""H-C20: does CLS-weighted concept coverage of student block-8 features keep more of the KD signal (masked-teacher
TCKD) than lateroll top-K on CUB? (Codex 11-01 IDEA 2; multi-view hypothesis, Allen-Zhu & Li ICLR 2023.)

Per student, per image (600 clean val images, eval mode), from student block-8 outputs H (CLS + 196 patches):
  X = [relu(H_p), relu(-H_p)] (196 x 2D), rank-6 NMF X ~ U V (15 multiplicative updates, fixed init),
  u_ic = U_ic / sum_j U_jc (each concept map sums to 1 over patches),
  w_c  = relu(<[relu(H_cls), relu(-H_cls)], V_c / |V_c|>), normalised to sum 1 (CLS relevance of each concept).
Conditions at the student's K (all keep exactly K patches, no position input):
  L    lateroll top-K (rollout over blocks 6..11, selectors/lateroll.py)
  CC   greedy coverage: gain_i = sum_c w_c [log(1 + cov_c + u_ic) - log(1 + cov_c)], cov_c = sum of picked u_ic
  CCu  CC with uniform w (coverage without CLS relevance; closest to Waterbirds H-W23 conceptcov)
  CW   top-K of sum_c w_c u_ic (CLS relevance without diminishing returns)
  U    uniform random
Masked-teacher KL / TCKD vs the full teacher; paired diffs with image bootstrap 95% CI (2000 draws,
numpy default_rng). Verdict on the lateroll K=29 screen students (seeds 0, 4, 5, final weights); the common
student (CUB final full-KD s1) at K = 12, 29, 49 is descriptive only.
"""
import numpy
import torch

META = {
    "question": "on CUB, does a CLS-weighted concept-coverage choice of student block-8 tokens (NMF concepts, "
                "diminishing returns per concept) keep more masked-teacher TCKD than lateroll top-K, and is any gain "
                "from coverage (vs CLS relevance alone)?",
    "method": "Deep Feature Factorization (NMF of block-8 patch features, signed split) with CLS concept weights; "
              "greedy submodular coverage; attention rollout blocks 6..11; keep-only masked teacher DKD TCKD / KL; "
              "image bootstrap",
    "paper": "Collins et al. ECCV 2018 (Deep Feature Factorization); Allen-Zhu & Li ICLR 2023 (multi-view KD); "
             "Abnar & Zuidema ACL 2020 (rollout); Zhao et al. CVPR 2022 (DKD)",
}

SPLIT = 6
BLOCK = 8
RANK = 6
ITERS = 15
EPS = 1e-8
NBOOT = 2000
BATCH = 50
CONDS = ("L", "CC", "CCu", "CW", "U")


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


def _concepts(h, init):
    """h: [B, 197, D] block output. Returns u [B, 196, R] (columns sum to 1) and w [B, R] (sum to 1)."""
    h = h.float()
    X = torch.cat([h[:, 1:].clamp(min=0), (-h[:, 1:]).clamp(min=0)], -1)          # B, N, 2D
    xc = torch.cat([h[:, 0].clamp(min=0), (-h[:, 0]).clamp(min=0)], -1)           # B, 2D
    B, N, F = X.shape
    U = init["U"].expand(B, -1, -1).clone()                                        # B, N, R
    V = torch.bmm(U.transpose(1, 2), X) / (U.sum(1)[..., None] + EPS)              # B, R, F (weighted means)
    for _ in range(ITERS):
        U = U * torch.bmm(X, V.transpose(1, 2)) / (torch.bmm(U, torch.bmm(V, V.transpose(1, 2))) + EPS)
        V = V * torch.bmm(U.transpose(1, 2), X) / (torch.bmm(torch.bmm(U.transpose(1, 2), U), V) + EPS)
    u = U / (U.sum(1, keepdim=True) + EPS)
    vn = V / (V.norm(dim=-1, keepdim=True) + EPS)
    w = torch.bmm(vn, xc[..., None])[..., 0].clamp(min=0) + EPS                   # all-zero -> uniform, not raster
    w = w / w.sum(-1, keepdim=True)
    return u, w


def _greedy(u, w, k):
    B, N, R = u.shape
    cov = torch.zeros(B, R, device=u.device)
    picked = torch.zeros(B, N, dtype=torch.bool, device=u.device)
    for _ in range(k):
        gain = (w[:, None, :] * ((1 + cov[:, None, :] + u).log() - (1 + cov[:, None, :]).log())).sum(-1)
        gain = gain.masked_fill(picked, -1e9)
        i = gain.argmax(1)
        picked[torch.arange(B, device=u.device), i] = True
        cov = cov + u[torch.arange(B, device=u.device), i]
    return picked


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
                and int(float(r["keep"])) == 29 and r["seed"] in (0, 4, 5):
            students.append((f"lateroll29_s{r['seed']}", r_id, [29]))
    for r_id, r in sorted(runs.items()):
        if r.get("dataset") == "cub" and r["tier"] == "final" and r["method"] == "full" and r["seed"] == 1:
            students.append(("common_full196_s1", r_id, [12, 29, 49]))
    api.write_json("students", students)
    T = api.teacher.eval()
    n = api.num_patches
    gen = numpy.random.default_rng(20261101)
    g0 = numpy.random.default_rng(1101)
    init = {"U": torch.from_numpy(0.5 + g0.random((1, n, RANK))).float().to(api.device)}

    fulls = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            fulls.append(T(x).float().log_softmax(-1))

    summary = {}
    for name, rid, Ks in students:
        S = api.student(rid)
        for K in Ks:
            g = numpy.random.default_rng(20261101)
            acc = {c: {"kl": [], "tckd": []} for c in CONDS}
            ov = {"CC_L": [], "CW_L": [], "CC_CW": []}
            went = []
            for b, (x, y) in enumerate(api.val_batches(batch=BATCH)):
                B = x.shape[0]
                with torch.no_grad():
                    out = api.student_forward(S, x, attn_layers=list(range(SPLIT, len(S.blocks))),
                                              token_layers=[BLOCK])
                    attn = [out.attn[i].detach() for i in range(SPLIT, len(S.blocks))]
                    u, w = _concepts(out.tokens[BLOCK].detach(), init)
                L = _topk(_rollout(attn, 0), K)
                CC = _greedy(u, w, K)
                CCu = _greedy(u, torch.full_like(w, 1.0 / RANK), K)
                CW = _topk((u * w[:, None, :]).sum(-1), K)
                U_ = _topk(torch.from_numpy(g.random((B, n))).float().to(x.device), K)
                masks = {"L": L, "CC": CC, "CCu": CCu, "CW": CW, "U": U_}
                for c, mk in masks.items():
                    assert int(mk.sum(1).min()) == K and int(mk.sum(1).max()) == K
                    with torch.no_grad():
                        logits = api.teacher_forward_keepmask(T, x, mk)
                    kl, tc = _kl_tckd(fulls[b], logits, y)
                    acc[c]["kl"].append(kl.cpu()); acc[c]["tckd"].append(tc.cpu())
                ov["CC_L"].append(((CC & L).sum(1).float() / K).cpu())
                ov["CW_L"].append(((CW & L).sum(1).float() / K).cpu())
                ov["CC_CW"].append(((CC & CW).sum(1).float() / K).cpu())
                went.append((-(w.clamp(min=EPS) * w.clamp(min=EPS).log()).sum(-1)).cpu())
            v = {c: {m: torch.cat(a).float().numpy().astype(numpy.float64) for m, a in acc[c].items()}
                 for c in CONDS}
            o = {kk: torch.cat(a).float().numpy().astype(numpy.float64) for kk, a in ov.items()}
            res = {"run": rid, "K": K,
                   "mean": {c: {m: round(float(a.mean()), 4) for m, a in v[c].items()} for c in CONDS},
                   "overlap": {kk: round(float(a.mean()), 4) for kk, a in o.items()},
                   "w_entropy_mean": round(float(torch.cat(went).float().mean()), 4)}
            for c in CONDS[1:]:
                res[f"d_tckd_{c}_minus_L"] = _ci(v[c]["tckd"] - v["L"]["tckd"], gen)
                res[f"d_kl_{c}_minus_L"] = _ci(v[c]["kl"] - v["L"]["kl"], gen)
            res["d_tckd_CC_minus_CW"] = _ci(v["CC"]["tckd"] - v["CW"]["tckd"], gen)
            res["d_tckd_CC_minus_CCu"] = _ci(v["CC"]["tckd"] - v["CCu"]["tckd"], gen)
            key = f"{name}_k{K}"
            summary[key] = res
            api.write_json(f"cover_{key}", res)

    vk = [s for s in summary if s.startswith("lateroll29_")]
    verdict = {
        "P1_CC_minus_L_tckd_ci_hi_lt_0_all": all(summary[s]["d_tckd_CC_minus_L"][2] < 0 for s in vk),
        "P2_CC_minus_CW_tckd_ci_hi_lt_0_all": all(summary[s]["d_tckd_CC_minus_CW"][2] < 0 for s in vk),
        "P3_U_minus_L_tckd_ci_lo_gt_0_all": all(summary[s]["d_tckd_U_minus_L"][1] > 0 for s in vk),
        "n_verdict_students": len(vk),
    }
    api.write_json("summary", {"students": students, "cover": summary, "verdict": verdict})
