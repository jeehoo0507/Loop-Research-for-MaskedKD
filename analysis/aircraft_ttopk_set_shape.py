"""H-A148 (round 416, analysis only, no training): Aircraft replication of H-A147 (analysis/cub_lt_ttopk_set_shape.py;
code unchanged except runs, bootstrap seed 20261156 and text). Original text: H-A145/H-A146 found that the teacher's own rollout top-29
(teacher_topk rule) gives a masked teacher with more non-target signal loss than the final Rollout@29 student's rollout
top-29 (rollout - teacher_topk dG -0.62 cub_lt, -0.28 Aircraft, upper bound < 0). Why? Candidate: the teacher's top-29
set is spatially clumped (one blob), so the masked teacher sees less of the object/context than the student's set.
H-C14/15 measured spread/components only for student rollout sets (roll0 vs roll6) with a common CUB student;
the teacher set against the student set was not measured. cub_lt (CUB val/teacher), seeds 1-3.

Per image (clean val): teacher rollout map r_T (all teacher blocks, 0.5 A + 0.5 I, as harness/teacher_cache.py) ->
T29 = top-29 of r_T. Per seed s: final Rollout@29 student's rollout r_S -> R29_s = top-29 of r_S.
spread(set) = mean pairwise Euclidean distance between kept patches on the 14x14 grid.
G_set = non-target KL(full teacher || teacher on the set), T=1.
dSpread_i = mean_s spread(R29_s) - spread(T29); dG_i = G_T29 - mean_s G_R29_s.
P1: mean dSpread over val images, image bootstrap 2000, numpy default_rng(20261155), 95% quantile CI.
P2: Spearman(dSpread, dG), same bootstrap.
Record: 4-connected components of each set, top-29 mass of each map (own map, normalised), Jaccard(T29, R29_s),
Spearman(dG, 1 - Jaccard), Spearman(G_T29, spread_T29) within teacher, per-seed t of mean dSpread, masked acc.
Limit: clean val views, final students; spread is descriptive (no selector is built from it).
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final Rollout@29, seeds 1-3), is the teacher's own rollout top-29 patch set more spatially "
                "clumped than the student's rollout top-29, and are the images where it is relatively more clumped "
                "the images where the teacher-top-29-masked teacher loses more non-target signal?",
    "method": "teacher and student attention rollout maps on clean val; top-29 sets; mean pairwise grid distance and "
              "4-connected components; non-target KL(full teacher || masked teacher); paired differences, image "
              "bootstrap; Spearman across images; seed t CI",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Hinton et al. 2015 (dark knowledge); Zhao et al. CVPR 2022 "
             "(DKD, NCKD); Son et al. ECCV 2024 (MaskedKD); Darcet et al. ICLR 2024 (ViT register/sink tokens); "
             "Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261156
K = 29
GRID = 14
RUNS = {1: "aircraft_final_rollout_k29_s1_1102_132227_c3d7",
        2: "aircraft_final_rollout_k29_s2_1102_180325_20b6",
        3: "aircraft_final_rollout_k29_s3_1102_180341_8606"}

_YX = numpy.stack(numpy.divmod(numpy.arange(GRID * GRID), GRID), 1).astype(numpy.float64)


def _t95(d):
    mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
    return [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _nt_logp(logp, y):
    lp = logp.copy()
    lp[numpy.arange(len(y)), y] = -numpy.inf
    m = lp.max(1, keepdims=True)
    return lp - (m + numpy.log(numpy.exp(lp - m).sum(1, keepdims=True)))


def _nt_kl(tn, sn):
    pt = numpy.exp(tn)
    return numpy.where(pt > 0, pt * (tn - numpy.where(numpy.isfinite(sn), sn, 0.0)), 0.0).sum(1)


def _rank(a):
    r = numpy.empty(len(a))
    r[numpy.argsort(a, kind="stable")] = numpy.arange(len(a))
    return r


def _spearman(a, b):
    return float(numpy.corrcoef(_rank(a), _rank(b))[0, 1])


def _topk_mask(scores, k):
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(1, scores.topk(k, dim=1).indices, True)
    return keep


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _spread(keep):
    out = numpy.zeros(len(keep))
    for i, m in enumerate(keep):
        p = _YX[m]
        d = numpy.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        n = len(p)
        out[i] = d.sum() / (n * (n - 1))
    return out


def _comps(keep):
    out = numpy.zeros(len(keep))
    for i, m in enumerate(keep):
        g = m.reshape(GRID, GRID)
        seen = numpy.zeros_like(g)
        c = 0
        for y0 in range(GRID):
            for x0 in range(GRID):
                if g[y0, x0] and not seen[y0, x0]:
                    c += 1
                    stack = [(y0, x0)]
                    seen[y0, x0] = True
                    while stack:
                        yy, xx = stack.pop()
                        for ny, nx in ((yy + 1, xx), (yy - 1, xx), (yy, xx + 1), (yy, xx - 1)):
                            if 0 <= ny < GRID and 0 <= nx < GRID and g[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
        out[i] = c
    return out


def _mass(scores, keep):
    s = scores / scores.sum(1, keepdims=True)
    return (s * keep).sum(1)


def _masked(api, Tm, batches, maps):
    gs, ks = [], []
    for (x, _), sc in zip(batches, maps):
        with torch.no_grad():
            keep = _topk_mask(sc, K)
            gs.append(api.teacher_forward_keepmask(Tm, x, keep).float().cpu())
            ks.append(keep.cpu())
    return torch.cat(gs).numpy().astype(numpy.float64), torch.cat(ks).numpy()


def run(api):
    runs = api.runs()
    api.write_json("students", {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                         runs[rid]["seed"], runs[rid]["val_acc_last"]] for s, rid in sorted(RUNS.items())})
    Tm = api.teacher.eval()
    batches, tl, tr = [], [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
            out = api.student_forward(Tm, x, attn_layers=list(range(len(Tm.blocks))))
            tr.append(_rollout(out.attn).float())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    tn = _nt_logp(_logp(torch.cat(tl).numpy().astype(numpy.float64)), y)

    GT, kT = _masked(api, Tm, batches, tr)
    rT = torch.cat(tr).cpu().numpy().astype(numpy.float64)
    G_T = _nt_kl(tn, _nt_logp(_logp(GT), y))
    spT, cpT, msT = _spread(kT), _comps(kT), _mass(rT, kT)

    seeds = sorted(RUNS)
    G_R, spR, cpR, msR, jac, per_seed = {}, {}, {}, {}, {}, {}
    for s in seeds:
        S = api.student(RUNS[s])
        maps = []
        for x, _ in batches:
            with torch.no_grad():
                out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
                maps.append(_rollout(out.attn).float())
        del S
        torch.cuda.empty_cache()
        GR, kR = _masked(api, Tm, batches, maps)
        rS = torch.cat(maps).cpu().numpy().astype(numpy.float64)
        G_R[s] = _nt_kl(tn, _nt_logp(_logp(GR), y))
        spR[s], cpR[s], msR[s] = _spread(kR), _comps(kR), _mass(rS, kR)
        jac[s] = (kR & kT).sum(1) / (kR | kT).sum(1)
        per_seed[str(s)] = {"mean_spread_R": round(float(spR[s].mean()), 4),
                            "mean_comps_R": round(float(cpR[s].mean()), 4),
                            "mean_own_top29_mass_R": round(float(msR[s].mean()), 4),
                            "mean_jaccard_T_R": round(float(jac[s].mean()), 4),
                            "signal_nt_kl_R": round(float(G_R[s].mean()), 4),
                            "signal_acc_R": round(100 * float((GR.argmax(1) == y).mean()), 2),
                            "dSpread": round(float((spR[s] - spT).mean()), 4),
                            "dG": round(float((G_T - G_R[s]).mean()), 4),
                            "spearman_dSpread_dG": round(_spearman(spR[s] - spT, G_T - G_R[s]), 4)}

    dSp = numpy.mean([spR[s] for s in seeds], 0) - spT
    dCp = numpy.mean([cpR[s] for s in seeds], 0) - cpT
    dMs = numpy.mean([msR[s] for s in seeds], 0) - msT
    dG = G_T - numpy.mean([G_R[s] for s in seeds], 0)
    J = numpy.mean([jac[s] for s in seeds], 0)

    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def boot(fn):
        b = numpy.array([fn(ix) for ix in idxs])
        return [round(float(numpy.quantile(b, 0.025)), 4), round(float(numpy.quantile(b, 0.975)), 4)]

    def verdict(lo, hi):
        return "positive" if lo > 0 else ("negative" if hi < 0 else "uncertain")

    P1 = [round(float(dSp.mean()), 4)] + boot(lambda ix: dSp[ix].mean())
    P2 = [round(_spearman(dSp, dG), 4)] + boot(lambda ix: _spearman(dSp[ix], dG[ix]))
    api.write_json("summary", {
        "n_val": int(n), "K": K,
        "P1_mean_dSpread_rollout_minus_teacher_top29": P1,
        "P2_spearman_dSpread_dG": P2,
        "record": {
            "mean_spread_T": round(float(spT.mean()), 4),
            "mean_comps_T": round(float(cpT.mean()), 4),
            "mean_own_top29_mass_T": round(float(msT.mean()), 4),
            "signal_nt_kl_T": round(float(G_T.mean()), 4),
            "signal_acc_T": round(100 * float((GT.argmax(1) == y).mean()), 2),
            "mean_dG": [round(float(dG.mean()), 4)] + boot(lambda ix: dG[ix].mean()),
            "mean_dComps_R_minus_T": [round(float(dCp.mean()), 4)] + boot(lambda ix: dCp[ix].mean()),
            "mean_dMass_R_minus_T": [round(float(dMs.mean()), 4)] + boot(lambda ix: dMs[ix].mean()),
            "spearman_dComps_dG": [round(_spearman(dCp, dG), 4)] + boot(lambda ix: _spearman(dCp[ix], dG[ix])),
            "spearman_1mJaccard_dG": [round(_spearman(1 - J, dG), 4)] + boot(lambda ix: _spearman(1 - J[ix], dG[ix])),
            "mean_jaccard_T_R": round(float(J.mean()), 4),
            "within_teacher_spearman_spread_T_G_T": [round(_spearman(spT, G_T), 4)]
                                                    + boot(lambda ix: _spearman(spT[ix], G_T[ix])),
            "per_seed_t_dSpread": _t95([per_seed[str(s)]["dSpread"] for s in seeds]),
            "per_seed_t_spearman": _t95([per_seed[str(s)]["spearman_dSpread_dG"] for s in seeds]),
        },
        "per_seed": per_seed,
        "verdict": {"P1": verdict(*P1[1:]), "P2": verdict(*P2[1:])}})
