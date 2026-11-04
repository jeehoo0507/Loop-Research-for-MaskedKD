"""H-A150 (round 417, analysis only, no training): Aircraft replication of H-A149 (analysis/cub_lt_ttopk_sink_diff.py;
code unchanged except runs, bootstrap seed 20261158 and text). Original text: H-A147/H-A148 found that the teacher's own rollout top-29 (T29) is
MORE scattered than the final Rollout@29 student's rollout top-29 (R29), and that the set difference 1 - Jaccard(T29, R29)
tracks the extra non-target signal loss of the T29-masked teacher (Spearman +0.31 cub_lt, +0.14 Aircraft, recorded).
Two open questions: (1) are the teacher-only patches (D_T = T29 minus R29) the teacher's own high-norm "sink" tokens
(Darcet et al. 2024), i.e. low-content tokens the teacher's late attention collects? H-C10 measured sinks only for the
student (last-block attention vs rollout), never the teacher's sinks against the teacher set. (2) does the set
difference still explain dG after controlling for spread (partial Spearman)? cub_lt (CUB val/teacher), seeds 1-3.

Per image (clean val): teacher rollout r_T (as H-A147) -> T29. Teacher sinks: patch tokens whose L2 norm at the output
of block depth-2 (= input of the last block) exceeds median + 3 * MAD of that image's 196 patch norms (H-C10 rule,
applied to the teacher). Per seed s: R29_s = top-29 of the Rollout@29 student's rollout. D_T = T29 minus R29_s,
D_R = R29_s minus T29 (same size). sink_frac(D) = share of D that are teacher sinks (image skipped for seed s if D
empty). dSink_i = mean over seeds (with D non-empty) of sink_frac(D_T) - sink_frac(D_R).
G_set = non-target KL(full teacher || teacher on the set), T=1; dG_i = G_T29 - mean_s G_R29_s.
P1: mean dSink over images with a value, image bootstrap 2000, numpy default_rng(20261157), 95% quantile CI.
P2: partial Spearman(1 - J, dG | dSpread) (Pearson of rank residuals after linear fit on rank of dSpread), same bootstrap.
Record: sinks per image, sink share of T29 and of R29, raw Spearman(1 - J, dG), Spearman(dSink, dG),
Spearman(#sinks in T29 - #sinks in R29, dG), teacher last-block CLS attention mass on sinks, per-seed t of dSink.
Limit: clean val views, final students; correlational; no selector is built from this.
"""
import numpy
import torch

META = {
    "question": "on Aircraft (final Rollout@29, seeds 1-3), are the patches only in the teacher's own rollout top-29 "
                "(not in the student's rollout top-29) the teacher's high-norm sink tokens more often than the "
                "student-only patches, and does the set difference explain the extra masked-teacher non-target signal "
                "loss after controlling for spatial spread?",
    "method": "teacher and student attention rollout maps on clean val; top-29 sets and their differences; teacher "
              "high-norm token detection (per-image median+3MAD of block-10 output norms); non-target KL(full teacher "
              "|| masked teacher); image bootstrap; partial Spearman on ranks; seed t CI",
    "paper": "Darcet et al. ICLR 2024 (ViT register/sink tokens); Abnar & Zuidema ACL 2020 (attention rollout); "
             "Hinton et al. 2015 (dark knowledge); Zhao et al. CVPR 2022 (DKD, NCKD); Son et al. ECCV 2024 (MaskedKD); "
             "Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261158
K = 29
GRID = 14
MAD_K = 3.0
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


def _partial_spearman(a, b, c):
    ra, rb, rc = _rank(a), _rank(b), _rank(c)
    X = numpy.stack([numpy.ones_like(rc), rc], 1)
    ea = ra - X @ numpy.linalg.lstsq(X, ra, rcond=None)[0]
    eb = rb - X @ numpy.linalg.lstsq(X, rb, rcond=None)[0]
    return float(numpy.corrcoef(ea, eb)[0, 1])


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
    depth = len(Tm.blocks)
    batches, tl, tr, sk, lastmass = [], [], [], [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
            out = api.student_forward(Tm, x, attn_layers=list(range(depth)), token_layers=[depth - 2])
            tr.append(_rollout(out.attn).float())
            nrm = out.tokens[depth - 2][:, 1:].float().norm(dim=-1)
            med = nrm.median(dim=1, keepdim=True).values
            mad = (nrm - med).abs().median(dim=1, keepdim=True).values
            sink = nrm > med + MAD_K * mad
            sk.append(sink.cpu())
            cls = out.attn[-1].float().mean(1)[:, 0, 1:]
            lastmass.append(((cls * sink).sum(1) / cls.sum(1)).cpu())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    tn = _nt_logp(_logp(torch.cat(tl).numpy().astype(numpy.float64)), y)
    sink = torch.cat(sk).numpy()
    lmass = torch.cat(lastmass).numpy().astype(numpy.float64)

    GT, kT = _masked(api, Tm, batches, tr)
    G_T = _nt_kl(tn, _nt_logp(_logp(GT), y))
    spT = _spread(kT)
    nsT = (kT & sink).sum(1).astype(numpy.float64)

    seeds = sorted(RUNS)
    G_R, spR, jac, dsk, nsR, per_seed = {}, {}, {}, {}, {}, {}
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
        G_R[s] = _nt_kl(tn, _nt_logp(_logp(GR), y))
        spR[s] = _spread(kR)
        jac[s] = (kR & kT).sum(1) / (kR | kT).sum(1)
        DT, DR = kT & ~kR, kR & ~kT
        nd = DT.sum(1).astype(numpy.float64)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            fT = numpy.where(nd > 0, (DT & sink).sum(1) / nd, numpy.nan)
            fR = numpy.where(nd > 0, (DR & sink).sum(1) / nd, numpy.nan)
        dsk[s] = fT - fR
        nsR[s] = (kR & sink).sum(1).astype(numpy.float64)
        ok = numpy.isfinite(dsk[s])
        per_seed[str(s)] = {"n_images_with_diff": int(ok.sum()),
                            "mean_sink_frac_DT": round(float(numpy.nanmean(fT)), 4),
                            "mean_sink_frac_DR": round(float(numpy.nanmean(fR)), 4),
                            "dSink": round(float(numpy.nanmean(dsk[s])), 4),
                            "mean_sink_share_R29": round(float(nsR[s].mean() / K), 4),
                            "mean_jaccard_T_R": round(float(jac[s].mean()), 4),
                            "dG": round(float((G_T - G_R[s]).mean()), 4)}

    dS = numpy.nanmean(numpy.stack([dsk[s] for s in seeds]), 0)
    okS = numpy.isfinite(dS)
    dSp = numpy.mean([spR[s] for s in seeds], 0) - spT
    dG = G_T - numpy.mean([G_R[s] for s in seeds], 0)
    J = numpy.mean([jac[s] for s in seeds], 0)
    dNs = nsT - numpy.mean([nsR[s] for s in seeds], 0)

    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def boot(fn):
        b = numpy.array([fn(ix) for ix in idxs])
        return [round(float(numpy.nanquantile(b, 0.025)), 4), round(float(numpy.nanquantile(b, 0.975)), 4)]

    def verdict(lo, hi):
        return "positive" if lo > 0 else ("negative" if hi < 0 else "uncertain")

    def sub(ix):
        return ix[okS[ix]]

    P1 = [round(float(dS[okS].mean()), 4)] + boot(lambda ix: dS[sub(ix)].mean())
    P2 = [round(_partial_spearman(1 - J, dG, dSp), 4)] + boot(lambda ix: _partial_spearman(1 - J[ix], dG[ix], dSp[ix]))
    api.write_json("summary", {
        "n_val": int(n), "K": K, "n_images_P1": int(okS.sum()),
        "P1_mean_dSink_teacher_only_minus_student_only": P1,
        "P2_partial_spearman_1mJaccard_dG_given_dSpread": P2,
        "record": {
            "mean_sinks_per_image": round(float(sink.sum(1).mean()), 4),
            "mean_sink_share_T29": round(float(nsT.mean() / K), 4),
            "mean_sink_share_R29": round(float(numpy.mean([nsR[s] for s in seeds]) / K), 4),
            "teacher_last_block_cls_attn_mass_on_sinks": round(float(lmass.mean()), 4),
            "mean_dG": [round(float(dG.mean()), 4)] + boot(lambda ix: dG[ix].mean()),
            "spearman_1mJaccard_dG_raw": [round(_spearman(1 - J, dG), 4)] + boot(lambda ix: _spearman(1 - J[ix], dG[ix])),
            "spearman_dSink_dG": [round(_spearman(dS[okS], dG[okS]), 4)]
                                 + boot(lambda ix: _spearman(dS[sub(ix)], dG[sub(ix)])),
            "spearman_dNsinks_T_minus_R_dG": [round(_spearman(dNs, dG), 4)] + boot(lambda ix: _spearman(dNs[ix], dG[ix])),
            "signal_nt_kl_T": round(float(G_T.mean()), 4),
            "signal_acc_T": round(100 * float((GT.argmax(1) == y).mean()), 2),
            "per_seed_t_dSink": _t95([per_seed[str(s)]["dSink"] for s in seeds]),
        },
        "per_seed": per_seed,
        "verdict": {"P1": verdict(*P1[1:]), "P2": verdict(*P2[1:])}})
