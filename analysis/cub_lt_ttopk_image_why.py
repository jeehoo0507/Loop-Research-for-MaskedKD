"""H-A155 (round 420, analysis only, no training): H-A145..H-A154 found that the teacher's own rollout top-29 (T29)
keeps less non-target signal than the final Rollout@29 student's rollout top-29 (R29) (dG = G_T29 - G_R29 > 0), and
that patch-level properties (rollout score, teacher-feature novelty, adjacency) barely explain it (|rho| < 0.06).
Move to the image level: is dG larger on images where the full teacher is less confident, or where the teacher
rollout map is flatter (lower own top-29 mass)? H-A147 recorded only the mean top-29 mass, not its image-level link.
cub_lt (CUB val/teacher), seeds 1-3, same runs as H-A147..H-A154.

Per image (clean val): conf_T = full-teacher max softmax prob (T=1). T29, R29_s, G as H-A147.
mass_T = own top-29 mass of the normalised teacher rollout map. dG_i = G_T29 - mean_s G_R29_s.
P1: Spearman(conf_T, dG), image bootstrap 2000, numpy default_rng(20261163), 95% quantile CI.
P2: Spearman(mass_T, dG), same bootstrap.
Record: partial Spearman (rank residuals) of each given the other, Spearman(conf_T, mass_T),
Spearman(nt-entropy_T, dG) (entropy of the full teacher's renormalised non-target distribution),
within-set Spearman(conf_T, G_T29) and (conf_T, mean G_R29), dG by conf_T tercile, per-seed t of both rhos.
Limit: clean val views, final students; correlation only; no selector is built from it.
"""
import numpy
import torch

META = {
    "question": "on cub_lt (final Rollout@29, seeds 1-3), is the image-level gap in masked-teacher non-target signal "
                "between the teacher's own rollout top-29 and the student's rollout top-29 larger on images where "
                "the full teacher is less confident or where the teacher rollout map is flatter?",
    "method": "teacher and student attention rollout maps on clean val; top-29 sets; non-target KL(full teacher || "
              "masked teacher); teacher max prob and own top-29 rollout mass; Spearman and rank-residual partial "
              "Spearman across images, image bootstrap; seed t CI",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Hinton et al. 2015 (dark knowledge); Zhao et al. CVPR 2022 "
             "(DKD, NCKD); Son et al. ECCV 2024 (MaskedKD); Wah et al. 2011 (CUB-200-2011)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261163
K = 29
RUNS = {1: "cub_lt_final_rollout_k29_s1_1103_234609_8382",
        2: "cub_lt_final_rollout_k29_s2_1103_234647_9b4c",
        3: "cub_lt_final_rollout_k29_s3_1103_234704_0edf"}


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
    tlp = _logp(torch.cat(tl).numpy().astype(numpy.float64))
    tn = _nt_logp(tlp, y)
    conf = numpy.exp(tlp).max(1)
    pn = numpy.exp(tn)
    ent = -numpy.where(pn > 0, pn * numpy.where(numpy.isfinite(tn), tn, 0.0), 0.0).sum(1)

    GT, kT = _masked(api, Tm, batches, tr)
    rT = torch.cat(tr).cpu().numpy().astype(numpy.float64)
    G_T = _nt_kl(tn, _nt_logp(_logp(GT), y))
    msT = _mass(rT, kT)

    seeds = sorted(RUNS)
    G_R, per_seed = {}, {}
    for s in seeds:
        S = api.student(RUNS[s])
        maps = []
        for x, _ in batches:
            with torch.no_grad():
                out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
                maps.append(_rollout(out.attn).float())
        del S
        torch.cuda.empty_cache()
        GR, _ = _masked(api, Tm, batches, maps)
        G_R[s] = _nt_kl(tn, _nt_logp(_logp(GR), y))
        d = G_T - G_R[s]
        per_seed[str(s)] = {"signal_nt_kl_R": round(float(G_R[s].mean()), 4),
                            "dG": round(float(d.mean()), 4),
                            "spearman_conf_dG": round(_spearman(conf, d), 4),
                            "spearman_mass_dG": round(_spearman(msT, d), 4)}

    GRm = numpy.mean([G_R[s] for s in seeds], 0)
    dG = G_T - GRm

    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def boot(fn):
        b = numpy.array([fn(ix) for ix in idxs])
        return [round(float(numpy.quantile(b, 0.025)), 4), round(float(numpy.quantile(b, 0.975)), 4)]

    def verdict(lo, hi):
        return "positive" if lo > 0 else ("negative" if hi < 0 else "uncertain")

    def sp(a, b):
        return [round(_spearman(a, b), 4)] + boot(lambda ix: _spearman(a[ix], b[ix]))

    def psp(a, b, c):
        return [round(_partial_spearman(a, b, c), 4)] + boot(lambda ix: _partial_spearman(a[ix], b[ix], c[ix]))

    P1 = sp(conf, dG)
    P2 = sp(msT, dG)
    edges = numpy.quantile(conf, [1 / 3, 2 / 3])
    terc = numpy.digitize(conf, edges)
    api.write_json("summary", {
        "n_val": int(n), "K": K,
        "P1_spearman_conf_T_dG": P1,
        "P2_spearman_mass_T_dG": P2,
        "record": {
            "mean_dG": [round(float(dG.mean()), 4)] + boot(lambda ix: dG[ix].mean()),
            "mean_conf_T": round(float(conf.mean()), 4),
            "mean_mass_T": round(float(msT.mean()), 4),
            "partial_spearman_conf_dG_given_mass": psp(conf, dG, msT),
            "partial_spearman_mass_dG_given_conf": psp(msT, dG, conf),
            "spearman_conf_mass": sp(conf, msT),
            "spearman_ntentropy_dG": sp(ent, dG),
            "spearman_conf_G_T29": sp(conf, G_T),
            "spearman_conf_G_R29_mean": sp(conf, GRm),
            "conf_tercile_edges": [round(float(e), 4) for e in edges],
            "dG_by_conf_tercile": [round(float(dG[terc == t].mean()), 4) for t in range(3)],
            "G_T29_by_conf_tercile": [round(float(G_T[terc == t].mean()), 4) for t in range(3)],
            "G_R29_by_conf_tercile": [round(float(GRm[terc == t].mean()), 4) for t in range(3)],
            "n_by_conf_tercile": [int((terc == t).sum()) for t in range(3)],
            "per_seed_t_spearman_conf": _t95([per_seed[str(s)]["spearman_conf_dG"] for s in seeds]),
            "per_seed_t_spearman_mass": _t95([per_seed[str(s)]["spearman_mass_dG"] for s in seeds]),
        },
        "per_seed": per_seed,
        "verdict": {"P1": verdict(*P1[1:]), "P2": verdict(*P2[1:])}})
