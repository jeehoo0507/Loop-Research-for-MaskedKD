"""H-A194 (round 458, analysis only, no training): H-A193 found that the mass-threshold per-view token count carries
per-image information (mass - shuffle -0.12) but its spread costs about as much through the convexity of KL in K
(shuffle - uniform +0.13), so mass ~ uniform. Information gain should scale ~linearly with the spread and the
convexity loss ~quadratically, so shrinking the counts toward M should leave a net gain. This tests that.

Students, views, score, mass counts K_i and q fitting: identical to H-A193 (analysis/caltech_mass_alloc.py).
Arms per target mean budget M in (12, 29, 49) (decomposition controls, not selectors):
  uniform     top-M per view
  massL       top-K_i(L) per view, K_i(L) = clip(rint(M + L * (K_i - M)), 4, 196), L in (0.25, 0.5, 0.75, 1.0)
  shufL       the same K_i(L) permuted across images within each view index (H-A193 permutation, default_rng(20261217))
L = 1.0 reproduces H-A193 mass / shuffle exactly (check).
Per image: KL(full teacher on the view || keep-only masked teacher on the same view), T=1, mean over the V views.
P1: KL_mass0.5 - KL_uniform, M = 29, aug views, full student.
P2: KL_mass0.5 - KL_mass1.0, M = 29, aug views, full student.
CI: 95% percentile bootstrap over val images (numpy default_rng(20261220), 2000 resamples, shared index).
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on Caltech-256 crop+flip views of val, at a matched mean teacher-token budget, does shrinking the "
                "student-rollout mass-threshold per-view token counts halfway toward the fixed K preserve the "
                "full-teacher soft label better than the fixed K and better than the unshrunk counts?",
    "method": "keep-only masked-teacher KL on RandomResizedCrop+flip views, lateroll ranking, shrunk mass-threshold vs "
              "fixed vs shuffled per-view token counts at equal mean count, paired image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Holtzman et al. ICLR 2020 (top-p "
             "nucleus threshold); Szegedy et al. 2015 (RandomResizedCrop); Efron & Tibshirani 1993",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261220
SHUF_SEED = 20261217
V = 2
IMG = 224
MS = (12, 29, 49)
K_MIN = 4
STUDENTS = {
    "full": "caltech256_final_full_k196_s1_1103_123120_d6de",
    "maskedkd98": "caltech256_final_maskedkd_k98_s1_1103_123120_9e43",
}
LAMS = (0.25, 0.5, 0.75, 1.0)
ARMS = ("uniform",) + tuple(f"mass{l}" for l in LAMS) + tuple(f"shuf{l}" for l in LAMS)


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _grids(n, device):
    """V crop+flip affine grids per val image, harness GpuAugment RandomResizedCrop formula, fixed draws (H-A181)."""
    gen = numpy.random.default_rng(20261203)
    area = gen.uniform(0.08, 1.0, (V, n))
    r = numpy.exp(gen.uniform(math.log(3 / 4), math.log(4 / 3), (V, n)))
    w = numpy.minimum(numpy.sqrt(area * r), 1.0)
    h = numpy.minimum(numpy.sqrt(area / r), 1.0)
    cx = (gen.uniform(0, 1, (V, n)) * 2 - 1) * (1 - w)
    cy = (gen.uniform(0, 1, (V, n)) * 2 - 1) * (1 - h)
    flip = numpy.where(gen.uniform(0, 1, (V, n)) < 0.5, -1.0, 1.0)
    th = numpy.zeros((V, n, 2, 3))
    th[:, :, 0, 0] = w * flip
    th[:, :, 0, 2] = cx
    th[:, :, 1, 1] = h
    th[:, :, 1, 2] = cy
    return torch.tensor(th, dtype=torch.float32, device=device)


def _view(x, theta):
    if theta is None:
        return x
    grid = F.affine_grid(theta, (len(x), 3, IMG, IMG), align_corners=False)
    return F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _counts(cum, q):
    """cum: (N, 196) cumulative sorted mass. Fewest tokens reaching q, clamped to [K_MIN, 196]."""
    return numpy.clip((cum < q).sum(1) + 1, K_MIN, cum.shape[1])


def _fit_q(cum, m):
    """Smallest q (bisection) whose mean count is >= m."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _counts(cum, mid).mean() >= m:
            hi = mid
        else:
            lo = mid
    return hi


def _mask_from_counts(scores, k):
    order = scores.argsort(dim=1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(scores.shape[1], device=scores.device).expand_as(order))
    return rank < k[:, None]


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"]] for rid in STUDENTS.values()})
    Tm = api.teacher.eval()
    xs, ys = [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
    y_all = torch.cat(ys)
    n = len(y_all)
    th = _grids(n, y_all.device)
    vts = ["clean"] + list(range(V))
    vsets = {"clean": ["clean"], "aug": list(range(V))}

    views, tl = [], {vt: [] for vt in vts}
    off = 0
    for x in xs:
        b = len(x)
        vb = [("clean", None)] + [(v, th[v, off:off + b]) for v in range(V)]
        for vt, theta in vb:
            with torch.no_grad():
                tl[vt].append(Tm(_view(x, theta)).float().cpu())
        views.append(vb)
        off += b
    yn = y_all.cpu().numpy().astype(int)
    tlp = {vt: _logp(torch.cat(tl[vt]).numpy().astype(numpy.float64)) for vt in vts}
    full_acc = {"clean": round(float((tlp["clean"].argmax(1) == yn).mean() * 100), 3),
                "aug": round(float(numpy.mean([(tlp[v].argmax(1) == yn).mean() for v in range(V)]) * 100), 3)}

    shuf = numpy.random.default_rng(SHUF_SEED)
    perms = {vt: shuf.permutation(n) for vt in vts}

    kl, acc, s_acc, kinfo = {}, {}, {}, {}
    for sname, rid in STUDENTS.items():
        S = api.student(rid)
        depth = len(S.blocks)
        sc = {vt: [] for vt in vts}
        sl = []
        for x, vb in zip(xs, views):
            for vt, theta in vb:
                xv = _view(x, theta)
                with torch.no_grad():
                    if vt == "clean":
                        sl.append(api.student_forward(S, xv).logits.float().cpu())
                    out = api.student_forward(S, xv, attn_layers=list(range(6, depth)))
                    r = _rollout([out.attn[i] for i in range(6, depth)])
                    sc[vt].append((r / r.sum(1, keepdim=True)).float().cpu())
        del S
        torch.cuda.empty_cache()
        s_acc[sname] = round(float((torch.cat(sl).numpy().argmax(1) == yn).mean() * 100), 3)
        sc = {vt: torch.cat(sc[vt]) for vt in vts}
        cum = {vt: numpy.cumsum(numpy.sort(sc[vt].numpy().astype(numpy.float64), 1)[:, ::-1], 1) for vt in vts}

        counts = {}
        for tag, vset in vsets.items():
            allcum = numpy.concatenate([cum[vt] for vt in vset], 0)
            for m in MS:
                q = _fit_q(allcum, m)
                kk = {vt: _counts(cum[vt], q) for vt in vset}
                ka = numpy.concatenate([kk[vt] for vt in vset])
                kinfo[(sname, tag, m)] = {
                    "q": round(float(q), 5), "mean_k": round(float(ka.mean()), 3),
                    "mean_k2_over_m2": round(float((ka.astype(numpy.float64) ** 2).mean() / m ** 2), 4),
                    "k_quantiles_5_25_50_75_95": [int(numpy.percentile(ka, p)) for p in (5, 25, 50, 75, 95)],
                    "frac_at_kmin": round(float((ka == K_MIN).mean()), 4),
                    "frac_at_196": round(float((ka == 196).mean()), 4)}
                for vt in vset:
                    counts[(vt, m, "uniform")] = numpy.full(n, m)
                    for lam in LAMS:
                        kl_ = numpy.clip(numpy.rint(m + lam * (kk[vt] - m)), K_MIN, 196).astype(int)
                        counts[(vt, m, f"mass{lam}")] = kl_
                        counts[(vt, m, f"shuf{lam}")] = kl_[perms[vt]]
                for lam in LAMS:
                    ka = numpy.concatenate([counts[(vt, m, f"mass{lam}")] for vt in vset]).astype(numpy.float64)
                    kinfo[(sname, tag, m)][f"mean_k_lam{lam}"] = round(float(ka.mean()), 3)
                    kinfo[(sname, tag, m)][f"mean_k2_over_m2_lam{lam}"] = round(float((ka ** 2).mean() / m ** 2), 4)

        gl = {key: [] for key in counts}
        off = 0
        for x, vb in zip(xs, views):
            b = len(x)
            for vt, theta in vb:
                xv = _view(x, theta)
                s_b = sc[vt][off:off + b].to(xv.device)
                for m in MS:
                    for arm in ARMS:
                        k = torch.as_tensor(counts[(vt, m, arm)][off:off + b], device=xv.device)
                        with torch.no_grad():
                            gl[(vt, m, arm)].append(
                                api.teacher_forward_keepmask(Tm, xv, _mask_from_counts(s_b, k)).float().cpu())
            off += b

        for tag, vset in vsets.items():
            for m in MS:
                for arm in ARMS:
                    ka, aa = [], []
                    for vt in vset:
                        glp = _logp(torch.cat(gl[(vt, m, arm)]).numpy().astype(numpy.float64))
                        ka.append((numpy.exp(tlp[vt]) * (tlp[vt] - glp)).sum(1))
                        aa.append((glp.argmax(1) == yn).astype(numpy.float64) * 100)
                    kl[(sname, tag, m, arm)] = numpy.mean(ka, 0)
                    acc[(sname, tag, m, arm)] = numpy.mean(aa, 0)
                kmean = numpy.mean([counts[(vt, m, "mass1.0")] for vt in vset], 0)
                kinfo[(sname, tag, m)]["spearman_k_vs_kl_uniform"] = _spearman(kmean, kl[(sname, tag, m, "uniform")])

    rng = numpy.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))

    def boot(dv):
        dv = numpy.asarray(dv, dtype=numpy.float64)
        bs = dv[idx].mean(1)
        return {"mean": round(float(dv.mean()), 4),
                "ci": [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]}

    p1 = boot(kl[("full", "aug", 29, "mass0.5")] - kl[("full", "aug", 29, "uniform")])
    p2 = boot(kl[("full", "aug", 29, "mass0.5")] - kl[("full", "aug", 29, "mass1.0")])
    rec = {}
    for s in STUDENTS:
        rec[s] = {"student_acc": s_acc[s]}
        for tag in vsets:
            rec[s][tag] = {}
            for m in MS:
                r_ = {"kl_mean": {a: round(float(kl[(s, tag, m, a)].mean()), 4) for a in ARMS},
                      "masked_teacher_acc": {a: round(float(acc[(s, tag, m, a)].mean()), 3) for a in ARMS},
                      "counts": kinfo[(s, tag, m)]}
                for lam in LAMS:
                    mk, sk = (s, tag, m, f"mass{lam}"), (s, tag, m, f"shuf{lam}")
                    r_[f"lam{lam}"] = {
                        "mass-uniform": boot(kl[mk] - kl[(s, tag, m, "uniform")]),
                        "mass-shuffle": boot(kl[mk] - kl[sk]),
                        "shuffle-uniform": boot(kl[sk] - kl[(s, tag, m, "uniform")]),
                        "mass-mass1.0": boot(kl[mk] - kl[(s, tag, m, "mass1.0")]),
                        "acc_mass-uniform": boot(acc[mk] - acc[(s, tag, m, "uniform")])}
                rec[s][tag][f"M{m}"] = r_
    api.write_json("summary", {
        "n_val": int(n), "views_per_image": V, "full_teacher_acc": full_acc,
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_kl_mass0.5_minus_uniform_M29_aug_full": p1,
        "P2_kl_mass0.5_minus_mass1.0_M29_aug_full": p2,
        "record": rec,
        "note": "KL(full teacher on view || keep-only masked teacher on same view), T=1, nats; aug = mean over V "
                "crop+flip views per image (H-A181 draws); lateroll = student rollout blocks 6-11 with residual "
                "identity, normalised; mass1.0 K_i = fewest lateroll-ranked tokens with cumulative mass >= q, clamp "
                "[4, 196], q fitted per (student, view set, M) so mean K_i ~ M (= H-A193 mass); massL = clip(rint(M + "
                "L (K_i - M)), 4, 196); shufL = same counts permuted across images within each view index (H-A193 "
                "permutation); teacher attention FLOPs grow with mean K^2 (mean_k2_over_m2*); 95% "
                "percentile bootstrap over val images (2000 resamples, shared index); seed-1 students only"})


def _spearman(a, b):
    ra = numpy.argsort(numpy.argsort(a)).astype(numpy.float64)
    rb = numpy.argsort(numpy.argsort(b)).astype(numpy.float64)
    return round(float(numpy.corrcoef(ra, rb)[0, 1]), 4)
