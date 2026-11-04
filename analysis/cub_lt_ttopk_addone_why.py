"""H-A153 (round 419, analysis only, no training): H-A151/H-A152 found that the set-level signal loss of the teacher's
own rollout top-29 (T29) vs the final Rollout@29 student's rollout top-29 (R29) sits at the patch level (single-patch
gap dM > 0, Spearman(dM, dG) +0.55/+0.65), but the rollout score inside the difference sets does not predict a
patch's value (rho ~ 0). Open question: is a difference patch's value explained by a non-rollout property relative to
the shared patches I = T29 & R29? Two candidates fixed before results:
  nov_T(p) = 1 - max_{q in I} cos(h_p, h_q), h = teacher output tokens of the last block (before final norm)
             (redundancy with what I already shows, teacher features; H-C6 tested student-feature novelty on CUB
             random/lateroll-12 bases, not teacher features, not these difference sets)
  adj(p)   = number of 4-neighbours of p (14x14 grid) that are in I (relative geometry only, no fixed location)
Ground truth per row (image, seed, p in D_T or D_R): G(I + p) = non-target KL(full teacher || teacher on I + {p}), T=1,
exactly as H-A151 (lower G = the patch restores more signal).
Per image-seed with I non-empty and 2|D| >= 6 rows: within-image Spearman (average ranks for ties) of G vs nov_T and
of G vs adj over the pooled D_T + D_R rows; skipped if a feature is constant. Image value = mean over seeds with a
value.
P1: mean within-image Spearman(nov_T, G); image bootstrap 2000, numpy default_rng(20261161), 95% quantile CI.
P2: mean within-image Spearman(adj, G); same images with a value, same bootstrap.
Record: mean nov_T and adj on D_T minus D_R (per image, mean over seeds; bootstrap), within-image Spearman of
teacher cos(h_p, h_CLS) vs G, per-seed t of P1/P2, number of images.
Limit: clean val views, final students; correlational; no selector is built from this.
"""
import numpy
import torch

META = {
    "question": "on cub_lt (final Rollout@29, seeds 1-3), is the single-patch value of patches in the difference "
                "between the teacher's own rollout top-29 and the student's rollout top-29 explained by the "
                "patch's teacher-feature novelty relative to the shared patches, or by its spatial adjacency to them?",
    "method": "teacher and student attention rollout maps on clean val; top-29 sets, intersection and differences; "
              "keep-only masked teacher on intersection plus one patch; non-target KL(full teacher || masked "
              "teacher); teacher last-block token cosine novelty; 4-neighbour count; within-image Spearman; "
              "image bootstrap; seed t CI",
    "paper": "Abnar & Zuidema ACL 2020 (attention rollout); Hinton et al. 2015 (dark knowledge); Zhao et al. CVPR 2022 "
             "(DKD, NCKD); Son et al. ECCV 2024 (MaskedKD); Wah et al. 2011 (CUB-200-2011)",
}

BATCH = 50
CHUNK = 256
N_BOOT = 2000
SEED = 20261161
K = 29
GRID = 14
MIN_ROWS = 6
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
    """Average ranks (ties share the mean rank)."""
    a = numpy.asarray(a, dtype=numpy.float64)
    o = numpy.argsort(a, kind="stable")
    s = a[o]
    r = numpy.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(numpy.corrcoef(ra, rb)[0, 1])


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


def _forward_rows(api, Tm, X, img, keeps):
    """X: all val images [N, ...] on device; img: image index per row; keeps: bool [rows, 196]. Returns logits."""
    out = []
    for a in range(0, len(img), CHUNK):
        ix = torch.as_tensor(img[a:a + CHUNK], device=X.device)
        kp = torch.as_tensor(keeps[a:a + CHUNK], device=X.device)
        with torch.no_grad():
            out.append(api.teacher_forward_keepmask(Tm, X[ix], kp).float().cpu())
    return torch.cat(out).numpy().astype(numpy.float64)


def _adj_count(keepI, p):
    r, c = divmod(int(p), GRID)
    n = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < GRID and 0 <= cc < GRID and keepI[rr * GRID + cc]:
            n += 1
    return n


def run(api):
    runs = api.runs()
    api.write_json("students", {str(s): [rid, runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                         runs[rid]["seed"], runs[rid]["val_acc_last"]] for s, rid in sorted(RUNS.items())})
    Tm = api.teacher.eval()
    depth = len(Tm.blocks)
    xs, ys, tl, tr, hp, hc = [], [], [], [], [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
            out = api.student_forward(Tm, x, attn_layers=list(range(depth)), token_layers=[depth - 1])
            tr.append(_rollout(out.attn).float().cpu())
            h = torch.nn.functional.normalize(out.tokens[depth - 1].float(), dim=-1)
            hp.append(h[:, 1:].cpu())
            hc.append(h[:, 0].cpu())
    X = torch.cat(xs)
    y = torch.cat(ys).cpu().numpy().astype(int)
    n = len(y)
    Hp = torch.cat(hp)
    Hc = torch.cat(hc)
    cos_cls = torch.einsum("npd,nd->np", Hp, Hc).numpy().astype(numpy.float64)
    tn = _nt_logp(_logp(torch.cat(tl).numpy().astype(numpy.float64)), y)
    rT = torch.cat(tr)
    kT = _topk_mask(rT, K).numpy()
    allimg = numpy.arange(n)

    seeds = sorted(RUNS)
    rho_nov, rho_adj, rho_cls, d_nov, d_adj, per_seed = {}, {}, {}, {}, {}, {}
    for s in seeds:
        S = api.student(RUNS[s])
        maps = []
        for a in range(0, n, BATCH):
            with torch.no_grad():
                out = api.student_forward(S, X[a:a + BATCH], attn_layers=list(range(len(S.blocks))))
                maps.append(_rollout(out.attn).float().cpu())
        del S
        torch.cuda.empty_cache()
        kR = _topk_mask(torch.cat(maps), K).numpy()
        kI = kT & kR
        rows_img, rows_p, rows_side = [], [], []
        for i in range(n):
            if not kI[i].any():
                continue
            for p in numpy.flatnonzero(kT[i] & ~kR[i]):
                rows_img.append(i); rows_p.append(p); rows_side.append(0)
            for p in numpy.flatnonzero(kR[i] & ~kT[i]):
                rows_img.append(i); rows_p.append(p); rows_side.append(1)
        rows_img = numpy.array(rows_img, dtype=int)
        rows_p = numpy.array(rows_p, dtype=int)
        rows_side = numpy.array(rows_side, dtype=int)
        keeps = kI[rows_img].copy()
        keeps[numpy.arange(len(rows_img)), rows_p] = True
        Gp = _nt_kl(tn[rows_img], _nt_logp(_logp(_forward_rows(api, Tm, X, rows_img, keeps)), y[rows_img]))
        nov = numpy.empty(len(rows_img))
        adj = numpy.empty(len(rows_img))
        for i in numpy.unique(rows_img):
            sel = numpy.flatnonzero(rows_img == i)
            Iidx = torch.as_tensor(numpy.flatnonzero(kI[i]))
            P = torch.as_tensor(rows_p[sel])
            sim = Hp[i][P] @ Hp[i][Iidx].T
            nov[sel] = (1.0 - sim.max(1).values).numpy()
            adj[sel] = [_adj_count(kI[i], p) for p in rows_p[sel]]
        rn, ra, rc = (numpy.full(n, numpy.nan) for _ in range(3))
        dn, da = numpy.full(n, numpy.nan), numpy.full(n, numpy.nan)
        for i in numpy.unique(rows_img):
            sel = rows_img == i
            if sel.sum() < MIN_ROWS:
                continue
            g = Gp[sel]
            rn[i] = _spearman(nov[sel], g)
            ra[i] = _spearman(adj[sel], g)
            rc[i] = _spearman(cos_cls[i, rows_p[sel]], g)
            t_, r_ = sel & (rows_side == 0), sel & (rows_side == 1)
            dn[i] = nov[t_].mean() - nov[r_].mean()
            da[i] = adj[t_].mean() - adj[r_].mean()
        rho_nov[s], rho_adj[s], rho_cls[s], d_nov[s], d_adj[s] = rn, ra, rc, dn, da
        per_seed[str(s)] = {"n_rows": int(len(rows_img)), "n_images_rho_nov": int(numpy.isfinite(rn).sum()),
                            "n_images_rho_adj": int(numpy.isfinite(ra).sum()),
                            "rho_nov_G": round(float(numpy.nanmean(rn)), 4),
                            "rho_adj_G": round(float(numpy.nanmean(ra)), 4),
                            "rho_coscls_G": round(float(numpy.nanmean(rc)), 4),
                            "mean_nov_DT": round(float(nov[rows_side == 0].mean()), 4),
                            "mean_nov_DR": round(float(nov[rows_side == 1].mean()), 4),
                            "mean_adj_DT": round(float(adj[rows_side == 0].mean()), 4),
                            "mean_adj_DR": round(float(adj[rows_side == 1].mean()), 4),
                            "mean_G_rows": round(float(Gp.mean()), 4)}

    def img_mean(d):
        st = numpy.stack([d[s] for s in seeds])
        ok = numpy.isfinite(st).any(0)
        v = numpy.full(n, numpy.nan)
        v[ok] = numpy.nanmean(st[:, ok], 0)
        return v

    v_nov, v_adj, v_cls = img_mean(rho_nov), img_mean(rho_adj), img_mean(rho_cls)
    v_dn, v_da = img_mean(d_nov), img_mean(d_adj)

    rng = numpy.random.default_rng(SEED)
    idxs = rng.integers(0, n, size=(N_BOOT, n))

    def stat(v):
        ok = numpy.isfinite(v)
        b = numpy.array([v[ix[ok[ix]]].mean() for ix in idxs])
        return [round(float(v[ok].mean()), 4), round(float(numpy.quantile(b, 0.025)), 4),
                round(float(numpy.quantile(b, 0.975)), 4), int(ok.sum())]

    def verdict(r):
        return "positive" if r[1] > 0 else ("negative" if r[2] < 0 else "uncertain")

    P1 = stat(v_nov)
    P2 = stat(v_adj)
    api.write_json("summary", {
        "n_val": int(n), "K": K,
        "P1_mean_within_image_spearman_novT_G": P1,
        "P2_mean_within_image_spearman_adj_G": P2,
        "record": {
            "within_image_spearman_teacher_coscls_G": stat(v_cls),
            "novT_DT_minus_DR": stat(v_dn),
            "adj_DT_minus_DR": stat(v_da),
            "per_seed_t_P1": _t95([per_seed[str(s)]["rho_nov_G"] for s in seeds]),
            "per_seed_t_P2": _t95([per_seed[str(s)]["rho_adj_G"] for s in seeds]),
        },
        "per_seed": per_seed,
        "verdict": {"P1": verdict(P1), "P2": verdict(P2)}})
