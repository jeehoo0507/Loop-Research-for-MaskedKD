"""H-C22 (QUEUE 2026-11-01, stage 0 go/no-go): per-image teacher token budgets on CUB.

H1  Do images need very different numbers of teacher tokens?
H2  Can a cheap student-side signal predict that need? Two branches on equal terms (same mean K, same token
    ranking, same students): interpretability signals (student-teacher attribution agreement, attribution spread,
    last-block vs rollout drift, SBAM-style last-block spread) and non-interpretability signals (predictive
    entropy, the KD loss the student would report through observe() at K=29), plus random.
H3  At the same mean K, is there room at all? Oracle allocation (analysis-only upper bound) vs fixed K vs
    all-or-none (Dynamic-KD form).
H4  (mentor) Is the per-image KD gain an inverted U in student-teacher attribution agreement?

Quality of the KD signal for one image = G(K) = || p_T - q_K ||_2, the L2 error of the logit gradient of the KD
term against the full-teacher KD term (tau = 1: d KL(q || softmax z)/dz = softmax z - q). An image with 0 tokens
gets CE only in the harness (harness/train.py: CE weight 1 instead of (1-alpha) CE + alpha KD), whose logit
gradient differs from full KD by alpha (p_T - y): so G(0) = || p_T - y ||_2 with the label y. The constant alpha
is the same on both sides and dropped. This puts "no teacher" and "masked teacher" on one scale.

Tokens inside an image: student attention rollout (primary, harness `rollout` rule) and last-block CLS attention
(secondary, MaskedKD / ALLOC_REFS rule). G at a non-grid K is interpolated linearly between grid points (the same
for every policy). Paired bootstrap over the 600 val images, numpy default_rng.
"""
import numpy
import torch

META = {
    "question": "CUB stage-0 go/no-go for per-image teacher token budgets: spread of per-image need (H1), cheap "
                "interpretability vs non-interpretability need signals (H2), oracle allocation headroom vs fixed K "
                "and all-or-none (H3), inverted U of KD gain in attribution agreement (H4)",
    "method": "keep-only masked teacher on a K grid per image; KD-gradient error ||p_T - q_K||; convex-hull "
              "greedy oracle allocation; rank-linear and all-or-none allocations from student signals; student vs "
              "teacher attention-rollout agreement",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Li et al. EMNLP 2021 (Dynamic "
             "KD); SBAM ECCV 2024; Hinton et al. 2015 (KD gradient)",
}

GRID = [0, 6, 12, 20, 29, 40, 49, 64, 80, 98, 128, 196]
BUDGETS = (29, 49)
NBOOT = 2000
SEED = 20261101
EPS = 1e-12
STUDENTS = [("lateroll29_e25", "screen_lateroll_k29_s0_1031_130348_fd19", 25),
            ("lateroll29_e50", "screen_lateroll_k29_s0_1031_130348_fd19", 50),
            ("lateroll29_final", "screen_lateroll_k29_s0_1031_130348_fd19", None),
            ("maskedkd29_e50", "screen_maskedkd_k29_s0_1031_130150_1393", 50),
            ("maskedkd29_final", "screen_maskedkd_k29_s0_1031_130150_1393", None)]
CE_RUNS = ["screen_ce_k0_s0_1031_111002_24e1", "screen_ce_k0_s4_1031_114911_08b9",
           "screen_ce_k0_s5_1031_114911_5fdb", "cub_final_ce_k0_s1_1030_153055_1256"]
FULL_RUNS = ["screen_full_k196_s0_1031_111002_00d5", "screen_full_k196_s4_1031_114911_a837",
             "screen_full_k196_s5_1031_114911_9c0a", "cub_final_full_k196_s1_1030_153055_d87b"]
SIGNALS = {   # name: (branch, sign); need = sign * signal, fixed before results
    "ent": ("non_interp", 1), "kd29": ("non_interp", 1),
    "align": ("interp", -1), "spread": ("interp", 1), "sal_last": ("interp", 1), "drift": ("interp", 1),
    "random": ("control", 1)}
PRIMARY = {"interp": "align", "non_interp": "kd29"}


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(dim=1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        r = a if r is None else a @ r
    m = r[:, 0, 1:]
    return m / m.sum(dim=1, keepdim=True)


def _rank(x):
    return x.argsort(dim=1).argsort(dim=1).float()


def _spearman_rows(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(1, keepdim=True)
    rb = rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1) + EPS)


def _eff(p):
    p = p / p.sum(1, keepdim=True)
    return torch.exp(-(p * p.clamp_min(EPS).log()).sum(1))


def _interp(G, k):
    """G [n, len(GRID)], k [n] float -> G at k, linear between grid points."""
    g = numpy.asarray(GRID, dtype=numpy.float64)
    out = numpy.empty(len(k))
    for i in range(len(k)):
        out[i] = numpy.interp(k[i], g, G[i])
    return out


def _lower_hull(pts):
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (p[0] - x1) >= (p[1] - y1) * (x2 - x1):   # hull[-1] on/above the chord: drop it
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def _oracle(G, budget, start_k=0):
    """Greedy on hull segments (exact for the convexified curves): returns per-image K."""
    n = G.shape[0]
    gi = GRID.index(start_k)
    sub = G[:, gi:]
    grid_sub = GRID[gi:]
    k = numpy.full(n, float(start_k))
    total = budget * n - start_k * n
    segs = []
    for i in range(n):
        hull = _lower_hull(list(zip(grid_sub, sub[i])))
        for (x1, y1), (x2, y2) in zip(hull, hull[1:]):
            if y2 < y1:
                segs.append((-(y2 - y1) / (x2 - x1), i, x2 - x1))
    segs.sort(key=lambda s: -s[0])
    used = 0.0
    for _, i, dk in segs:
        if used + dk > total:
            break
        k[i] += dk
        used += dk
    return k


def _ci(diff, idx):
    boot = diff[idx].mean(1)
    return [round(float(diff.mean()), 5), round(float(numpy.percentile(boot, 2.5)), 5),
            round(float(numpy.percentile(boot, 97.5)), 5)]


def _sp(a, b):
    ra, rb = numpy.argsort(numpy.argsort(a)), numpy.argsort(numpy.argsort(b))
    return round(float(numpy.corrcoef(ra, rb)[0, 1]), 4)


def run(api):
    T = api.teacher.eval()
    depth = len(T.blocks)
    xs = list(api.val_batches(batch=50))
    n = sum(len(y) for _, y in xs)
    gen = numpy.random.default_rng(SEED)
    idx = gen.integers(0, n, size=(NBOOT, n))
    rand_need = gen.random(n)

    # teacher side, once
    pT, tmap, yall = [], [], []
    with torch.no_grad():
        for x, y in xs:
            out = api.student_forward(T, x, attn_layers=list(range(depth)))
            pT.append(out.logits.float().softmax(-1))
            tmap.append(_rollout(out.attn))
            yall.append(y)
    yall = torch.cat(yall)
    C = pT[0].shape[1]

    res = {"grid": GRID, "n_images": n, "students": {}}
    for name, rid, ep in STUDENTS:
        S = api.student(rid, ep)
        G = {o: [] for o in ("roll", "last")}
        KL = {o: [] for o in ("roll", "last")}
        sig = {s: [] for s in SIGNALS if s != "random"}
        with torch.no_grad():
            for b, (x, y) in enumerate(xs):
                out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
                ps = out.logits.float().softmax(-1)
                roll = _rollout(out.attn)
                last = out.attn[-1].float().mean(1)[:, 0, 1:]
                pt = pT[b]
                onehot = torch.nn.functional.one_hot(y, C).float()
                sig["ent"].append(-(ps * ps.clamp_min(EPS).log()).sum(-1))
                sig["align"].append(_spearman_rows(roll, tmap[b]))
                sig["spread"].append(_eff(roll))
                sig["sal_last"].append(_eff(last))
                sig["drift"].append(1 - _spearman_rows(last, roll))
                for o, sc in (("roll", roll), ("last", last)):
                    order = sc.argsort(dim=1, descending=True)
                    gs, kls = [], []
                    for k in GRID:
                        if k == 0:
                            q = onehot
                        elif k == 196:
                            q = pt
                        else:
                            keep = torch.zeros_like(sc, dtype=torch.bool).scatter_(1, order[:, :k], True)
                            q = api.teacher_forward_keepmask(T, x, keep).float().softmax(-1)
                        if o == "roll" and k == 29:
                            sig["kd29"].append((q * (q.clamp_min(EPS).log() - ps.clamp_min(EPS).log())).sum(-1))
                        gs.append((pt - q).norm(dim=1))
                        kls.append((pt * (pt.clamp_min(EPS).log() - q.clamp_min(EPS).log())).sum(-1) if k else
                                   torch.full_like(gs[-1], float("nan")))
                    G[o].append(torch.stack(gs, 1))
                    KL[o].append(torch.stack(kls, 1))
        G = {o: torch.cat(v).float().cpu().numpy().astype(numpy.float64) for o, v in G.items()}
        KL = {o: torch.cat(v).float().cpu().numpy().astype(numpy.float64) for o, v in KL.items()}
        s = {k: torch.cat(v).float().cpu().numpy().astype(numpy.float64) for k, v in sig.items()}
        s["random"] = rand_need
        r = {"run": rid, "epoch": ep}
        for o in ("roll", "last"):
            g = G[o]
            ro = {"mean_G_by_K": [round(float(v), 5) for v in g.mean(0)],
                  "mean_KL_by_K": [None if numpy.isnan(v) else round(float(v), 5) for v in numpy.nanmean(KL[o], 0)],
                  "frac_masked_worse_than_label": {str(k): round(float((g[:, j] > g[:, 0]).mean()), 4)
                                                   for j, k in enumerate(GRID) if 0 < k < 196}}
            # H1: per-image need at the quality of fixed K=29
            tau = float(g[:, GRID.index(29)].mean())
            ok = g <= tau
            kstar = numpy.array([GRID[int(numpy.argmax(row))] for row in ok], dtype=numpy.float64)
            q = numpy.percentile(kstar, [10, 25, 50, 75, 90])
            ro["H1"] = {"tau_meanG29": round(tau, 5), "kstar_q10_25_50_75_90": [float(v) for v in q],
                        "frac_kstar0": round(float((kstar == 0).mean()), 4),
                        "hist": {str(k): int((kstar == k).sum()) for k in GRID},
                        "pass": bool(q[1] <= 12 and q[3] >= 49)}
            pol = {}
            for B in BUDGETS:
                fixed = g[:, GRID.index(B)] if B in GRID else _interp(g, numpy.full(n, float(B)))
                P = {}
                ko = _oracle(g, B, 0)
                P["oracle_cont"] = (_interp(g, ko), ko)
                ko6 = _oracle(g, B, 6)
                P["oracle_nozero"] = (_interp(g, ko6), ko6)
                m = int(B * n // 196)
                top = numpy.zeros(n, dtype=bool)
                top[numpy.argsort(-g[:, 0], kind="stable")[:m]] = True
                P["oracle_aon"] = (numpy.where(top, 0.0, g[:, 0]), numpy.where(top, 196.0, 0.0))
                for sn, (_, sign) in SIGNALS.items():
                    need = sign * s[sn]
                    rk = numpy.argsort(numpy.argsort(need, kind="stable"), kind="stable").astype(numpy.float64)
                    k_lin = 6 + (B - 6) * 2 * (rk + 0.5) / n
                    P[f"{sn}_lin"] = (_interp(g, k_lin), k_lin)
                    top = numpy.zeros(n, dtype=bool)
                    top[numpy.argsort(-need, kind="stable")[:m]] = True
                    P[f"{sn}_aon"] = (numpy.where(top, 0.0, g[:, 0]), numpy.where(top, 196.0, 0.0))
                out_b = {"fixed_meanG": round(float(fixed.mean()), 5)}
                for pn, (gv, kv) in P.items():
                    d = gv - fixed
                    ci = _ci(d, idx)
                    out_b[pn] = {"meanG": round(float(gv.mean()), 5), "meanK": round(float(kv.mean()), 3),
                                 "minus_fixed": ci, "rel": round(float(d.mean() / fixed.mean()), 4)}
                ok_or = out_b["oracle_cont"]
                out_b["H3_pass"] = bool(ok_or["minus_fixed"][2] < 0 and ok_or["rel"] <= -0.10)
                for a, bb in (("oracle_cont", "oracle_aon"), ("oracle_cont", "oracle_nozero")):
                    out_b[f"{a}_minus_{bb}"] = _ci(P[a][0] - P[bb][0], idx)
                out_b["interp_minus_noninterp_lin"] = _ci(P[f"{PRIMARY['interp']}_lin"][0] -
                                                         P[f"{PRIMARY['non_interp']}_lin"][0], idx)
                ko_arr = P["oracle_cont"][1]
                out_b["spearman_need_vs_oracleK"] = {sn: _sp(SIGNALS[sn][1] * s[sn], ko_arr) for sn in SIGNALS}
                pol[str(B)] = out_b
            ro["policies"] = pol
            gain = g[:, GRID.index(12)] - g[:, GRID.index(49)]
            ro["spearman_need_vs_gain12to49"] = {sn: _sp(SIGNALS[sn][1] * s[sn], gain) for sn in SIGNALS}
            r[o] = ro
        r["signal_means"] = {k: round(float(v.mean()), 4) for k, v in s.items()}
        res["students"][name] = r
        del S
        torch.cuda.empty_cache()

    # verdicts (rollout ordering, budget 29, prereg rules)
    st = res["students"]
    v = {"H1_students_pass": sum(st[k]["roll"]["H1"]["pass"] for k in st),
         "H3_students_pass": sum(st[k]["roll"]["policies"]["29"]["H3_pass"] for k in st)}
    v["H1"] = v["H1_students_pass"] >= 3
    v["H3"] = v["H3_students_pass"] >= 3
    wins = {}
    for sn in SIGNALS:
        for form in ("lin", "aon"):
            c = sum(st[k]["roll"]["policies"]["29"][f"{sn}_{form}"]["minus_fixed"][2] < 0 for k in st)
            wins[f"{sn}_{form}"] = c
    v["signal_students_beating_fixed29"] = wins
    v["H2_any_signal"] = any(c >= 3 for kk, c in wins.items() if not kk.startswith("random"))
    cmp = [st[k]["roll"]["policies"]["29"]["interp_minus_noninterp_lin"] for k in st]
    v["branch"] = ("interp" if sum(c[2] < 0 for c in cmp) >= 3 else
                   "non_interp" if sum(c[1] > 0 for c in cmp) >= 3 else "uncertain")
    res["verdict"] = v

    # H4: per-image KD gain (full - CE correctness over 4 seeds each) vs CE students' attribution agreement
    corr = {"ce": [], "full": []}
    align_ce = []
    with torch.no_grad():
        for grp, ids in (("ce", CE_RUNS), ("full", FULL_RUNS)):
            for rid in ids:
                S = api.student(rid)
                c, al = [], []
                for b, (x, y) in enumerate(xs):
                    out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
                    c.append((out.logits.float().argmax(-1) == y).float())
                    if grp == "ce":
                        al.append(_spearman_rows(_rollout(out.attn), tmap[b]))
                corr[grp].append(torch.cat(c).cpu().numpy().astype(numpy.float64))
                if grp == "ce":
                    align_ce.append(torch.cat(al).float().cpu().numpy().astype(numpy.float64))
                del S
    kdgain = numpy.mean(corr["full"], 0) - numpy.mean(corr["ce"], 0)
    al = numpy.mean(align_ce, 0)
    qs = numpy.percentile(al, [20, 40, 60, 80])
    bins = numpy.digitize(al, qs)
    bm = [round(float(kdgain[bins == b].mean()), 4) for b in range(5)]
    mid = numpy.isin(bins, [1, 2, 3])
    d1 = numpy.zeros((NBOOT,))
    d5 = numpy.zeros((NBOOT,))
    for t in range(NBOOT):
        ii = idx[t]
        g_, b_ = kdgain[ii], bins[ii]
        mm = g_[numpy.isin(b_, [1, 2, 3])].mean()
        d1[t] = mm - g_[b_ == 0].mean()
        d5[t] = mm - g_[b_ == 4].mean()
    h4 = {"ce_runs": CE_RUNS, "full_runs": FULL_RUNS, "mean_kdgain": round(float(kdgain.mean()), 4),
          "align_quintile_edges": [round(float(v), 4) for v in qs], "kdgain_by_align_quintile": bm,
          "mid_minus_q1": [round(float(kdgain[mid].mean() - kdgain[bins == 0].mean()), 4),
                           round(float(numpy.percentile(d1, 2.5)), 4), round(float(numpy.percentile(d1, 97.5)), 4)],
          "mid_minus_q5": [round(float(kdgain[mid].mean() - kdgain[bins == 4].mean()), 4),
                           round(float(numpy.percentile(d5, 2.5)), 4), round(float(numpy.percentile(d5, 97.5)), 4)],
          "spearman_align_kdgain": _sp(al, kdgain),
          "ce_correct_mean": round(float(numpy.mean(corr["ce"])), 4),
          "full_correct_mean": round(float(numpy.mean(corr["full"])), 4)}
    h4["inverted_u"] = bool(h4["mid_minus_q1"][1] > 0 and h4["mid_minus_q5"][1] > 0)
    res["H4"] = h4
    api.write_json("summary", res)
