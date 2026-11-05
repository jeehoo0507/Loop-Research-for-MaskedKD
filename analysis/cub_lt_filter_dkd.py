"""H-A174 (round 438, analysis only, no training): cub_lt version of H-A160, aimed at the open H-A171~173 mismatch.
On cub_lt final K=29 the trustfilt arm has the same masked-teacher KL as maskedkd (clean mean, median and crop+flip
views all uncertain: H-A171/172/173) yet +2.3 %p val accuracy. Equal total KL can hide different compositions. Exact
chain rule (Zhao et al. CVPR 2022): KL(p || q) = KL_bin([p_t, 1 - p_t] || [q_t, 1 - q_t]) + (1 - p_t) KL(p_nt || q_nt),
t = ground-truth class, p = full teacher, q = masked teacher, p_nt / q_nt = renormalised non-target distributions.
Does trustfilt lose more true-class probability (TCKD) but less non-target shape (wNCKD) than maskedkd?

Runs: the 18 cub_lt final K=29 runs of H-A171~173 (6 arms x seeds 1-3, same pinned ids, same masks as H-A173 clean).
Per run (own rule, own final student, clean val, teacher map = teacher all-block rollout = cache rule): mean KL, TCKD,
wNCKD (all images and tail-class images), masked p_true, top-1 agreement with the full teacher, masked accuracy.
P1: trustfilt - maskedkd TCKD; P2: trustfilt - maskedkd wNCKD; seed-paired t 95% CI (df 2).
Record: image bootstrap of both (numpy default_rng 20261187, 2000), tail splits, trustfilt - lateroll splits,
Spearman (TCKD / wNCKD vs val acc, tail versions vs tail acc) with one-sided permutation p (10000, same rng after
bootstrap), arm means.
Limit: clean val views, final students; correlation only; no selector is built from it.
"""
import math

import numpy
import torch

META = {
    "question": "on cub_lt final K=29 runs, is the equal masked-teacher KL of trustfilt and maskedkd made of different "
                "parts: more true-class probability loss (TCKD) but less non-target shape loss (NCKD) for trustfilt?",
    "method": "exact DKD chain-rule split of keep-only masked-teacher KL into TCKD and (1-p_t)-weighted NCKD, "
              "seed-paired t CI, image bootstrap, Spearman across runs with permutation test",
    "paper": "Zhao et al. CVPR 2022 (decoupled KD); Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & "
             "Zuidema ACL 2020 (attention rollout); Son et al. ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD); "
             "Wah et al. 2011 (CUB-200-2011)",
}

BATCH = 50
N_BOOT = 2000
N_PERM = 10000
SEED = 20261187
K = 29
WIDEN = 1.5
TAIL = 1
EPS = 1e-12
T_CRIT = 4.302653  # t(0.975, df 2)
RUNS = {
    "maskedkd": ["cub_lt_final_maskedkd_k29_s1_1103_234609_76d1", "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"],
    "rollout": ["cub_lt_final_rollout_k29_s1_1103_234609_8382", "cub_lt_final_rollout_k29_s2_1103_234647_9b4c",
                "cub_lt_final_rollout_k29_s3_1103_234704_0edf"],
    "lateroll": ["cub_lt_final_lateroll_k29_s1_1104_084157_2b15", "cub_lt_final_lateroll_k29_s2_1104_084204_9f09",
                 "cub_lt_final_lateroll_k29_s3_1104_084209_a3f8"],
    "trustmask_v1": ["cub_lt_final_trustmask_v1_k29_s1_1103_234609_10df",
                     "cub_lt_final_trustmask_v1_k29_s2_1103_234647_4e79",
                     "cub_lt_final_trustmask_v1_k29_s3_1103_234704_d748"],
    "trustfilt": ["cub_lt_final_trustfilt_k29_s1_1103_234609_dbe3", "cub_lt_final_trustfilt_k29_s2_1103_234647_43e9",
                  "cub_lt_final_trustfilt_k29_s3_1103_234704_8732"],
    "teacher_topk": ["cub_lt_final_teacher_topk_k29_s1_1103_234627_9ef0",
                     "cub_lt_final_teacher_topk_k29_s2_1103_234655_cfa0",
                     "cub_lt_final_teacher_topk_k29_s3_1103_234716_6d91"],
}


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _kl(tp, sp):
    return (numpy.exp(tp) * (tp - sp)).sum(1)


def _dkd(tp, sp, y):
    """Exact split KL = TCKD + wNCKD (per image)."""
    n = len(y)
    p, q = numpy.exp(tp), numpy.exp(sp)
    pt, qt = p[numpy.arange(n), y], q[numpy.arange(n), y]
    pt_c, qt_c = numpy.clip(pt, EPS, 1 - EPS), numpy.clip(qt, EPS, 1 - EPS)
    tckd = pt_c * numpy.log(pt_c / qt_c) + (1 - pt_c) * numpy.log((1 - pt_c) / (1 - qt_c))
    mask = numpy.ones_like(p, dtype=bool)
    mask[numpy.arange(n), y] = False
    pn = numpy.where(mask, p, 0.0) / (1 - pt_c)[:, None]
    qn = numpy.where(mask, q, 0.0) / (1 - qt_c)[:, None]
    nckd = (numpy.where(mask, pn * (numpy.log(pn + EPS) - numpy.log(qn + EPS)), 0.0)).sum(1)
    return tckd, (1 - pt_c) * nckd, qt


def _rank(a):
    r = numpy.empty(len(a))
    r[numpy.argsort(a, kind="stable")] = numpy.arange(len(a))
    return r


def _spearman(a, b):
    return float(numpy.corrcoef(_rank(a), _rank(b))[0, 1])


def _perm(a, b, rng):
    rho = _spearman(a, b)
    null = numpy.array([_spearman(a, rng.permutation(b)) for _ in range(N_PERM)])
    return [round(rho, 4), round(float((null <= rho).mean()), 4), round(float((null >= rho).mean()), 4)]


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


def _filt(s, t, k):
    m = int(math.ceil(WIDEN * k))
    allowed = t.argsort(1, descending=True).argsort(1) < m
    s = s.masked_fill(~allowed, -1.0)
    return s.argsort(1, descending=True).argsort(1) < k


def _keep(api, arm, S, x, tm):
    if arm == "teacher_topk":
        return _topk_mask(tm, K)
    if arm == "maskedkd":
        out = api.student_forward(S, x, attn_layers=[len(S.blocks) - 1])
        return _topk_mask(out.attn[-1].float().mean(1)[:, 0, 1:], K)
    if arm == "rollout":
        out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
        return _topk_mask(_rollout(out.attn), K)
    out = api.student_forward(S, x, attn_layers=list(range(6, len(S.blocks))))
    lr = _rollout([a for a in out.attn if a is not None])
    return _filt(lr, tm, K) if arm == "trustfilt" else _topk_mask(lr, K)


def _tcrit(d):
    d = numpy.asarray(d, dtype=float)
    h = T_CRIT * d.std(ddof=1) / math.sqrt(len(d))
    return [round(float(d.mean()), 4), round(float(d.mean() - h), 4), round(float(d.mean() + h), 4),
            [round(float(v), 4) for v in d]]


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid].get("val_wga_last")]
                                for arm in RUNS for rid in RUNS[arm]})
    groups = api.val_groups().cpu().numpy()
    Tm = api.teacher.eval()
    batches, tl, tmap = [], [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
            out = api.student_forward(Tm, x, attn_layers=list(range(len(Tm.blocks))))
            tmap.append(_rollout(out.attn).float())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    tail = groups[:n] == TAIL
    tlp = _logp(torch.cat(tl).numpy().astype(numpy.float64))
    full_top1 = tlp.argmax(1)
    full_pt = float(numpy.exp(tlp[numpy.arange(n), y]).mean())

    per_run, img = {}, {}
    for arm, ids in RUNS.items():
        for rid in ids:
            S = None if arm == "teacher_topk" else api.student(rid)
            gl = []
            for (x, _), tm in zip(batches, tmap):
                with torch.no_grad():
                    keep = _keep(api, arm, S, x, tm)
                    gl.append(api.teacher_forward_keepmask(Tm, x, keep).float().cpu())
            if S is not None:
                del S
                torch.cuda.empty_cache()
            glp = _logp(torch.cat(gl).numpy().astype(numpy.float64))
            kl = _kl(tlp, glp)
            tckd, wnckd, qt = _dkd(tlp, glp, y)
            img[rid] = {"kl": kl, "tckd": tckd, "wnckd": wnckd}
            per_run[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": float(runs[rid]["val_acc_last"]),
                            "val_tail_last": float(runs[rid]["val_wga_last"]),
                            "kl": round(float(kl.mean()), 4), "tckd": round(float(tckd.mean()), 4),
                            "wnckd": round(float(wnckd.mean()), 4),
                            "tail_kl": round(float(kl[tail].mean()), 4), "tail_tckd": round(float(tckd[tail].mean()), 4),
                            "tail_wnckd": round(float(wnckd[tail].mean()), 4),
                            "split_residual": round(float(numpy.abs(kl - tckd - wnckd).max()), 6),
                            "p_true": round(float(qt.mean()), 4),
                            "top1_agree_full": round(float((glp.argmax(1) == full_top1).mean() * 100), 2),
                            "t_acc": round(float((glp.argmax(1) == y).mean() * 100), 2)}

    def paired(a, b, key):
        return _tcrit([per_run[x][key] - per_run[z][key] for x, z in zip(RUNS[a], RUNS[b])])

    rng = numpy.random.default_rng(SEED)

    def boot(a, b, key, sel=None):
        d = numpy.mean([img[x][key] - img[z][key] for x, z in zip(RUNS[a], RUNS[b])], axis=0)
        d = d if sel is None else d[sel]
        m = len(d)
        bs = numpy.array([d[rng.integers(0, m, m)].mean() for _ in range(N_BOOT)])
        return [round(float(d.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
                round(float(numpy.quantile(bs, 0.975)), 4)]

    P1 = paired("trustfilt", "maskedkd", "tckd")
    P2 = paired("trustfilt", "maskedkd", "wnckd")
    raw = paired("trustfilt", "maskedkd", "kl")
    ids = [rid for arm in RUNS for rid in RUNS[arm]]

    def col(key):
        return numpy.array([per_run[r][key] for r in ids])

    def verdict(c):
        return "positive" if c[1] > 0 else ("negative" if c[2] < 0 else "uncertain")

    record = {
        "trustfilt_minus_maskedkd_kl_seed_t": raw,
        "trustfilt_minus_maskedkd_tckd_boot": boot("trustfilt", "maskedkd", "tckd"),
        "trustfilt_minus_maskedkd_wnckd_boot": boot("trustfilt", "maskedkd", "wnckd"),
        "trustfilt_minus_maskedkd_tail_tckd_seed_t": paired("trustfilt", "maskedkd", "tail_tckd"),
        "trustfilt_minus_maskedkd_tail_wnckd_seed_t": paired("trustfilt", "maskedkd", "tail_wnckd"),
        "trustfilt_minus_maskedkd_tail_tckd_boot": boot("trustfilt", "maskedkd", "tckd", tail),
        "trustfilt_minus_maskedkd_tail_wnckd_boot": boot("trustfilt", "maskedkd", "wnckd", tail),
        "trustfilt_minus_maskedkd_p_true_seed_t": paired("trustfilt", "maskedkd", "p_true"),
        "trustfilt_minus_maskedkd_top1_agree_seed_t": paired("trustfilt", "maskedkd", "top1_agree_full"),
        "trustfilt_minus_maskedkd_t_acc_seed_t": paired("trustfilt", "maskedkd", "t_acc"),
        "trustfilt_minus_lateroll_tckd_seed_t": paired("trustfilt", "lateroll", "tckd"),
        "trustfilt_minus_lateroll_wnckd_seed_t": paired("trustfilt", "lateroll", "wnckd"),
        "teacher_topk_minus_maskedkd_tckd_seed_t": paired("teacher_topk", "maskedkd", "tckd"),
        "teacher_topk_minus_maskedkd_wnckd_seed_t": paired("teacher_topk", "maskedkd", "wnckd"),
    }
    record["tckd_share_of_kl_gap"] = round(P1[0] / raw[0], 4) if raw[0] != 0 else None
    record["spearman"] = {
        "tckd_acc_18runs": _perm(col("tckd"), col("val_acc_last"), rng),
        "wnckd_acc_18runs": _perm(col("wnckd"), col("val_acc_last"), rng),
        "tail_tckd_tail_18runs": _perm(col("tail_tckd"), col("val_tail_last"), rng),
        "tail_wnckd_tail_18runs": _perm(col("tail_wnckd"), col("val_tail_last"), rng),
    }
    record["arm_means_kl_tckd_wnckd_ptrue_agree_acc_tail"] = {
        a: [round(float(numpy.mean([per_run[r][k] for r in RUNS[a]])), 4)
            for k in ("kl", "tckd", "wnckd", "p_true", "top1_agree_full", "val_acc_last", "val_tail_last")]
        for a in RUNS}

    api.write_json("summary", {
        "n_val": int(n), "n_tail": int(tail.sum()), "K": K, "n_runs": len(ids),
        "full_teacher_p_true": round(full_pt, 4),
        "P1_trustfilt_minus_maskedkd_tckd_seed_t": P1,
        "P2_trustfilt_minus_maskedkd_wnckd_seed_t": P2,
        "record": record,
        "per_run": per_run,
        "verdict": {"P1": verdict(P1), "P2": verdict(P2)},
        "note": "CI entries are [mean, low, high(, per-seed diffs s1..s3)]; seed_t pairs run s_i of one arm with s_i of "
                "the other; TCKD/NCKD use the ground-truth class; one rng shared in order: boot keys as listed, then "
                "spearman keys as listed; permutation entries are [rho, p_lower, p_upper]; kl should equal H-A171"})
