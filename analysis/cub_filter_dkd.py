"""H-A160 (round 424, analysis only, no training): H-A159 showed the teacher-area filter's extra masked-teacher KL
(trustfilt vs lateroll, CUB K=29) is a shape loss, not a temperature shift (tau* > 1: mass sits on wrong classes).
Where is that loss: in the true-class probability (TCKD) or in the shape among non-target classes (NCKD)? Exact
chain rule (Zhao et al. CVPR 2022): KL(p || q) = KL_bin([p_t, 1 - p_t] || [q_t, 1 - q_t]) + (1 - p_t) KL(p_nt || q_nt),
t = ground-truth class, p = full teacher, q = masked teacher, p_nt / q_nt = renormalised non-target distributions.

Runs: CUB final K=29 seeds 1-3, arms lateroll, trustfilt, rollout, maskedkd (12 runs, ids from H-A158 / H-A159).
Per run (own rule, own final student, clean val, teacher map = teacher all-block rollout = cache rule): mean KL, TCKD,
weighted NCKD (wNCKD), masked p_true, masked top-1 agreement with the full teacher, masked teacher accuracy.
P1: trustfilt - lateroll TCKD; P2: trustfilt - lateroll wNCKD; seed-paired t 95% CI (df 2). Image bootstrap as record.
Limit: clean val views, final students; correlation only; no selector is built from it.
"""
import math

import numpy
import torch

META = {
    "question": "on CUB final K=29 runs, is the teacher-area filter's extra masked-teacher KL (trustfilt vs lateroll) "
                "a loss of true-class probability (TCKD) or of the non-target shape (NCKD)?",
    "method": "exact DKD chain-rule split of keep-only masked-teacher KL into TCKD and (1-p_t)-weighted NCKD, "
              "seed-paired t CI and image bootstrap",
    "paper": "Zhao et al. CVPR 2022 (decoupled KD); Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & "
             "Zuidema ACL 2020 (attention rollout); Son et al. ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261169
K = 29
WIDEN = 1.5
EPS = 1e-12
T_CRIT = 4.302653  # t(0.975, df 2)
RUNS = {
    "lateroll": ["final_lateroll_k29_s1_1101_104303_ef38", "final_lateroll_k29_s2_1101_125958_6470",
                 "final_lateroll_k29_s3_1101_140937_ec98"],
    "trustfilt": ["final_trustfilt_k29_s1_1103_095835_0073", "final_trustfilt_k29_s2_1103_095842_42fd",
                  "final_trustfilt_k29_s3_1103_095847_a7af"],
    "rollout": ["final_rollout_k29_s1_1101_104303_3861", "final_rollout_k29_s2_1101_125958_2ad2",
                "final_rollout_k29_s3_1101_140937_8573"],
    "maskedkd": ["final_maskedkd_k29_s1_1101_104303_b676", "final_maskedkd_k29_s2_1101_125958_5b02",
                 "final_maskedkd_k29_s3_1101_140937_b00f"],
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


def _allowed(t, k):
    m = int(math.ceil(WIDEN * k))
    return t.argsort(1, descending=True).argsort(1) < m


def _filt(s, t, k):
    s = s.masked_fill(~_allowed(t, k), -1.0)
    return s.argsort(1, descending=True).argsort(1) < k


def _tcrit(d):
    d = numpy.asarray(d, dtype=float)
    h = T_CRIT * d.std(ddof=1) / math.sqrt(len(d))
    return [round(float(d.mean()), 4), round(float(d.mean() - h), 4), round(float(d.mean() + h), 4)]


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"]]
                                for arm in RUNS for rid in RUNS[arm]})
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
    tlp = _logp(torch.cat(tl).numpy().astype(numpy.float64))
    full_top1 = tlp.argmax(1)
    full_pt = float(numpy.exp(tlp[numpy.arange(n), y]).mean())

    per_run, img = {}, {}
    for arm, ids in RUNS.items():
        for rid in ids:
            S = api.student(rid)
            gl = []
            for (x, _), tm in zip(batches, tmap):
                with torch.no_grad():
                    if arm == "maskedkd":
                        out = api.student_forward(S, x, attn_layers=[len(S.blocks) - 1])
                        keep = _topk_mask(out.attn[-1].float().mean(1)[:, 0, 1:], K)
                    elif arm == "rollout":
                        out = api.student_forward(S, x, attn_layers=list(range(len(S.blocks))))
                        keep = _topk_mask(_rollout(out.attn), K)
                    else:
                        out = api.student_forward(S, x, attn_layers=list(range(6, len(S.blocks))))
                        lr = _rollout([a for a in out.attn if a is not None])
                        keep = _filt(lr, tm, K) if arm == "trustfilt" else _topk_mask(lr, K)
                    gl.append(api.teacher_forward_keepmask(Tm, x, keep).float().cpu())
            del S
            torch.cuda.empty_cache()
            glp = _logp(torch.cat(gl).numpy().astype(numpy.float64))
            kl = _kl(tlp, glp)
            tckd, wnckd, qt = _dkd(tlp, glp, y)
            img[rid] = {"kl": kl, "tckd": tckd, "wnckd": wnckd}
            per_run[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": float(runs[rid]["val_acc_last"]),
                            "kl": round(float(kl.mean()), 4), "tckd": round(float(tckd.mean()), 4),
                            "wnckd": round(float(wnckd.mean()), 4),
                            "split_residual": round(float(numpy.abs(kl - tckd - wnckd).max()), 6),
                            "p_true": round(float(qt.mean()), 4),
                            "top1_agree_full": round(float((glp.argmax(1) == full_top1).mean() * 100), 2),
                            "t_acc": round(float((glp.argmax(1) == y).mean() * 100), 2)}

    def paired(a, b, key):
        return _tcrit([per_run[x][key] - per_run[z][key] for x, z in zip(RUNS[a], RUNS[b])])

    rng = numpy.random.default_rng(SEED)

    def boot(a, b, key):
        d = numpy.mean([img[x][key] - img[z][key] for x, z in zip(RUNS[a], RUNS[b])], axis=0)
        bs = numpy.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        return [round(float(d.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
                round(float(numpy.quantile(bs, 0.975)), 4)]

    P1 = paired("trustfilt", "lateroll", "tckd")
    P2 = paired("trustfilt", "lateroll", "wnckd")
    raw = paired("trustfilt", "lateroll", "kl")
    ids = [rid for arm in RUNS for rid in RUNS[arm]]
    acc = numpy.array([per_run[r]["val_acc_last"] for r in ids])

    def verdict(c):
        return "positive" if c[1] > 0 else ("negative" if c[2] < 0 else "uncertain")

    api.write_json("summary", {
        "n_val": int(n), "K": K, "n_runs": len(ids), "full_teacher_p_true": round(full_pt, 4),
        "P1_trustfilt_minus_lateroll_tckd_seed_t": P1,
        "P2_trustfilt_minus_lateroll_wnckd_seed_t": P2,
        "record": {
            "trustfilt_minus_lateroll_kl_seed_t": raw,
            "tckd_share_of_kl_gap": round(P1[0] / raw[0], 4) if raw[0] != 0 else None,
            "trustfilt_minus_lateroll_tckd_boot": boot("trustfilt", "lateroll", "tckd"),
            "trustfilt_minus_lateroll_wnckd_boot": boot("trustfilt", "lateroll", "wnckd"),
            "trustfilt_minus_lateroll_p_true_seed_t": paired("trustfilt", "lateroll", "p_true"),
            "trustfilt_minus_lateroll_top1_agree_seed_t": paired("trustfilt", "lateroll", "top1_agree_full"),
            "trustfilt_minus_lateroll_t_acc_seed_t": paired("trustfilt", "lateroll", "t_acc"),
            "trustfilt_minus_rollout_tckd_seed_t": paired("trustfilt", "rollout", "tckd"),
            "trustfilt_minus_rollout_wnckd_seed_t": paired("trustfilt", "rollout", "wnckd"),
            "maskedkd_minus_trustfilt_tckd_seed_t": paired("maskedkd", "trustfilt", "tckd"),
            "maskedkd_minus_trustfilt_wnckd_seed_t": paired("maskedkd", "trustfilt", "wnckd"),
            "spearman_tckd_acc_12runs": round(_spearman(numpy.array([per_run[r]["tckd"] for r in ids]), acc), 4),
            "spearman_wnckd_acc_12runs": round(_spearman(numpy.array([per_run[r]["wnckd"] for r in ids]), acc), 4),
            "arm_means_kl_tckd_wnckd_ptrue_agree_acc": {
                a: [round(float(numpy.mean([per_run[r][k] for r in RUNS[a]])), 4)
                    for k in ("kl", "tckd", "wnckd", "p_true", "top1_agree_full", "val_acc_last")] for a in RUNS},
        },
        "per_run": per_run,
        "verdict": {"P1": verdict(P1), "P2": verdict(P2)},
        "note": "CI entries are [mean, low, high]; seed_t pairs run s_i of one arm with s_i of the other; "
                "TCKD/NCKD use the ground-truth class"})
