"""H-A179 (round 443, analysis only, no training): signal-side follow-up of H-A178.
H-A178: on Waterbirds val images where the full teacher is wrong, the trustfilt student copies the teacher's wrong
label less often (70.0%) than the lateroll (77.2%) or Full KD (79.3%) student. Does trustfilt's own masked teacher
(teacher region filter + lateroll choice, K=29) already disagree with the full teacher's errors more often than the
lateroll masked teacher (same student-side rule without the filter)? With 2 classes, any masking noise that flips a
prediction "rescues" a teacher error, so the harm side (full teacher right -> masked teacher wrong) is measured too.

Runs: Waterbirds final K=29 seeds 1-3, arms maskedkd, lateroll, trustfilt, teacher_topk (12 runs, dataset waterbirds).
Masks as in H-A170 (clean val, teacher map = teacher all-block rollout recomputed on val).
Per run: rescue = masked-teacher accuracy on val images the full teacher gets wrong (W); harm = masked-teacher error
on images the full teacher gets right (R); same on group 2 only; mean masked-teacher p(y) on W; student copy rate on W
(student pred == full teacher wrong pred); student accuracy check vs metrics.
P1: trustfilt - lateroll rescue on W, same-seed pairs, t 95% CI (df 2).
P2: trustfilt - lateroll harm on R, same scheme.
Record: group-2 versions, trustfilt - maskedkd, lateroll - maskedkd, teacher_topk - lateroll, net (rescue - harm),
p(y) on W, Spearman(rescue, student copy rate on W) over 12 runs (one-sided permutation, numpy default_rng(20261201)).
Limit: clean val views, final students' masks (not the training-time masks), correlation only, W has ~79 images.
"""
import math

import numpy
import torch

META = {
    "question": "on Waterbirds final K=29, does trustfilt's own masked teacher disagree with the full teacher's errors "
                "(and not with its correct answers) more often than the lateroll masked teacher?",
    "method": "keep-only masked-teacher accuracy split by full-teacher correctness (rescue vs harm), student and teacher "
              "attention rollout, same-seed paired t intervals",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); Son et al. "
             "ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

BATCH = 50
N_PERM = 10000
SEED = 20261201
K = 29
WIDEN = 1.5
WORST = 2
T_CRIT = 4.302652729911275
RUNS = {
    "maskedkd": ["waterbirds_final_maskedkd_k29_s1_1031_201502_4be8", "waterbirds_final_maskedkd_k29_s2_1103_170931_92bd",
                 "waterbirds_final_maskedkd_k29_s3_1103_170935_907a"],
    "lateroll": ["waterbirds_final_lateroll_k29_s1_1104_025003_b396", "waterbirds_final_lateroll_k29_s2_1104_025008_c94e",
                 "waterbirds_final_lateroll_k29_s3_1104_025012_629b"],
    "trustfilt": ["waterbirds_final_trustfilt_k29_s1_1103_171121_24be", "waterbirds_final_trustfilt_k29_s2_1103_171126_b33d",
                  "waterbirds_final_trustfilt_k29_s3_1103_171130_db65"],
    "teacher_topk": ["waterbirds_final_teacher_topk_k29_s1_1103_170914_f015",
                     "waterbirds_final_teacher_topk_k29_s2_1103_170918_4535",
                     "waterbirds_final_teacher_topk_k29_s3_1103_170923_d12a"],
}


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


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


def _tci(d):
    d = numpy.asarray(d, dtype=numpy.float64)
    m = float(d.mean())
    h = T_CRIT * float(d.std(ddof=1)) / math.sqrt(len(d))
    return {"mean": round(m, 3), "ci": [round(m - h, 3), round(m + h, 3)], "per_seed": [round(float(v), 3) for v in d]}


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
    g2 = groups[:n] == WORST
    tlp = _logp(torch.cat(tl).numpy().astype(numpy.float64))
    tpred = tlp.argmax(1)
    W = tpred != y
    R = ~W

    per_run, rec = {}, {a: [] for a in RUNS}
    for arm, ids in RUNS.items():
        for rid in ids:
            S = api.student(rid)
            gl, sl = [], []
            for (x, _), tm in zip(batches, tmap):
                with torch.no_grad():
                    if arm == "maskedkd":
                        out = api.student_forward(S, x, attn_layers=[len(S.blocks) - 1])
                        keep = _topk_mask(out.attn[-1].float().mean(1)[:, 0, 1:], K)
                    elif arm == "teacher_topk":
                        keep = _topk_mask(tm, K)
                    else:
                        out = api.student_forward(S, x, attn_layers=list(range(6, len(S.blocks))))
                        lr = _rollout([a for a in out.attn if a is not None])
                        keep = _filt(lr, tm, K) if arm == "trustfilt" else _topk_mask(lr, K)
                    gl.append(api.teacher_forward_keepmask(Tm, x, keep).float().cpu())
                    sl.append(api.student_forward(S, x).logits.float().cpu())
            del S
            torch.cuda.empty_cache()
            glp = _logp(torch.cat(gl).numpy().astype(numpy.float64))
            spred = torch.cat(sl).numpy().argmax(1)
            gpred = glp.argmax(1)
            py = numpy.exp(glp[numpy.arange(n), y])
            r = {
                "rescue": float((gpred[W] == y[W]).mean() * 100),
                "harm": float((gpred[R] != y[R]).mean() * 100),
                "rescue_g2": float((gpred[W & g2] == y[W & g2]).mean() * 100),
                "harm_g2": float((gpred[R & g2] != y[R & g2]).mean() * 100),
                "py_W": float(py[W].mean()),
                "py_W_g2": float(py[W & g2].mean()),
                "copy_W": float((spred[W] == tpred[W]).mean() * 100),
                "t_acc": float((gpred == y).mean() * 100),
                "s_acc": float((spred == y).mean() * 100),
            }
            r["net"] = r["rescue"] - r["harm"]
            r["net_g2"] = r["rescue_g2"] - r["harm_g2"]
            rec[arm].append(r)
            per_run[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": runs[rid]["val_acc_last"],
                            **{k: round(v, 4) for k, v in r.items()}}

    def diff(a, b, key):
        return _tci([ra[key] - rb[key] for ra, rb in zip(rec[a], rec[b])])

    keys = ["rescue", "harm", "net", "rescue_g2", "harm_g2", "net_g2", "py_W", "py_W_g2", "copy_W", "t_acc"]
    pairs = [("trustfilt", "lateroll"), ("trustfilt", "maskedkd"), ("lateroll", "maskedkd"),
             ("teacher_topk", "lateroll"), ("trustfilt", "teacher_topk")]
    rng = numpy.random.default_rng(SEED)
    allr = [r for a in RUNS for r in rec[a]]
    api.write_json("summary", {
        "n_val": int(n), "n_W": int(W.sum()), "n_R": int(R.sum()), "n_W_g2": int((W & g2).sum()),
        "n_R_g2": int((R & g2).sum()), "teacher_acc": round(float(R.mean() * 100), 3),
        "P1_trustfilt_minus_lateroll_rescue": diff("trustfilt", "lateroll", "rescue"),
        "P2_trustfilt_minus_lateroll_harm": diff("trustfilt", "lateroll", "harm"),
        "record": {
            "pairs": {f"{a}-{b}": {k: diff(a, b, k) for k in keys} for a, b in pairs},
            "spearman_rescue_copyW_runs": _perm(numpy.array([r["rescue"] for r in allr]),
                                                numpy.array([r["copy_W"] for r in allr]), rng),
            "arm_means": {a: {k: round(float(numpy.mean([r[k] for r in rec[a]])), 4) for k in rec[a][0]}
                          for a in RUNS},
        },
        "per_run": per_run,
        "note": "t intervals over seeds 1-3 (df 2); rescue = masked-teacher acc on full-teacher-wrong images (W), "
                "harm = masked-teacher error on full-teacher-right images (R); permutation entry [rho, p_lower, p_upper]"})
