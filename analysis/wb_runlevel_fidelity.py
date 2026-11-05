"""H-A170 (round 434, analysis only, no training): the Waterbirds counterpart of H-A157 (Aircraft) and H-A158 (CUB).
At the same token budget K=29 (same teacher GFLOPs), does the masked teacher's fidelity under each run's own token rule,
read on that run's own final student, rank the runs' val WGA and val accuracy? analysis_proxy_validation (1030) asked
this on Waterbirds screen runs; here: Waterbirds final seeds 1-3, six K=29 arms, 18 runs (all dataset waterbirds,
trained before this analysis was written). random has only seed 1 on Waterbirds final and is left out, unlike
H-A157/158 (so no arm far below the rest drives the rank).

Arms (rule applied to un-augmented val, teacher map = teacher all-block rollout = cache rule):
maskedkd (last-block CLS attn), rollout (student all-block rollout), lateroll (student blocks 6..11),
trustmask_v1 (lateroll mask; its KD weight is not part of the mask and is not modelled), trustfilt (lateroll inside the
teacher top-ceil(1.5K)), teacher_topk (teacher top-K; same mask for every seed).
Per run: KL(full teacher || masked teacher) at T=1 mean over val (KL), KL on val group 2 only (KL_g2),
masked-teacher accuracy (t_acc), masked-teacher group-2 accuracy (t_acc_g2). 2 classes: no non-target KL.
P1: Spearman(KL_run, val_wga_last) over 18 runs; one-sided permutation p (10000, numpy default_rng(20261170)).
P2: Spearman(KL_run, val_acc_last) over 18 runs, same scheme.
Record: arm-mean versions, KL_g2 vs WGA, t_acc / t_acc_g2 vs WGA, maskedkd-excluded P1.
Limit: clean val views, final students; correlation only; no selector is built from it.
"""
import math

import numpy
import torch

META = {
    "question": "on Waterbirds final K=29 runs (six token rules, seeds 1-3), does the masked teacher's fidelity to the "
                "full teacher under each run's own rule and own student rank the runs' val WGA and val accuracy?",
    "method": "keep-only masked-teacher faithfulness (KL to the full teacher) per token rule, student/teacher attention "
              "rollout and last-block CLS attention; Spearman across runs with permutation test",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); Son et al. "
             "ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

BATCH = 50
N_PERM = 10000
SEED = 20261170
K = 29
WIDEN = 1.5
WORST = 2
RUNS = {
    "maskedkd": ["waterbirds_final_maskedkd_k29_s1_1031_201502_4be8", "waterbirds_final_maskedkd_k29_s2_1103_170931_92bd",
                 "waterbirds_final_maskedkd_k29_s3_1103_170935_907a"],
    "rollout": ["waterbirds_final_rollout_k29_s1_1031_201502_e135", "waterbirds_final_rollout_k29_s2_1103_170940_2b72",
                "waterbirds_final_rollout_k29_s3_1103_170944_dd2c"],
    "lateroll": ["waterbirds_final_lateroll_k29_s1_1104_025003_b396", "waterbirds_final_lateroll_k29_s2_1104_025008_c94e",
                 "waterbirds_final_lateroll_k29_s3_1104_025012_629b"],
    "trustmask_v1": ["waterbirds_final_trustmask_v1_k29_s1_1103_170855_2e3c",
                     "waterbirds_final_trustmask_v1_k29_s2_1103_170951_279c",
                     "waterbirds_final_trustmask_v1_k29_s3_1103_170956_fdca"],
    "trustfilt": ["waterbirds_final_trustfilt_k29_s1_1103_171121_24be", "waterbirds_final_trustfilt_k29_s2_1103_171126_b33d",
                  "waterbirds_final_trustfilt_k29_s3_1103_171130_db65"],
    "teacher_topk": ["waterbirds_final_teacher_topk_k29_s1_1103_170914_f015",
                     "waterbirds_final_teacher_topk_k29_s2_1103_170918_4535",
                     "waterbirds_final_teacher_topk_k29_s3_1103_170923_d12a"],
}


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _kl(tp, sp):
    pt = numpy.exp(tp)
    return (pt * (tp - sp)).sum(1)


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

    per_run, rows = {}, []
    for arm, ids in RUNS.items():
        for rid in ids:
            S = None if arm == "teacher_topk" else api.student(rid)
            gl = []
            for (x, _), tm in zip(batches, tmap):
                with torch.no_grad():
                    if arm == "teacher_topk":
                        keep = _topk_mask(tm, K)
                    elif arm == "maskedkd":
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
            if S is not None:
                del S
                torch.cuda.empty_cache()
            glp = _logp(torch.cat(gl).numpy().astype(numpy.float64))
            kl = _kl(tlp, glp)
            right = glp.argmax(1) == y
            acc = float(runs[rid]["val_acc_last"])
            wga = float(runs[rid]["val_wga_last"])
            rec = {"kl": float(kl.mean()), "kl_g2": float(kl[g2].mean()), "t_acc": float(right.mean() * 100),
                   "t_acc_g2": float(right[g2].mean() * 100)}
            per_run[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": round(acc, 2), "val_wga_last": wga,
                            **{k: round(v, 4) for k, v in rec.items()}}
            rows.append({"arm": arm, "acc": acc, "wga": wga, **rec})

    rng = numpy.random.default_rng(SEED)
    arms = list(RUNS)
    am = {a: {k: float(numpy.mean([r[k] for r in rows if r["arm"] == a]))
              for k in ("kl", "kl_g2", "t_acc", "t_acc_g2", "acc", "wga")} for a in arms}

    def col(key, skip=None):
        return numpy.array([r[key] for r in rows if r["arm"] != skip])

    def acol(key):
        return numpy.array([am[a][key] for a in arms])

    P1 = _perm(col("kl"), col("wga"), rng)
    P2 = _perm(col("kl"), col("acc"), rng)
    no_mkd = "maskedkd"

    def verdict(p):
        return "negative" if p[1] < 0.05 else ("positive" if p[2] < 0.05 else "uncertain")

    api.write_json("summary", {
        "n_val": int(n), "n_g2": int(g2.sum()), "K": K, "n_runs": len(rows),
        "P1_spearman_kl_wga_runs": P1,
        "P2_spearman_kl_acc_runs": P2,
        "record": {
            "spearman_kl_wga_arms": _perm(acol("kl"), acol("wga"), rng),
            "spearman_kl_acc_arms": _perm(acol("kl"), acol("acc"), rng),
            "spearman_klg2_wga_runs": _perm(col("kl_g2"), col("wga"), rng),
            "spearman_tacc_wga_runs": _perm(col("t_acc"), col("wga"), rng),
            "spearman_taccg2_wga_runs": _perm(col("t_acc_g2"), col("wga"), rng),
            "spearman_kl_wga_runs_no_maskedkd": _perm(col("kl", no_mkd), col("wga", no_mkd), rng),
            "spearman_kl_acc_runs_no_maskedkd": _perm(col("kl", no_mkd), col("acc", no_mkd), rng),
            "arm_means": {a: {k: round(v, 4) for k, v in am[a].items()} for a in arms},
        },
        "per_run": per_run,
        "verdict": {"P1": verdict(P1), "P2": verdict(P2)},
        "note": "permutation entries are [rho, p_lower (rho <= observed), p_upper (rho >= observed)]; "
                "one rng shared in order P1, P2, then record keys in listed order"})
