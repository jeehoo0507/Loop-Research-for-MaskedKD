"""H-A171 (round 435, analysis only, no training): the cub_lt counterpart of H-A157 (Aircraft), H-A158 (CUB) and
H-A170 (Waterbirds). At the same token budget K=29 (same teacher GFLOPs), does the masked teacher's fidelity under each
run's own token rule, read on that run's own final student, rank the runs' val accuracy (primary metric) and val tail
accuracy? cub_lt final seeds 1-3, the same six K=29 arms as H-A170, 18 runs (all dataset cub_lt, trained before this
analysis was written). random has only seed 1 on cub_lt final and is left out, as in H-A170.
H-A62 asked the tail t_acc vs run tail accuracy question on 15 of these runs (no lateroll); H-A78 asked q_y vs run
accuracy on 21 runs. Here the statistic is the KL used in H-A157/158/170, so the four stages share one table.

Arms (rule applied to un-augmented val, teacher map = teacher all-block rollout = cache rule):
maskedkd (last-block CLS attn), rollout (student all-block rollout), lateroll (student blocks 6..11),
trustmask_v1 (lateroll mask; its KD weight is not part of the mask and is not modelled), trustfilt (lateroll inside the
teacher top-ceil(1.5K)), teacher_topk (teacher top-K; same mask for every seed).
Per run: KL(full teacher || masked teacher) at T=1 mean over val (KL), KL on tail images only (KL_tail, group 1),
masked-teacher accuracy (t_acc), masked-teacher tail accuracy (t_acc_tail).
P1: Spearman(KL_run, val_acc_last) over 18 runs; one-sided permutation p (10000, numpy default_rng(20261171)).
P2: Spearman(KL_run, val_wga_last = tail accuracy) over 18 runs, same scheme.
Record: arm-mean versions, KL_tail vs tail, t_acc / t_acc_tail vs accuracy / tail, maskedkd-excluded P1 and P2.
Limit: clean val views, final students; correlation only; no selector is built from it.
"""
import math

import numpy
import torch

META = {
    "question": "on cub_lt final K=29 runs (six token rules, seeds 1-3), does the masked teacher's fidelity to the "
                "full teacher under each run's own rule and own student rank the runs' val accuracy and tail accuracy?",
    "method": "keep-only masked-teacher faithfulness (KL to the full teacher) per token rule, student/teacher attention "
              "rollout and last-block CLS attention; Spearman across runs with permutation test",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); Son et al. "
             "ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD); Wah et al. 2011 (CUB-200-2011)",
}

BATCH = 50
N_PERM = 10000
SEED = 20261171
K = 29
WIDEN = 1.5
TAIL = 1
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
    tail = groups[:n] == TAIL
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
            tacc = float(runs[rid]["val_wga_last"])
            rec = {"kl": float(kl.mean()), "kl_tail": float(kl[tail].mean()), "t_acc": float(right.mean() * 100),
                   "t_acc_tail": float(right[tail].mean() * 100)}
            per_run[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": round(acc, 2),
                            "val_tail_last": tacc, **{k: round(v, 4) for k, v in rec.items()}}
            rows.append({"arm": arm, "acc": acc, "tail": tacc, **rec})

    rng = numpy.random.default_rng(SEED)
    arms = list(RUNS)
    am = {a: {k: float(numpy.mean([r[k] for r in rows if r["arm"] == a]))
              for k in ("kl", "kl_tail", "t_acc", "t_acc_tail", "acc", "tail")} for a in arms}

    def col(key, skip=None):
        return numpy.array([r[key] for r in rows if r["arm"] != skip])

    def acol(key):
        return numpy.array([am[a][key] for a in arms])

    P1 = _perm(col("kl"), col("acc"), rng)
    P2 = _perm(col("kl"), col("tail"), rng)
    no_mkd = "maskedkd"

    def verdict(p):
        return "negative" if p[1] < 0.05 else ("positive" if p[2] < 0.05 else "uncertain")

    api.write_json("summary", {
        "n_val": int(n), "n_tail": int(tail.sum()), "K": K, "n_runs": len(rows),
        "P1_spearman_kl_acc_runs": P1,
        "P2_spearman_kl_tail_runs": P2,
        "record": {
            "spearman_kl_acc_arms": _perm(acol("kl"), acol("acc"), rng),
            "spearman_kl_tail_arms": _perm(acol("kl"), acol("tail"), rng),
            "spearman_kltail_tail_runs": _perm(col("kl_tail"), col("tail"), rng),
            "spearman_tacc_acc_runs": _perm(col("t_acc"), col("acc"), rng),
            "spearman_tacctail_tail_runs": _perm(col("t_acc_tail"), col("tail"), rng),
            "spearman_kl_acc_runs_no_maskedkd": _perm(col("kl", no_mkd), col("acc", no_mkd), rng),
            "spearman_kl_tail_runs_no_maskedkd": _perm(col("kl", no_mkd), col("tail", no_mkd), rng),
            "arm_means": {a: {k: round(v, 4) for k, v in am[a].items()} for a in arms},
        },
        "per_run": per_run,
        "verdict": {"P1": verdict(P1), "P2": verdict(P2)},
        "note": "permutation entries are [rho, p_lower (rho <= observed), p_upper (rho >= observed)]; "
                "one rng shared in order P1, P2, then record keys in listed order; val_wga_last is tail accuracy"})
