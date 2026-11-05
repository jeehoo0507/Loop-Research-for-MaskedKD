"""H-A172 (round 436, analysis only, no training): follow-up of H-A171. On cub_lt final K=29 the trustfilt arm has
the same mean masked-teacher KL as maskedkd (1.152 vs 1.157) but +2.3 %p val accuracy. Is that because the per-image
KL distribution has a different shape: trustfilt close to the full teacher on the typical image, with a heavy tail of
images where the teacher-region filter breaks the masked teacher? Same 18 runs, same masks as H-A171.
Per run: median, 75th and 90th percentile of per-image KL, 10%-trimmed mean (top 10% dropped), share of images with
KL > 1.
P1: trustfilt - maskedkd median KL, paired by seed, t 95% CI (df 2).
P2: Spearman(median KL_run, val_acc_last) over 18 runs; one-sided permutation p (10000, numpy default_rng(20261185)).

Arms (rule applied to un-augmented val, teacher map = teacher all-block rollout = cache rule):
maskedkd (last-block CLS attn), rollout (student all-block rollout), lateroll (student blocks 6..11),
trustmask_v1 (lateroll mask; its KD weight is not part of the mask and is not modelled), trustfilt (lateroll inside the
teacher top-ceil(1.5K)), teacher_topk (teacher top-K; same mask for every seed).
Record: same pair for q75, q90, trimmed mean, share KL>1, mean (check vs H-A171); trustfilt - lateroll and
trustfilt - maskedkd median on tail images; Spearman of the other shape statistics with accuracy and tail accuracy;
Spearman(median KL, acc) without maskedkd.
Limit: clean val views, final students; correlation only; no selector is built from it.
"""
import math

import numpy
import torch

META = {
    "question": "on cub_lt final K=29 runs, is trustfilt's mean masked-teacher KL (equal to maskedkd's) driven by a "
                "heavy tail of images, with a lower median KL, and does the median KL rank run accuracy?",
    "method": "keep-only masked-teacher faithfulness (KL to the full teacher) per token rule, student/teacher attention "
              "rollout and last-block CLS attention; Spearman across runs with permutation test",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); Son et al. "
             "ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD); Wah et al. 2011 (CUB-200-2011)",
}

BATCH = 50
N_PERM = 10000
SEED = 20261185
T_CRIT = 4.303
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
            acc = float(runs[rid]["val_acc_last"])
            tacc = float(runs[rid]["val_wga_last"])
            srt = numpy.sort(kl)
            kt = kl[tail]
            rec = {"mean": float(kl.mean()), "median": float(numpy.median(kl)),
                   "q75": float(numpy.quantile(kl, 0.75)), "q90": float(numpy.quantile(kl, 0.90)),
                   "trim10": float(srt[:int(round(0.9 * n))].mean()), "share_gt1": float((kl > 1.0).mean()),
                   "median_tail": float(numpy.median(kt))}
            per_run[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": round(acc, 2),
                            "val_tail_last": tacc, **{k: round(v, 4) for k, v in rec.items()}}
            rows.append({"arm": arm, "seed": runs[rid]["seed"], "acc": acc, "tail": tacc, **rec})

    rng = numpy.random.default_rng(SEED)
    arms = list(RUNS)
    stats = ("mean", "median", "q75", "q90", "trim10", "share_gt1", "median_tail")
    am = {a: {k: float(numpy.mean([r[k] for r in rows if r["arm"] == a])) for k in stats + ("acc", "tail")}
          for a in arms}

    def col(key, skip=None):
        return numpy.array([r[key] for r in rows if r["arm"] != skip])

    def paired(a, b, key):
        da = {r["seed"]: r[key] for r in rows if r["arm"] == a}
        db = {r["seed"]: r[key] for r in rows if r["arm"] == b}
        d = numpy.array([da[s] - db[s] for s in sorted(da)])
        h = T_CRIT * d.std(ddof=1) / math.sqrt(len(d))
        return [round(float(d.mean()), 4), round(float(d.mean() - h), 4), round(float(d.mean() + h), 4),
                [round(float(v), 4) for v in d]]

    def ci_verdict(c):
        return "negative" if c[2] < 0 else ("positive" if c[1] > 0 else "uncertain")

    def verdict(p):
        return "negative" if p[1] < 0.05 else ("positive" if p[2] < 0.05 else "uncertain")

    P1 = paired("trustfilt", "maskedkd", "median")
    P2 = _perm(col("median"), col("acc"), rng)
    rec_pairs = {f"trustfilt_minus_maskedkd_{k}": paired("trustfilt", "maskedkd", k)
                 for k in ("mean", "q75", "q90", "trim10", "share_gt1", "median_tail")}
    rec_pairs.update({f"trustfilt_minus_lateroll_{k}": paired("trustfilt", "lateroll", k)
                      for k in ("median", "median_tail", "q90", "share_gt1")})
    rec_sp = {}
    for k in ("mean", "q75", "q90", "trim10", "share_gt1"):
        rec_sp[f"spearman_{k}_acc_runs"] = _perm(col(k), col("acc"), rng)
    rec_sp["spearman_median_tail_runs"] = _perm(col("median"), col("tail"), rng)
    rec_sp["spearman_mediantail_tail_runs"] = _perm(col("median_tail"), col("tail"), rng)
    rec_sp["spearman_median_acc_runs_no_maskedkd"] = _perm(col("median", "maskedkd"), col("acc", "maskedkd"), rng)

    api.write_json("summary", {
        "n_val": int(n), "n_tail": int(tail.sum()), "K": K, "n_runs": len(rows),
        "P1_trustfilt_minus_maskedkd_median_kl": P1,
        "P2_spearman_median_kl_acc_runs": P2,
        "record": {"pairs": rec_pairs, "spearman": rec_sp,
                   "arm_means": {a: {k: round(v, 4) for k, v in am[a].items()} for a in arms}},
        "per_run": per_run,
        "verdict": {"P1": ci_verdict(P1), "P2": verdict(P2)},
        "note": "pairs are [mean diff, t95 low, t95 high (df 2), per-seed diffs s1..s3]; permutation entries are "
                "[rho, p_lower, p_upper]; one rng shared in order P2 then record spearman keys in listed order"})
