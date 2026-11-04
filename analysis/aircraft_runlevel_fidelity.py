"""H-A157 (round 421, analysis only, no training): on Aircraft at the same token budget K=29 (same teacher GFLOPs),
does the masked teacher's fidelity under each run's own token rule, read on that run's own final student, rank the
runs' val accuracy? H-A2 (analysis_aircraft_proxy_validation_1102_174801_94bc) asked this on seed 1 only (K=29 rho
-0.800, 4 arms); H-C4 on CUB screen seed 0. Here: final seeds 1-3, seven K=29 arms, 21 runs, all trained before this
analysis was written.

Arms (rule applied to un-augmented val, teacher map = teacher all-block rollout = cache rule):
maskedkd (last-block CLS attn), rollout (student all-block rollout), lateroll (student blocks 6..11),
trustmask_v1 (lateroll mask; its KD weight is not part of the mask and is not modelled), trustfilt (lateroll inside the
teacher top-ceil(1.5K)), teacher_topk (teacher top-K; same mask for every seed), random (numpy default_rng scores).
Per run: KL(full teacher || masked teacher) at T=1 mean over val (KL), masked-teacher accuracy (t_acc), non-target KL.
P1: Spearman(KL_run, val_acc_last) over 21 runs; one-sided permutation p (10000, numpy default_rng(20261165)).
P2: Spearman over the 7 arm means (seed mean of KL and acc), same permutation scheme.
Limit: clean val views, final students; correlation only; no selector is built from it.
"""
import math

import numpy
import torch

META = {
    "question": "on Aircraft final K=29 runs (seven token rules, seeds 1-3), does the masked teacher's fidelity to the "
                "full teacher under each run's own rule and own student rank the runs' val accuracy?",
    "method": "keep-only masked-teacher faithfulness (KL to the full teacher) per token rule, student/teacher attention "
              "rollout and last-block CLS attention; Spearman across runs with permutation test",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); Son et al. "
             "ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD); Maji et al. 2013 (FGVC-Aircraft)",
}

BATCH = 50
N_PERM = 10000
SEED = 20261165
K = 29
WIDEN = 1.5
RUNS = {
    "maskedkd": ["aircraft_final_maskedkd_k29_s1_1102_132227_6542", "aircraft_final_maskedkd_k29_s2_1102_180321_52e2",
                 "aircraft_final_maskedkd_k29_s3_1102_180336_347e"],
    "rollout": ["aircraft_final_rollout_k29_s1_1102_132227_c3d7", "aircraft_final_rollout_k29_s2_1102_180325_20b6",
                "aircraft_final_rollout_k29_s3_1102_180341_8606"],
    "lateroll": ["aircraft_final_lateroll_k29_s1_1103_065756_f0d9", "aircraft_final_lateroll_k29_s2_1103_065801_38e1",
                 "aircraft_final_lateroll_k29_s3_1103_065805_2130"],
    "trustmask_v1": ["aircraft_final_trustmask_v1_k29_s1_1103_083809_92a4",
                     "aircraft_final_trustmask_v1_k29_s2_1103_083818_f341",
                     "aircraft_final_trustmask_v1_k29_s3_1103_083822_5bcd"],
    "trustfilt": ["aircraft_final_trustfilt_k29_s1_1103_074527_357e", "aircraft_final_trustfilt_k29_s2_1103_074535_db76",
                  "aircraft_final_trustfilt_k29_s3_1103_074540_2203"],
    "teacher_topk": ["aircraft_final_teacher_topk_k29_s1_1103_074554_cd8b",
                     "aircraft_final_teacher_topk_k29_s2_1103_074559_69a5",
                     "aircraft_final_teacher_topk_k29_s3_1103_074603_470b"],
    "random": ["aircraft_final_random_k29_s1_1102_132227_7454", "aircraft_final_random_k29_s2_1103_042255_db43",
               "aircraft_final_random_k29_s3_1103_042259_2272"],
}


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


def _nt_logp(logp, y):
    lp = logp.copy()
    lp[numpy.arange(len(y)), y] = -numpy.inf
    m = lp.max(1, keepdims=True)
    return lp - (m + numpy.log(numpy.exp(lp - m).sum(1, keepdims=True)))


def _kl(tp, sp):
    pt = numpy.exp(tp)
    return numpy.where(pt > 0, pt * (tp - numpy.where(numpy.isfinite(sp), sp, 0.0)), 0.0).sum(1)


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
    tnt = _nt_logp(tlp, y)
    rng_sel = numpy.random.default_rng(SEED + 1)

    per_run, rows = {}, []
    for arm, ids in RUNS.items():
        for rid in ids:
            S = None if arm in ("teacher_topk", "random") else api.student(rid)
            gl = []
            for (x, _), tm in zip(batches, tmap):
                with torch.no_grad():
                    if arm == "teacher_topk":
                        keep = _topk_mask(tm, K)
                    elif arm == "random":
                        sc = torch.from_numpy(rng_sel.random((x.shape[0], tm.shape[1]))).to(tm.device)
                        keep = _topk_mask(sc, K)
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
            ntkl = _kl(tnt, _nt_logp(glp, y))
            tacc = (glp.argmax(1) == y).mean() * 100
            acc = float(runs[rid]["val_acc_last"])
            per_run[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": acc,
                            "kl": round(float(kl.mean()), 4), "nt_kl": round(float(ntkl.mean()), 4),
                            "t_acc": round(float(tacc), 2)}
            rows.append((arm, float(kl.mean()), float(ntkl.mean()), float(tacc), acc))

    rng = numpy.random.default_rng(SEED)
    kl_r = numpy.array([r[1] for r in rows])
    nt_r = numpy.array([r[2] for r in rows])
    ta_r = numpy.array([r[3] for r in rows])
    acc_r = numpy.array([r[4] for r in rows])
    arms = list(RUNS)
    am = {a: [numpy.mean([r[i] for r in rows if r[0] == a]) for i in (1, 2, 3, 4)] for a in arms}
    kl_a = numpy.array([am[a][0] for a in arms])
    acc_a = numpy.array([am[a][3] for a in arms])

    P1 = _perm(kl_r, acc_r, rng)
    P2 = _perm(kl_a, acc_a, rng)

    def verdict(p):
        return "negative" if p[1] < 0.05 else ("positive" if p[2] < 0.05 else "uncertain")

    api.write_json("summary", {
        "n_val": int(n), "K": K, "n_runs": len(rows),
        "P1_spearman_kl_acc_runs": P1,
        "P2_spearman_kl_acc_arms": P2,
        "record": {
            "spearman_ntkl_acc_runs": _perm(nt_r, acc_r, rng),
            "spearman_tacc_acc_runs": _perm(ta_r, acc_r, rng),
            "spearman_ntkl_acc_arms": _perm(numpy.array([am[a][1] for a in arms]), acc_a, rng),
            "spearman_tacc_acc_arms": _perm(numpy.array([am[a][2] for a in arms]), acc_a, rng),
            "spearman_kl_acc_runs_no_random": _perm(numpy.array([r[1] for r in rows if r[0] != "random"]),
                                                    numpy.array([r[4] for r in rows if r[0] != "random"]), rng),
            "arm_means_kl_ntkl_tacc_acc": {a: [round(float(v), 4) for v in am[a]] for a in arms},
        },
        "per_run": per_run,
        "verdict": {"P1": verdict(P1), "P2": verdict(P2)},
        "note": "permutation entries are [rho, p_lower (rho <= observed), p_upper (rho >= observed)]"})
