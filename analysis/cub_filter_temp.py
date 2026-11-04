"""H-A159 (round 423, analysis only, no training): H-A158 left one arm out of order on CUB K=29: trustfilt's masked
teacher is far less faithful than lateroll's (KL 1.150 vs 0.624, near maskedkd 1.225), yet its run accuracy (80.89) is
above rollout (0.740 / 80.44). trustfilt = lateroll restricted to the teacher top-ceil(1.5K) area, so the filter is the
only rule difference. Is the filter's extra KL mostly a temperature (confidence) shift of the masked teacher, which a
single scale removes, or a shape loss that stays after calibration?

Runs: CUB final K=29 seeds 1-3, arms lateroll, trustfilt, rollout, maskedkd (12 runs, ids from H-A158).
Per run (each run's own rule, own final student, clean val, teacher map = teacher all-block rollout = cache rule):
raw KL(full || masked) at T=1; tau* = argmin over a grid of mean KL(full || softmax(masked_logits / tau)); KL_cal at tau*;
masked/full teacher entropy; for trustfilt students also the fraction of lateroll top-K tokens the filter removes.
P1: trustfilt - lateroll KL_cal, seed-paired t 95% CI (df 2); image bootstrap (numpy default_rng) as record.
Limit: clean val views, final students; correlation only; no selector is built from it.
"""
import math

import numpy
import torch

META = {
    "question": "on CUB final K=29 runs, is the teacher-area filter's extra masked-teacher KL (trustfilt vs lateroll) "
                "mostly a temperature shift that one scale removes, or a shape loss that remains after calibration?",
    "method": "keep-only masked-teacher faithfulness with per-run temperature scaling (grid search of one scale), "
              "seed-paired t CI and image bootstrap",
    "paper": "Guo et al. ICML 2017 (temperature scaling); Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & "
             "Zuidema ACL 2020 (attention rollout); Son et al. ECCV 2024 (MaskedKD); Hinton et al. 2015 (KD)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261167
K = 29
WIDEN = 1.5
TAUS = numpy.round(numpy.arange(0.5, 4.0001, 0.05), 2)
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


def _ent(lp):
    return -(numpy.exp(lp) * lp).sum(1)


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
    h_full = float(_ent(tlp).mean())

    per_run, img = {}, {}
    for arm, ids in RUNS.items():
        for rid in ids:
            S = api.student(rid)
            gl, removed = [], []
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
                        if arm == "trustfilt":
                            keep = _filt(lr, tm, K)
                            plain = _topk_mask(lr, K)
                            removed.append(((plain & ~_allowed(tm, K)).sum(1).float() / K).cpu())
                        else:
                            keep = _topk_mask(lr, K)
                    gl.append(api.teacher_forward_keepmask(Tm, x, keep).float().cpu())
            del S
            torch.cuda.empty_cache()
            L = torch.cat(gl).numpy().astype(numpy.float64)
            glp = _logp(L)
            kl = _kl(tlp, glp)
            kl_by_tau = numpy.array([_kl(tlp, _logp(L / t)).mean() for t in TAUS])
            j = int(kl_by_tau.argmin())
            tau = float(TAUS[j])
            kl_cal = _kl(tlp, _logp(L / tau))
            img[rid] = (kl, kl_cal)
            per_run[rid] = {"arm": arm, "seed": runs[rid]["seed"], "val_acc_last": float(runs[rid]["val_acc_last"]),
                            "kl": round(float(kl.mean()), 4), "kl_cal": round(float(kl_cal.mean()), 4),
                            "tau": tau, "tau_at_grid_edge": j in (0, len(TAUS) - 1),
                            "h_masked_minus_full": round(float(_ent(glp).mean()) - h_full, 4),
                            "t_acc": round(float((glp.argmax(1) == y).mean() * 100), 2)}
            if removed:
                per_run[rid]["filter_removed_frac"] = round(float(torch.cat(removed).mean()), 4)

    def paired(a, b, key):
        return _tcrit([per_run[x][key] - per_run[z][key] for x, z in zip(RUNS[a], RUNS[b])])

    rng = numpy.random.default_rng(SEED)

    def boot(a, b, which):
        d = numpy.mean([img[x][which] - img[z][which] for x, z in zip(RUNS[a], RUNS[b])], axis=0)
        bs = numpy.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        return [round(float(d.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
                round(float(numpy.quantile(bs, 0.975)), 4)]

    P1 = paired("trustfilt", "lateroll", "kl_cal")
    raw = paired("trustfilt", "lateroll", "kl")
    ids = [rid for arm in RUNS for rid in RUNS[arm]]
    acc = numpy.array([per_run[r]["val_acc_last"] for r in ids])

    def verdict(c):
        return "positive" if c[1] > 0 else ("negative" if c[2] < 0 else "uncertain")

    api.write_json("summary", {
        "n_val": int(n), "K": K, "n_runs": len(ids), "h_full": round(h_full, 4),
        "P1_trustfilt_minus_lateroll_kl_cal_seed_t": P1,
        "record": {
            "trustfilt_minus_lateroll_kl_raw_seed_t": raw,
            "share_of_raw_gap_left_after_cal": round(P1[0] / raw[0], 4) if raw[0] != 0 else None,
            "trustfilt_minus_lateroll_kl_cal_boot": boot("trustfilt", "lateroll", 1),
            "trustfilt_minus_lateroll_kl_raw_boot": boot("trustfilt", "lateroll", 0),
            "trustfilt_minus_rollout_kl_cal_seed_t": paired("trustfilt", "rollout", "kl_cal"),
            "maskedkd_minus_trustfilt_kl_cal_seed_t": paired("maskedkd", "trustfilt", "kl_cal"),
            "trustfilt_minus_lateroll_tau_seed_t": paired("trustfilt", "lateroll", "tau"),
            "trustfilt_minus_lateroll_entropy_seed_t": paired("trustfilt", "lateroll", "h_masked_minus_full"),
            "spearman_kl_raw_acc_12runs": round(_spearman(numpy.array([per_run[r]["kl"] for r in ids]), acc), 4),
            "spearman_kl_cal_acc_12runs": round(_spearman(numpy.array([per_run[r]["kl_cal"] for r in ids]), acc), 4),
            "arm_means_kl_klcal_tau_dH_acc": {a: [round(float(numpy.mean([per_run[r][k] for r in RUNS[a]])), 4)
                                                  for k in ("kl", "kl_cal", "tau", "h_masked_minus_full",
                                                            "val_acc_last")] for a in RUNS},
        },
        "per_run": per_run,
        "verdict": {"P1": verdict(P1)},
        "note": "CI entries are [mean, low, high]; seed_t pairs run s_i of one arm with s_i of the other"})
