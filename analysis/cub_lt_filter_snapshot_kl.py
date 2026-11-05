"""H-A175 (round 439, analysis only, no training): timing version of the open H-A171~174 mismatch.
On cub_lt final K=29 the trustfilt arm has the same masked-teacher KL as maskedkd at the final student (clean mean,
median, crop+flip views and TCKD/wNCKD split all uncertain: H-A171~174) yet +2.3 %p val accuracy. The KD signal a
student received is the one its own rule gave during training, not at the end. Was trustfilt's masked teacher closer to
the full teacher than maskedkd's while the student was still learning (e25, e50 snapshots)?

Runs: trustfilt, maskedkd, lateroll cub_lt final K=29 seeds 1-3 (pinned ids of H-A171), snapshots e25, e50, final.
Per run and snapshot (own rule applied to that snapshot's student, clean val, teacher map = teacher all-block rollout =
cache rule): mean KL(full teacher || masked teacher) T=1, TCKD, wNCKD, tail-class mean KL, student val accuracy.
P1: trustfilt - maskedkd KL at e25; P2: same at e50; seed-paired t 95% CI (df 2).
Record: final (must equal H-A171: check), tail splits, TCKD/wNCKD at each epoch, trustfilt - lateroll, epoch-mean KL
(e25, e50, final mean) difference, image bootstrap of P1/P2 (numpy default_rng 20261188, 2000).
Limit: clean val views; three snapshots only; correlation only; no selector is built from it.
"""
import math

import numpy
import torch

META = {
    "question": "on cub_lt K=29, was trustfilt's own-rule masked teacher closer to the full teacher than maskedkd's at "
                "the e25 and e50 snapshots, although the two are equal at the final student?",
    "method": "own rule on e25/e50/final student snapshots, keep-only masked-teacher KL with exact DKD split, "
              "seed-paired t CI, image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Zhao et al. CVPR 2022 (decoupled KD); Hooker et al. NeurIPS 2019 "
             "(keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); Hinton et al. 2015 (KD); "
             "Wah et al. 2011 (CUB-200-2011)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261188
K = 29
WIDEN = 1.5
TAIL = 1
EPS = 1e-12
T_CRIT = 4.302653  # t(0.975, df 2)
EPOCHS = [25, 50, "final"]
RUNS = {
    "maskedkd": ["cub_lt_final_maskedkd_k29_s1_1103_234609_76d1", "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"],
    "lateroll": ["cub_lt_final_lateroll_k29_s1_1104_084157_2b15", "cub_lt_final_lateroll_k29_s2_1104_084204_9f09",
                 "cub_lt_final_lateroll_k29_s3_1104_084209_a3f8"],
    "trustfilt": ["cub_lt_final_trustfilt_k29_s1_1103_234609_dbe3", "cub_lt_final_trustfilt_k29_s2_1103_234647_43e9",
                  "cub_lt_final_trustfilt_k29_s3_1103_234704_8732"],
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
    return tckd, (1 - pt_c) * nckd


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


def _keep_and_logits(api, arm, S, x, tm):
    """Own-rule keep mask and the student's logits from one fused pass (no attn drift on logits)."""
    if arm == "maskedkd":
        out = api.student_forward(S, x, attn_layers=[len(S.blocks) - 1])
        return _topk_mask(out.attn[-1].float().mean(1)[:, 0, 1:], K), out.logits
    out = api.student_forward(S, x, attn_layers=list(range(6, len(S.blocks))))
    lr = _rollout([a for a in out.attn if a is not None])
    return (_filt(lr, tm, K) if arm == "trustfilt" else _topk_mask(lr, K)), out.logits


def _tcrit(d):
    d = numpy.asarray(d, dtype=float)
    h = T_CRIT * d.std(ddof=1) / math.sqrt(len(d))
    return [round(float(d.mean()), 4), round(float(d.mean() - h), 4), round(float(d.mean() + h), 4),
            [round(float(v), 4) for v in d]]


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid].get("val_wga_last"),
                                      runs[rid].get("snapshots")]
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

    per, img = {}, {}
    for arm, ids in RUNS.items():
        for rid in ids:
            for ep in EPOCHS:
                S = api.student(rid, epoch=None if ep == "final" else ep)
                gl, sl = [], []
                for (x, _), tm in zip(batches, tmap):
                    with torch.no_grad():
                        keep, slog = _keep_and_logits(api, arm, S, x, tm)
                        sl.append(slog.float().cpu())
                        gl.append(api.teacher_forward_keepmask(Tm, x, keep).float().cpu())
                del S
                torch.cuda.empty_cache()
                glp = _logp(torch.cat(gl).numpy().astype(numpy.float64))
                s_acc = float((torch.cat(sl).argmax(1).numpy() == y).mean() * 100)
                kl = _kl(tlp, glp)
                tckd, wnckd = _dkd(tlp, glp, y)
                key = f"{rid}|{ep}"
                img[key] = kl
                per[key] = {"arm": arm, "seed": runs[rid]["seed"], "epoch": ep,
                            "kl": round(float(kl.mean()), 4), "tckd": round(float(tckd.mean()), 4),
                            "wnckd": round(float(wnckd.mean()), 4), "tail_kl": round(float(kl[tail].mean()), 4),
                            "s_acc": round(s_acc, 2)}

    def paired(a, b, k, ep):
        return _tcrit([per[f"{x}|{ep}"][k] - per[f"{z}|{ep}"][k] for x, z in zip(RUNS[a], RUNS[b])])

    def paired_epmean(a, b, k):
        return _tcrit([numpy.mean([per[f"{x}|{e}"][k] - per[f"{z}|{e}"][k] for e in EPOCHS])
                       for x, z in zip(RUNS[a], RUNS[b])])

    rng = numpy.random.default_rng(SEED)

    def boot(a, b, ep):
        d = numpy.mean([img[f"{x}|{ep}"] - img[f"{z}|{ep}"] for x, z in zip(RUNS[a], RUNS[b])], axis=0)
        bs = numpy.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        return [round(float(d.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
                round(float(numpy.quantile(bs, 0.975)), 4)]

    P1 = paired("trustfilt", "maskedkd", "kl", 25)
    P2 = paired("trustfilt", "maskedkd", "kl", 50)

    def verdict(c):
        return "positive" if c[1] > 0 else ("negative" if c[2] < 0 else "uncertain")

    record = {"P1_boot": boot("trustfilt", "maskedkd", 25), "P2_boot": boot("trustfilt", "maskedkd", 50),
              "final_trustfilt_minus_maskedkd_kl_seed_t": paired("trustfilt", "maskedkd", "kl", "final"),
              "epoch_mean_trustfilt_minus_maskedkd_kl_seed_t": paired_epmean("trustfilt", "maskedkd", "kl"),
              "epoch_mean_trustfilt_minus_maskedkd_tail_kl_seed_t": paired_epmean("trustfilt", "maskedkd", "tail_kl")}
    for ep in EPOCHS:
        for k in ("tail_kl", "tckd", "wnckd", "s_acc"):
            record[f"e{ep}_trustfilt_minus_maskedkd_{k}_seed_t"] = paired("trustfilt", "maskedkd", k, ep)
        for k in ("kl", "tail_kl"):
            record[f"e{ep}_trustfilt_minus_lateroll_{k}_seed_t"] = paired("trustfilt", "lateroll", k, ep)
            record[f"e{ep}_lateroll_minus_maskedkd_{k}_seed_t"] = paired("lateroll", "maskedkd", k, ep)
    record["arm_epoch_means_kl_tailkl_tckd_wnckd_sacc"] = {
        f"{a}|{ep}": [round(float(numpy.mean([per[f"{r}|{ep}"][k] for r in RUNS[a]])), 4)
                      for k in ("kl", "tail_kl", "tckd", "wnckd", "s_acc")]
        for a in RUNS for ep in EPOCHS}

    api.write_json("summary", {
        "n_val": int(n), "n_tail": int(tail.sum()), "K": K, "n_runs": sum(len(v) for v in RUNS.values()),
        "P1_e25_trustfilt_minus_maskedkd_kl_seed_t": P1,
        "P2_e50_trustfilt_minus_maskedkd_kl_seed_t": P2,
        "verdict": {"P1": verdict(P1), "P2": verdict(P2)},
        "record": record,
        "per_run_epoch": per,
        "note": "CI entries are [mean, low, high(, per-seed diffs s1..s3)]; seed_t pairs run s_i of one arm with s_i of "
                "the other at the same snapshot; final kl should equal H-A171 (trustfilt 1.152, maskedkd 1.157 arm "
                "means); one rng: P1_boot then P2_boot"})
