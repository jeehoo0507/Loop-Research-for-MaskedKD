"""H-A76 (round 365, analysis only, no training): on cub_lt K=29 (gate 2), do the teacher-region rules keep the full
teacher's off-target (non-true-class) soft-label structure better than the student-side rules, which would explain why
their tail accuracy is not low although their column purity (H-A75) and target accuracy (H-A71) are?

H-A75: purity_tail ranks the student's tail head-row alignment (0.78) but only weakly ranks val tail (0.45), because
teacher_topk / v0 / trustfilt_attn have lower purity than maskedkd while their tail accuracy is higher. The KD
signal has two parts: the true-class probability (what H-A62/H-A75 measure) and the shape of the rest of the
distribution (Hinton 2015 "dark knowledge", the class-similarity structure). Here we measure the second part:
per val image, renormalise the full teacher p and the masked teacher q over the classes != y and take
  off_kl(i) = KL(p_off || q_off)        (lower = masked teacher keeps the full teacher's class relations)
  off_top5(i) = |top5(p_off) & top5(q_off)| / 5
plus  q_y(i) (masked true-class probability) and kl(i) = KL(p || q) (whole distribution).
tail = mean over val images of classes 100..199 (300 images), head = classes 0..99.

Students (pinned run ids, as in H-A75), cub_lt final seeds 1, 2, 3, K = 29, token rules as in H-A62/H-A71/H-A75.
Un-augmented val (600 images). Spearman rho over the 21 K=29 runs, run bootstrap 95% CI (2000 draws, numpy
default_rng 20261116). Paired same-seed t 95% CI (df 2) vs maskedkd. No labels, groups or positions enter any mask;
labels enter only the statistics.
"""
import math

import numpy
import torch

META = {
    "question": "on cub_lt K=29, do teacher-region token rules keep the full teacher's off-target soft-label "
                "structure on tail images better than student-side rules, and does that rank run tail accuracy "
                "better than column purity (H-A75)?",
    "method": "attention rollout (teacher all blocks, student all / blocks 6..11); last-block CLS attention; keep-only "
              "masked teacher; off-target renormalised KL and top-5 overlap to the unmasked teacher; Spearman with "
              "run bootstrap; paired t CI",
    "paper": "Hinton et al. 2015 (dark knowledge); Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 "
             "(MaskedKD); Wah et al. 2011 (CUB)",
}

K = 29
START = 6
WIDEN = 1.5
NBOOT = 2000
BATCH = 50
TAIL0 = 100
EPS = 1e-8
ARMS = {
    "maskedkd": {1: "cub_lt_final_maskedkd_k29_s1_1103_234609_76d1",
                 2: "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 3: "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"},
    "rollout": {1: "cub_lt_final_rollout_k29_s1_1103_234609_8382",
                2: "cub_lt_final_rollout_k29_s2_1103_234647_9b4c",
                3: "cub_lt_final_rollout_k29_s3_1103_234704_0edf"},
    "teacher_topk": {1: "cub_lt_final_teacher_topk_k29_s1_1103_234627_9ef0",
                     2: "cub_lt_final_teacher_topk_k29_s2_1103_234655_cfa0",
                     3: "cub_lt_final_teacher_topk_k29_s3_1103_234716_6d91"},
    "trustmask_v1": {1: "cub_lt_final_trustmask_v1_k29_s1_1103_234609_10df",
                     2: "cub_lt_final_trustmask_v1_k29_s2_1103_234647_4e79",
                     3: "cub_lt_final_trustmask_v1_k29_s3_1103_234704_d748"},
    "trustfilt": {1: "cub_lt_final_trustfilt_k29_s1_1103_234609_dbe3",
                  2: "cub_lt_final_trustfilt_k29_s2_1103_234647_43e9",
                  3: "cub_lt_final_trustfilt_k29_s3_1103_234704_8732"},
    "lateroll": {1: "cub_lt_final_lateroll_k29_s1_1104_084157_2b15",
                 2: "cub_lt_final_lateroll_k29_s2_1104_084204_9f09",
                 3: "cub_lt_final_lateroll_k29_s3_1104_084209_a3f8"},
    "trustfilt_attn": {1: "cub_lt_final_trustfilt_attn_k29_s1_1104_103757_81f3",
                       2: "cub_lt_final_trustfilt_attn_k29_s2_1104_103804_0d6e",
                       3: "cub_lt_final_trustfilt_attn_k29_s3_1104_103809_befb"},
}
KEYS = ("off_kl_tail", "off_kl_head", "off_top5_tail", "off_top5_head", "q_y_tail", "q_y_head", "kl_tail", "kl_head")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    m = r[:, 0, 1:]
    return m / m.sum(1, keepdim=True)


def _mask(api, arm, S, x, tm):
    if arm == "teacher_topk":
        return _topk(tm, K)
    depth = len(S.blocks)
    with torch.no_grad():
        so = api.student_forward(S, x, attn_layers=list(range(depth)))
    last = so.attn[depth - 1].float().mean(1)[:, 0, 1:]
    if arm == "maskedkd":
        return _topk(last, K)
    if arm == "rollout":
        return _topk(_rollout([so.attn[i] for i in range(depth)]), K)
    late = _rollout([so.attn[i] for i in range(START, depth)])
    if arm in ("trustmask_v1", "lateroll"):
        return _topk(late, K)
    allowed = _topk(tm, min(int(math.ceil(WIDEN * K)), tm.shape[1]))
    s = last if arm == "trustfilt_attn" else late
    return _topk(s.masked_fill(~allowed, -1.0), K)


def _rank(v):
    r = numpy.empty(len(v))
    r[numpy.argsort(v, kind="stable")] = numpy.arange(len(v))
    return r


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(numpy.corrcoef(ra, rb)[0, 1])


def _rho_ci(a, b, gen):
    n = len(a)
    bs = []
    for _ in range(NBOOT):
        i = gen.integers(0, n, n)
        r = _spearman(a[i], b[i])
        if r == r:
            bs.append(r)
    bs = numpy.array(bs)
    return [round(_spearman(a, b), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4), int(n)]


def _stats(P, Q, Y, y):
    kl = (P * (P.clamp(min=EPS).log() - Q.clamp(min=EPS).log())).sum(1)
    Po, Qo = P * (1 - Y), Q * (1 - Y)
    Po = Po / Po.sum(1, keepdim=True).clamp(min=EPS)
    Qo = Qo / Qo.sum(1, keepdim=True).clamp(min=EPS)
    off_kl = (Po * (Po.clamp(min=EPS).log() - Qo.clamp(min=EPS).log())).sum(1)
    tp, tq = Po.topk(5, dim=1).indices, Qo.topk(5, dim=1).indices
    off_top5 = (tp.unsqueeze(2) == tq.unsqueeze(1)).any(2).float().sum(1) / 5
    q_y = (Q * Y).sum(1)
    tail_i = y >= TAIL0
    out = {}
    for name, v in (("off_kl", off_kl), ("off_top5", off_top5), ("q_y", q_y), ("kl", kl)):
        out[name + "_tail"] = v[tail_i].mean().item()
        out[name + "_head"] = v[~tail_i].mean().item()
    return out


def run(api):
    runs = api.runs()
    used = {a: {str(s): [rid, runs[rid]["method"], runs[rid]["keep"], runs[rid]["seed"], runs[rid]["tier"],
                         runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]] for s, rid in sorted(ids.items())}
            for a, ids in ARMS.items()}
    api.write_json("students", used)
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        batches.append((x, y, _rollout(out.attn), out.logits.float().softmax(-1)))
    y = torch.cat([b[1] for b in batches])
    ncls = int(y.max().item()) + 1
    Y = torch.nn.functional.one_hot(y, ncls).float()
    P = torch.cat([b[3] for b in batches]).to(y.device)

    rows = []
    for a, ids in ARMS.items():
        for seed, rid in sorted(ids.items()):
            S = api.student(rid)
            qs = []
            for x, _, tm, _ in batches:
                keep = _mask(api, a, S, x, tm)
                with torch.no_grad():
                    qs.append(api.teacher_forward_keepmask(T, x, keep).float().softmax(-1))
            Q = torch.cat(qs).to(y.device)
            st = _stats(P, Q, Y, y)
            row = {"arm": a, "seed": seed, "run": rid, "val_tail": runs[rid]["val_wga_last"],
                   "val_acc": runs[rid]["val_acc_last"]}
            row.update({k: round(v, 4) for k, v in st.items()})
            rows.append(row)

    gen = numpy.random.default_rng(20261116)
    col = lambda rs, k: numpy.array([r[k] for r in rs], dtype=float)
    neg_off = -col(rows, "off_kl_tail")
    rho = {"neg_off_kl_tail_vs_val_tail": _rho_ci(neg_off, col(rows, "val_tail"), gen),
           "off_top5_tail_vs_val_tail": _rho_ci(col(rows, "off_top5_tail"), col(rows, "val_tail"), gen),
           "q_y_tail_vs_val_tail": _rho_ci(col(rows, "q_y_tail"), col(rows, "val_tail"), gen),
           "neg_kl_tail_vs_val_tail": _rho_ci(-col(rows, "kl_tail"), col(rows, "val_tail"), gen),
           "neg_off_kl_tail_vs_q_y_tail": _rho_ci(neg_off, col(rows, "q_y_tail"), gen)}
    by_arm = {}
    for a in ARMS:
        rs = [r for r in rows if r["arm"] == a]
        by_arm[a] = {k: round(float(numpy.mean([r[k] for r in rs])), 4) for k in KEYS + ("val_tail",)}
        by_arm[a]["n_seeds"] = len(rs)
    pair = {}
    for p in ("teacher_topk", "trustfilt", "trustfilt_attn", "trustmask_v1", "lateroll", "rollout"):
        for k in ("off_kl_tail", "off_top5_tail", "q_y_tail", "kl_tail"):
            d = [next(r[k] for r in rows if r["arm"] == p and r["seed"] == s)
                 - next(r[k] for r in rows if r["arm"] == "maskedkd" and r["seed"] == s) for s in (1, 2, 3)]
            mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
            pair[f"{p}-maskedkd:{k}"] = [round(mu, 4), round(mu - 4.303 * se, 4), round(mu + 4.303 * se, 4), len(d)]
    tt = pair["teacher_topk-maskedkd:off_kl_tail"]
    out = {"n_val": int(len(y)), "n_tail_val": int((y >= TAIL0).sum().item()), "n_runs": len(rows),
           "rho": rho, "paired_t95": pair, "by_arm": by_arm, "runs": rows,
           "verdict": {"P1_teacher_topk_minus_maskedkd_off_kl_tail_ci_high_lt0": bool(tt[2] < 0),
                       "P2_rho_neg_off_kl_tail_vs_val_tail_ge_0.5":
                       bool(rho["neg_off_kl_tail_vs_val_tail"][0] >= 0.5)}}
    api.write_json("summary", out)
