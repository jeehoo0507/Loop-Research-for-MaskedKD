"""H-A75 (round 364, analysis only, no training): on cub_lt K=29 (gate 2), does the masked teacher's soft-label mass
for a tail class land on that class's images (column purity), and does that rank the student's head-row alignment?

H-A74 found the method gap in tail accuracy lives in the direction of the tail-class head rows: Spearman(align_tail,
orig tail) 0.90, v1 - maskedkd align_tail +0.020 [+0.014, +0.025]. The KD part of the gradient on head row W_c is
-sum_i q_c(i) f_i (q = masked-teacher soft label, TAU = 1), so W_c is pulled toward the features of the images on
which the masked teacher puts class-c mass. H-A62/H-A63 looked at rows (per image: is the masked teacher right);
this looks at columns (per class: where does the masked teacher's class-c mass go).

Students (pinned run ids, final weights), cub_lt final seeds 1, 2, 3, K = 29, token rules as in H-A62/H-A71:
  maskedkd        student last-block head-mean CLS attention top-K
  rollout         student rollout over all blocks top-K
  teacher_topk    teacher map top-K (teacher rollout over all blocks = the cached map's rule)
  trustmask_v1    student late-block (6..11) rollout top-K (its KD weight acts only on the loss, left out here)
  trustfilt (v0)  late-block rollout top-K inside the teacher map's top ceil(1.5 K)
  lateroll        late-block rollout top-K
  trustfilt_attn  last-block CLS attention top-K inside the teacher map's top ceil(1.5 K)
Reference: full@196 seeds 1-3 (unmasked teacher; q is the same for every seed, the student features differ).
Un-augmented val (600 images). Per run, q = softmax of the keep-only masked teacher logits.
  purity_c = sum_{i: y_i = c} q_c(i) / sum_i q_c(i);   purity_tail / purity_head = mean over classes 100-199 / 0-99
  P_c = sum_i q_c(i) f_i / sum_i q_c(i) (f = the run's own pre-logit val feature, mu = val mean feature,
        mu_c = val class-mean feature, labels only)
  palign_c = cos(P_c - mu, mu_c - mu);  align_c = cos(W_c, mu_c - mu) (H-A74);  prow_c = cos(P_c - mu, W_c)
Spearman rho over the 21 K=29 runs, run bootstrap 95% CI (2000 draws, numpy default_rng 20261115). Paired same-seed
t 95% CI (df 2): v1 - maskedkd, v0 - maskedkd, lateroll - maskedkd, teacher_topk - maskedkd. No labels, groups or
positions enter any mask; labels enter only the statistics.
"""
import math

import numpy
import torch

META = {
    "question": "on cub_lt K=29, does the masked teacher put its tail-class soft-label mass on that class's images "
                "(column purity), and does that purity rank the alignment of the student's tail head rows (H-A74)?",
    "method": "attention rollout (teacher all blocks, student all / blocks 6..11); last-block CLS attention; keep-only "
              "masked teacher; per-class soft-label column purity and soft-label-weighted feature prototype; "
              "Spearman with run bootstrap; paired t CI",
    "paper": "Hinton et al. 2015 (KD gradient on logits); Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 "
             "(MaskedKD); Kang et al. ICLR 2020 (classifier direction under long tail); Wah et al. 2011 (CUB)",
}

K = 29
START = 6
WIDEN = 1.5
NBOOT = 2000
BATCH = 50
TAIL0 = 100
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
KEYS = ("purity_tail", "purity_head", "palign_tail", "palign_head", "align_tail", "align_head", "prow_tail",
        "t_acc_tail")


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


def _feats(M, xs):
    fs = []
    for x in xs:
        with torch.no_grad():
            h = M.patch_embed(x)
            h = M._pos_embed(h)
            h = M.patch_drop(h)
            h = M.norm_pre(h)
            h = M.blocks(h)
            h = M.norm(h)
            fs.append(M.forward_head(h, pre_logits=True).float())
    return torch.cat(fs)


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


def _stats(Q, F, W, Y, y, tail_c):
    cos = torch.nn.functional.cosine_similarity
    mass = Q.sum(0).clamp(min=1e-12)
    purity = (Q * Y).sum(0) / mass
    mu = F.mean(0, keepdim=True)
    mu_c = (Y.T @ F) / Y.sum(0).clamp(min=1).unsqueeze(1)
    P = (Q.T @ F) / mass.unsqueeze(1)
    palign = cos(P - mu, mu_c - mu, dim=1)
    align = cos(W, mu_c - mu, dim=1)
    prow = cos(P - mu, W, dim=1)
    tail_i = y >= TAIL0
    return {"purity_tail": purity[tail_c].mean().item(), "purity_head": purity[~tail_c].mean().item(),
            "palign_tail": palign[tail_c].mean().item(), "palign_head": palign[~tail_c].mean().item(),
            "align_tail": align[tail_c].mean().item(), "align_head": align[~tail_c].mean().item(),
            "prow_tail": prow[tail_c].mean().item(),
            "t_acc_tail": (Q.argmax(1) == y).float()[tail_i].mean().item()}


def run(api):
    runs = api.runs()
    full_ids = {r["seed"]: rid for rid, r in sorted(runs.items())
                if r.get("dataset") == "cub_lt" and r["tier"] == "final" and r["method"] == "full"
                and r.get("keep") == 196 and r["seed"] in (1, 2, 3)}
    used = {a: {str(s): [rid, runs[rid]["method"], runs[rid]["keep"], runs[rid]["seed"], runs[rid]["tier"],
                         runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]] for s, rid in sorted(ids.items())}
            for a, ids in list(ARMS.items()) + [("full", full_ids)]}
    api.write_json("students", used)
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        batches.append((x, y, _rollout(out.attn), out.logits.float().softmax(-1)))
    xs = [b[0] for b in batches]
    y = torch.cat([b[1] for b in batches])
    ncls = int(y.max().item()) + 1
    Y = torch.nn.functional.one_hot(y, ncls).float()
    tail_c = torch.arange(ncls, device=y.device) >= TAIL0
    Q_full = torch.cat([b[3] for b in batches]).to(y.device)

    rows = []
    for a, ids in list(ARMS.items()) + [("full", full_ids)]:
        for seed, rid in sorted(ids.items()):
            S = api.student(rid)
            if a == "full":
                Q = Q_full
            else:
                qs = []
                for x, _, tm, _ in batches:
                    keep = _mask(api, a, S, x, tm)
                    with torch.no_grad():
                        qs.append(api.teacher_forward_keepmask(T, x, keep).float().softmax(-1))
                Q = torch.cat(qs).to(y.device)
            F = _feats(S, xs).to(y.device)
            W = S.head.weight.detach().float().to(y.device)
            st = _stats(Q, F, W, Y, y, tail_c)
            row = {"arm": a, "seed": seed, "run": rid, "val_tail": runs[rid]["val_wga_last"],
                   "val_acc": runs[rid]["val_acc_last"]}
            row.update({k: round(v, 4) for k, v in st.items()})
            rows.append(row)

    gen = numpy.random.default_rng(20261115)
    k29 = [r for r in rows if r["arm"] in ARMS]
    col = lambda rs, k: numpy.array([r[k] for r in rs], dtype=float)
    rho = {"purity_tail_vs_align_tail_k29": _rho_ci(col(k29, "purity_tail"), col(k29, "align_tail"), gen),
           "purity_tail_vs_val_tail_k29": _rho_ci(col(k29, "purity_tail"), col(k29, "val_tail"), gen),
           "palign_tail_vs_align_tail_k29": _rho_ci(col(k29, "palign_tail"), col(k29, "align_tail"), gen),
           "align_tail_vs_val_tail_k29": _rho_ci(col(k29, "align_tail"), col(k29, "val_tail"), gen),
           "t_acc_tail_vs_align_tail_k29": _rho_ci(col(k29, "t_acc_tail"), col(k29, "align_tail"), gen)}
    by_arm = {}
    for a in list(ARMS) + ["full"]:
        rs = [r for r in rows if r["arm"] == a]
        if rs:
            by_arm[a] = {k: round(float(numpy.mean([r[k] for r in rs])), 4) for k in KEYS + ("val_tail",)}
            by_arm[a]["n_seeds"] = len(rs)
    pair = {}
    for p, q in (("trustmask_v1", "maskedkd"), ("trustfilt", "maskedkd"), ("lateroll", "maskedkd"),
                 ("teacher_topk", "maskedkd"), ("full", "maskedkd")):
        for k in ("purity_tail", "palign_tail", "align_tail", "prow_tail"):
            d = [next(r[k] for r in rows if r["arm"] == p and r["seed"] == s)
                 - next(r[k] for r in rows if r["arm"] == q and r["seed"] == s)
                 for s in (1, 2, 3)
                 if any(r["arm"] == p and r["seed"] == s for r in rows)
                 and any(r["arm"] == q and r["seed"] == s for r in rows)]
            if len(d) >= 2:
                mu, se = float(numpy.mean(d)), float(numpy.std(d, ddof=1)) / len(d) ** 0.5
                tq = {2: 12.706, 3: 4.303}[len(d)]
                pair[f"{p}-{q}:{k}"] = [round(mu, 4), round(mu - tq * se, 4), round(mu + tq * se, 4), len(d)]
    v1 = pair.get("trustmask_v1-maskedkd:purity_tail")
    out = {"n_val": int(len(y)), "n_tail_val": int((y >= TAIL0).sum().item()), "n_runs": len(rows),
           "rho": rho, "paired_t95": pair, "by_arm": by_arm, "runs": rows,
           "verdict": {"P1_rho_purity_tail_vs_align_tail_ge_0.5":
                       bool(rho["purity_tail_vs_align_tail_k29"][0] >= 0.5),
                       "P2_v1_minus_maskedkd_purity_tail_ci_low_gt0": bool(v1 is not None and v1[1] > 0)}}
    api.write_json("summary", out)
