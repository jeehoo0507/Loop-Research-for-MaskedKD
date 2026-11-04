"""H-A65 (round 340, analysis only, no training): on cub_lt (gate 2), does the per-image v1 - maskedkd gap in
masked-teacher target quality at the e25 snapshot predict the per-image v1 - maskedkd gap in final student
accuracy?

H-A64: the masked-teacher target gap (tail t_acc +0.110) is already present at e25 and does not grow, while the
student tail accuracy gap grows from +0.029 (e25) to +0.067 (final). This asks whether, image by image, the
images where v1's targets were better at e25 are the images where the v1 student ends up better.

Students (pinned run ids, the same 6 as H-A64), cub_lt final seeds 1, 2, 3, K = 29:
  maskedkd       student last-block head-mean CLS attention top-K
  trustmask_v1   student late-block (6..11) rollout top-K (the teacher map only weights the KD term)
Per image, per epoch (e25, final):
  t_acc   masked teacher (keep-only) top-1 == label
  pt      masked teacher p(label)
  s_acc   student top-1 == label
Per image the 3 seeds are averaged; gaps are v1 - maskedkd on the same image. Within each group:
  Pearson r(e25 dpt, final ds_acc) with an image bootstrap 95% CI (2000 draws, numpy default_rng 20261110);
  mean final ds_acc of images with e25 dt_acc > 0 minus that of images with e25 dt_acc < 0, bootstrap CI.
No labels, groups or positions enter any mask; groups are used only to slice the results.
"""
import numpy
import torch

META = {
    "question": "on cub_lt K=29, does the per-image v1 - maskedkd masked-teacher target gap at e25 predict the "
                "per-image v1 - maskedkd final student accuracy gap?",
    "method": "attention rollout (student blocks 6..11); last-block CLS attention; keep-only masked-teacher "
              "p(label) at epoch snapshots; per-image gap correlation sliced by head/tail class; image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 "
             "(keep-only); Wah et al. 2011 (CUB-200-2011)",
}

START = 6
K = 29
NBOOT = 2000
BATCH = 50
EPOCHS = (25, "final")
ARMS = {
    "maskedkd": {1: "cub_lt_final_maskedkd_k29_s1_1103_234609_76d1",
                 2: "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 3: "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"},
    "trustmask_v1": {1: "cub_lt_final_trustmask_v1_k29_s1_1103_234609_10df",
                     2: "cub_lt_final_trustmask_v1_k29_s2_1103_234647_4e79",
                     3: "cub_lt_final_trustmask_v1_k29_s3_1103_234704_d748"},
}
GROUPS = (0, 1)
TAIL = 1
STATS = ("t_acc", "pt", "s_acc")


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


def _r(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = numpy.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _ci_r(a, b, gen):
    n = len(a)
    bs = []
    for _ in range(NBOOT):
        i = gen.integers(0, n, n)
        bs.append(_r(a[i], b[i]))
    bs = numpy.array(bs)
    return [round(_r(a, b), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4), int(n)]


def _ci_split(cond, v, gen):
    pos, neg = v[cond > 0], v[cond < 0]
    if len(pos) == 0 or len(neg) == 0:
        return [None, None, None, int(len(pos)), int(len(neg))]
    bs = numpy.array([pos[gen.integers(0, len(pos), len(pos))].mean() -
                      neg[gen.integers(0, len(neg), len(neg))].mean() for _ in range(NBOOT)])
    return [round(float(pos.mean() - neg.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4), int(len(pos)), int(len(neg))]


def _mask(api, method, S, x):
    depth = len(S.blocks)
    with torch.no_grad():
        so = api.student_forward(S, x, attn_layers=list(range(depth)))
    if method == "maskedkd":
        return _topk(so.attn[depth - 1].float().mean(1)[:, 0, 1:], K), so.logits
    return _topk(_rollout([so.attn[i] for i in range(START, depth)]), K), so.logits


def run(api):
    runs = api.runs()
    used = {a: {str(s): [rid, runs[rid]["method"], runs[rid]["keep"], runs[rid]["seed"], runs[rid]["tier"],
                         runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]]
                for s, rid in sorted(ids.items())} for a, ids in ARMS.items()}
    api.write_json("students", used)
    T = api.teacher.eval()

    batches = [(x, y) for x, y in api.val_batches(batch=BATCH)]
    groups = api.val_groups().numpy()

    per = {e: {a: {s: [] for s in STATS} for a in ARMS} for e in EPOCHS}
    for e in EPOCHS:
        ep = None if e == "final" else e
        for a, ids in ARMS.items():
            for seed, rid in sorted(ids.items()):
                S = api.student(rid, epoch=ep)
                acc = {s: [] for s in STATS}
                for x, y in batches:
                    keep, slog = _mask(api, a, S, x)
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    acc["t_acc"].append((lm.argmax(-1) == y).float().cpu())
                    acc["pt"].append(lm.gather(1, y.view(-1, 1)).squeeze(1).exp().float().cpu())
                    acc["s_acc"].append((slog.float().argmax(-1) == y).float().cpu())
                for s in STATS:
                    per[e][a][s].append(torch.cat(acc[s]).numpy())

    gen = numpy.random.default_rng(20261110)
    d = {e: {a: {s: numpy.mean(numpy.stack(per[e][a][s]), 0) for s in STATS} for a in ARMS} for e in EPOCHS}
    gap = {e: {s: d[e]["trustmask_v1"][s] - d[e]["maskedkd"][s] for s in STATS} for e in EPOCHS}
    dpt25, dt25 = gap[25]["pt"], gap[25]["t_acc"]
    ds_fin, ds25 = gap["final"]["s_acc"], gap[25]["s_acc"]
    out = {"n_val": int(len(groups)), "group_n": {str(g): int((groups == g).sum()) for g in GROUPS},
           "students": used}
    out["mean_gap"] = {str(e): {s: {str(g): round(float(gap[e][s][groups == g].mean()), 4) for g in GROUPS}
                                for s in STATS} for e in EPOCHS}
    out["r_e25_dpt_vs_final_ds"] = {str(g): _ci_r(dpt25[groups == g], ds_fin[groups == g], gen) for g in GROUPS}
    out["split_e25_dtacc_final_ds"] = {str(g): _ci_split(dt25[groups == g], ds_fin[groups == g], gen)
                                       for g in GROUPS}
    out["r_e25_dpt_vs_ds_growth"] = {str(g): _ci_r(dpt25[groups == g], (ds_fin - ds25)[groups == g], gen)
                                     for g in GROUPS}
    out["r_e25_dpt_vs_e25_ds"] = {str(g): _ci_r(dpt25[groups == g], ds25[groups == g], gen) for g in GROUPS}
    out["r_final_dpt_vs_final_ds"] = {str(g): _ci_r(gap["final"]["pt"][groups == g], ds_fin[groups == g], gen)
                                      for g in GROUPS}

    p1 = out["r_e25_dpt_vs_final_ds"][str(TAIL)]
    p2 = out["split_e25_dtacc_final_ds"][str(TAIL)]
    out["verdict"] = {
        "P1_tail_r_e25_dpt_vs_final_ds": p1,
        "P1_holds_ci_low_gt0": bool(p1[1] > 0),
        "P2_tail_split_e25_dtacc_final_ds": p2,
        "P2_holds_ci_low_gt0": bool(p2[1] is not None and p2[1] > 0),
    }
    api.write_json("summary", out)
