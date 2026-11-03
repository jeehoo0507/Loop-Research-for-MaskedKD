"""H-A49 (round 296, analysis only, no training): on Waterbirds K=29, does trustmask_v1's per-image KD weight shift
the class mass of its KD target toward the minority class?

H-A48: the class mass of each run's masked-teacher target ranks the runs like the student head's offset (rho 0.88),
but v1's unweighted target (mean p_T^mask(waterbird) 0.208) is as landbird-leaning as maskedkd's (0.198) while its
WGA is much higher. v1 also weights each image's KD term by the cached teacher-map mass its kept tokens cover
(weight = coverage / mean). If waterbird images get larger weights, the weighted target carries more waterbird mass
than the unweighted one, which would close the gap.

Students: Waterbirds final trustmask_v1 @29, seeds 1, 2, 3 (main); maskedkd @29 seeds 1, 2, 3 (record: the same
coverage weight computed on maskedkd's tokens, which maskedkd does not use).
Mask: each run's own rule on the un-augmented val image (same as H-A48), with the epoch-25 and final student.
Weight: coverage of the teacher rollout map (normalised to sum 1, as the cache) by the kept tokens, divided by the
mean coverage over all val images of that run/epoch (training normalises per batch; mean 1 either way).
Per run and epoch, over all val images:
  qT    mean p_T^mask(waterbird)               (TAU = 1)
  qTw   sum(w * p_T^mask(waterbird)) / sum(w)  (class mass of the weighted KD target)
  r     mean w on waterbird images / mean w on landbird images (class labels only split the result)
Image bootstrap 95% CI (2000 draws, numpy default_rng 20261113) for r and qTw - qT per run; seed means per method.
No labels, groups or positions enter any mask or weight; groups are not used.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29, does trustmask_v1's teacher-map coverage weight put more KD weight on minority-"
                "class (waterbird) images, so that its weighted KD target carries more waterbird mass than the "
                "unweighted one?",
    "method": "attention rollout (teacher all blocks, student blocks 6..11); last-block CLS attention; keep-only masked "
              "teacher; coverage weight; class-conditional weight ratio; image bootstrap",
    "paper": "Hinton et al. 2015 (KD soft targets); Menon et al. ICLR 2021 (logit adjustment); Son et al. ECCV 2024 "
             "(MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Sagawa et al. ICLR 2020 (Waterbirds)",
}

K = 29
START = 6
NBOOT = 2000
BATCH = 50
EPOCHS = (25, "final")
METHODS = ("trustmask_v1", "maskedkd")


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


def _mask(api, m, S, x):
    depth = len(S.blocks)
    with torch.no_grad():
        so = api.student_forward(S, x, attn_layers=list(range(depth)))
    if m == "maskedkd":
        return _topk(so.attn[depth - 1].float().mean(1)[:, 0, 1:], K)
    return _topk(_rollout([so.attn[i] for i in range(START, depth)]), K)


def _stats(cov, pt, y):
    w = cov / cov.mean()
    qt = float(pt.mean())
    qtw = float((w * pt).sum() / w.sum())
    r = float(w[y == 1].mean() / w[y == 0].mean())
    return qt, qtw, r


def run(api):
    runs = api.runs()
    studs = []
    for rid, rr in sorted(runs.items()):
        if (rr.get("dataset") == "waterbirds" and rr["tier"] == "final" and rr["method"] in METHODS
                and rr.get("keep") == K and rr["seed"] in (1, 2, 3)):
            studs.append((rr["method"], rr["seed"], rid))
    api.write_json("students", [[m, s, rid, runs[rid]["val_wga_last"], runs[rid]["val_acc_last"]]
                                for m, s, rid in studs])
    T = api.teacher.eval()
    depth_t = len(T.blocks)

    batches, ys = [], []
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(T, x, attn_layers=list(range(depth_t)))
        batches.append((x, _rollout(out.attn)))
        ys.append(y.cpu())
    y = torch.cat(ys).numpy().astype(int)

    gen = numpy.random.default_rng(20261113)
    n = len(y)
    rows = []
    for m, seed, rid in studs:
        row = {"method": m, "seed": seed, "run": rid, "val_wga": runs[rid]["val_wga_last"],
               "val_acc": runs[rid]["val_acc_last"]}
        for ep in EPOCHS:
            S = api.student(rid, epoch=None if ep == "final" else ep)
            covs, pts = [], []
            for x, tm in batches:
                keep = _mask(api, m, S, x)
                covs.append((tm * keep).sum(1).float().cpu())
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float()
                pts.append(lm.softmax(-1)[:, 1].cpu())
            cov = torch.cat(covs).numpy().astype(float)
            pt = torch.cat(pts).numpy().astype(float)
            qt, qtw, r = _stats(cov, pt, y)
            bd, br = [], []
            for _ in range(NBOOT):
                i = gen.integers(0, n, n)
                if y[i].min() == y[i].max():
                    continue
                a, b, c = _stats(cov[i], pt[i], y[i])
                bd.append(b - a)
                br.append(c)
            q = lambda v, p: round(float(numpy.quantile(numpy.array(v), p)), 4)
            row[f"qT_{ep}"] = round(qt, 4)
            row[f"qTw_{ep}"] = round(qtw, 4)
            row[f"shift_{ep}"] = [round(qtw - qt, 4), q(bd, 0.025), q(bd, 0.975)]
            row[f"r_{ep}"] = [round(r, 4), q(br, 0.025), q(br, 0.975)]
            row[f"cov_mean_{ep}"] = round(float(cov.mean()), 4)
        rows.append(row)

    by_method = {}
    for m in METHODS:
        rs = [x for x in rows if x["method"] == m]
        if not rs:
            continue
        d = {"n_seeds": len(rs), "val_wga": round(float(numpy.mean([x["val_wga"] for x in rs])), 4)}
        for ep in EPOCHS:
            for k in ("qT", "qTw", "cov_mean"):
                d[f"{k}_{ep}"] = round(float(numpy.mean([x[f"{k}_{ep}"] for x in rs])), 4)
            for k in ("shift", "r"):
                d[f"{k}_{ep}"] = round(float(numpy.mean([x[f"{k}_{ep}"][0] for x in rs])), 4)
        by_method[m] = d
    v1 = [x for x in rows if x["method"] == "trustmask_v1"]
    out = {"k": K, "n_val": int(n), "n_waterbird": int(y.sum()), "n_runs": len(rows), "by_method": by_method,
           "runs": rows,
           "verdict": {"P1_v1_r25_ci_lo_gt_1_all_seeds": bool(len(v1) == 3 and all(x["r_25"][1] > 1 for x in v1)),
                       "P2_v1_qTw25_seedmean_ge_0.231": bool(len(v1) == 3 and
                                                             by_method["trustmask_v1"]["qTw_25"] >= 0.231)}}
    api.write_json("summary", out)
