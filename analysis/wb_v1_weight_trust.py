"""H-A50 (round 297, analysis only, no training): on Waterbirds K=29, does trustmask_v1's per-image KD weight move
KD weight away from images whose masked-teacher target is wrong, in particular in the minority groups?

H-A49: v1's coverage weight is class-neutral (waterbird/landbird weight ratio 0.99), so it does not re-weight the
target's class mass. The other way a per-image weight can change what the student learns is by trusting the right
images: if low coverage marks images where the masked teacher points to the wrong class, the weighted KD signal is
more often correct than the unweighted one, and minority-group images are where the masked teacher is most often
wrong (H-A41).

Students: Waterbirds final trustmask_v1 @29, seeds 1, 2, 3 (main); maskedkd @29 seeds 1, 2, 3 (record: the same
coverage weight computed on maskedkd's tokens, which maskedkd does not use).
Mask: each run's own rule on the un-augmented val image (same as H-A48/49), with the epoch-25 and final student.
Weight: w = teacher-rollout-map coverage of the kept tokens / mean over all val images of that run/epoch (as H-A49).
Per image: pt = p_T^mask(y) (TAU = 1), tacc = 1[argmax masked teacher = y].
Per run and epoch:
  rho     Spearman(w, pt) over all val images
  dacc    weighted - unweighted masked-teacher accuracy, sum(w * tacc) / sum(w) - mean(tacc), all images
  dacc_m  the same on minority-group images (groups 1 and 2)
  rmin    mean w on minority-group images / mean w on majority-group images
Image bootstrap 95% CI (2000 draws, numpy default_rng 20261114) per run; seed means per method.
No labels, groups or positions enter any mask or weight; labels enter only measured quantities; groups only slice.
"""
import numpy
import torch

META = {
    "question": "on Waterbirds K=29, does trustmask_v1's teacher-map coverage weight put less KD weight on images "
                "whose masked-teacher target is wrong (overall and in the minority groups)?",
    "method": "attention rollout (teacher all blocks, student blocks 6..11); last-block CLS attention; keep-only masked "
              "teacher; coverage weight; Spearman of weight vs masked-teacher p(y); weighted vs unweighted target "
              "accuracy sliced by Waterbirds group; image bootstrap",
    "paper": "Hinton et al. 2015 (KD soft targets); Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 "
             "(rollout); Sagawa et al. ICLR 2020 (Waterbirds groups); Zhou et al. ICLR 2021 (weighted soft labels)",
}

K = 29
START = 6
NBOOT = 2000
BATCH = 50
EPOCHS = (25, "final")
METHODS = ("trustmask_v1", "maskedkd")
MINOR = (1, 2)


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


def _rank(v):
    o = numpy.argsort(v, kind="stable")
    r = numpy.empty(len(v))
    r[o] = numpy.arange(len(v))
    return r


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = numpy.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def _stats(cov, pt, tacc, minor):
    w = cov / cov.mean()
    rho = _spearman(w, pt)
    dacc = float((w * tacc).sum() / w.sum() - tacc.mean())
    wm, tm = w[minor], tacc[minor]
    dacc_m = float((wm * tm).sum() / wm.sum() - tm.mean())
    rmin = float(wm.mean() / w[~minor].mean())
    return rho, dacc, dacc_m, rmin


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
        batches.append((x, y, _rollout(out.attn)))
        ys.append(y.cpu())
    y = torch.cat(ys).numpy().astype(int)
    groups = api.val_groups().numpy()
    minor = numpy.isin(groups, MINOR)

    gen = numpy.random.default_rng(20261114)
    n = len(y)
    keys = ("rho", "dacc", "dacc_m", "rmin")
    rows = []
    for m, seed, rid in studs:
        row = {"method": m, "seed": seed, "run": rid, "val_wga": runs[rid]["val_wga_last"],
               "val_acc": runs[rid]["val_acc_last"]}
        for ep in EPOCHS:
            S = api.student(rid, epoch=None if ep == "final" else ep)
            covs, pts, accs = [], [], []
            for x, yb, tm in batches:
                keep = _mask(api, m, S, x)
                covs.append((tm * keep).sum(1).float().cpu())
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float()
                yb = yb.to(lm.device).long()
                pts.append(lm.softmax(-1).gather(1, yb[:, None])[:, 0].cpu())
                accs.append((lm.argmax(-1) == yb).float().cpu())
            cov = torch.cat(covs).numpy().astype(float)
            pt = torch.cat(pts).numpy().astype(float)
            tacc = torch.cat(accs).numpy().astype(float)
            point = _stats(cov, pt, tacc, minor)
            boot = []
            for _ in range(NBOOT):
                i = gen.integers(0, n, n)
                if minor[i].all() or not minor[i].any():
                    continue
                boot.append(_stats(cov[i], pt[i], tacc[i], minor[i]))
            boot = numpy.array(boot)
            for j, k in enumerate(keys):
                row[f"{k}_{ep}"] = [round(point[j], 4), round(float(numpy.quantile(boot[:, j], 0.025)), 4),
                                    round(float(numpy.quantile(boot[:, j], 0.975)), 4)]
            row[f"tacc_{ep}"] = round(float(tacc.mean()), 4)
            row[f"tacc_minor_{ep}"] = round(float(tacc[minor].mean()), 4)
        rows.append(row)

    by_method = {}
    for m in METHODS:
        rs = [x for x in rows if x["method"] == m]
        if not rs:
            continue
        d = {"n_seeds": len(rs), "val_wga": round(float(numpy.mean([x["val_wga"] for x in rs])), 4)}
        for ep in EPOCHS:
            for k in keys:
                d[f"{k}_{ep}"] = round(float(numpy.mean([x[f"{k}_{ep}"][0] for x in rs])), 4)
            for k in ("tacc", "tacc_minor"):
                d[f"{k}_{ep}"] = round(float(numpy.mean([x[f"{k}_{ep}"] for x in rs])), 4)
        by_method[m] = d
    v1 = [x for x in rows if x["method"] == "trustmask_v1"]
    out = {"k": K, "n_val": int(n), "n_minor": int(minor.sum()), "n_runs": len(rows), "by_method": by_method,
           "runs": rows,
           "verdict": {"P1_v1_rho25_ci_lo_gt_0_all_seeds": bool(len(v1) == 3 and all(x["rho_25"][1] > 0 for x in v1)),
                       "P2_v1_dacc_m25_seedmean_gt_0": bool(len(v1) == 3 and
                                                            by_method["trustmask_v1"]["dacc_m_25"] > 0)}}
    api.write_json("summary", out)
