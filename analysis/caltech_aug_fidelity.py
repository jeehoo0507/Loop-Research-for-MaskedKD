"""H-A181 (round 445, analysis only, no training): the training-view (crop+flip) version of H-A180. On Caltech-256
clean val, the K=29 lateroll and trustfilt masked teachers were closer to the full teacher than MaskedKD's
(last-block CLS attention) masked teacher (H-A180: -1.14, -0.45 nats). On cub_lt/Aircraft the clean-view picture of
masked-teacher targets changed in the training view (H-A77, H-A80). Training never shows the clean centred view, so
does the H-A180 order hold on RandomResizedCrop+flip views, and does the gap shrink or grow?

Students (dataset caltech256, seed 1, the only Caltech runs): full@196 (primary, rule-neutral), maskedkd@98, ce.
Views: clean val + V = 2 fixed crop+flip views per val image (harness GpuAugment RandomResizedCrop formula, numpy
default_rng 20261203; no RandAugment, erasing or Mixup). Teacher map on a view = clean teacher all-block rollout
warped with the crop grid and patch-averaged, as GpuAugment does for the cache.
Rules, K = 29 and 98: maskedkd, lateroll (student rollout blocks 6-11), trustfilt (lateroll restricted to the teacher
map top ceil(1.5K)), teacher_topk (teacher map top K). Student rules use the student on the view.
Per image: KL(full teacher on the view || keep-only masked teacher on the same view), T=1, mean over the V aug views.
P1: lateroll - maskedkd mean aug KL, K=29, full@196 student; 95% percentile bootstrap over val images
    (numpy default_rng(20261204), 2000 resamples).
P2: trustfilt - maskedkd, same.
Record: clean versions (recomputed), aug - clean change of each gap (same image draws), K=98, other students,
masked-teacher accuracy, student accuracy check on clean val.
Limit: one seed per student (image-level CI only), fixed trained students, two views per image.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on Caltech-256 crop+flip views of val, is the lateroll / trustfilt masked teacher still closer to the "
                "full teacher than the MaskedKD masked teacher at K=29 (H-A180 was clean view only)?",
    "method": "keep-only masked-teacher KL to the full teacher on RandomResizedCrop+flip views, attention rollout, "
              "warped cached teacher map, image bootstrap",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); Son et al. "
             "ECCV 2024 (MaskedKD); Szegedy et al. 2015 (RandomResizedCrop); Beyer et al. CVPR 2022 (consistent "
             "teaching); Efron & Tibshirani 1993 (bootstrap)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261204
V = 2
IMG = 224
G = 14
KS = (29, 98)
WIDEN = 1.5
RULES = ("maskedkd", "lateroll", "trustfilt", "teacher_topk")
STUDENTS = {
    "full": "caltech256_final_full_k196_s1_1103_123120_d6de",
    "maskedkd98": "caltech256_final_maskedkd_k98_s1_1103_123120_9e43",
    "ce": "caltech256_final_ce_k0_s1_1103_123120_2db8",
}


def _logp(L):
    Z = L - L.max(1, keepdims=True)
    return Z - numpy.log(numpy.exp(Z).sum(1, keepdims=True))


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
    m = r[:, 0, 1:]
    return m / m.sum(1, keepdim=True)


def _filt(s, t, k):
    m = int(math.ceil(WIDEN * k))
    allowed = t.argsort(1, descending=True).argsort(1) < m
    s = s.masked_fill(~allowed, -1.0)
    return s.argsort(1, descending=True).argsort(1) < k


def _grids(n, device):
    """V crop+flip affine grids per val image, harness GpuAugment RandomResizedCrop formula, fixed draws."""
    gen = numpy.random.default_rng(20261203)
    area = gen.uniform(0.08, 1.0, (V, n))
    r = numpy.exp(gen.uniform(math.log(3 / 4), math.log(4 / 3), (V, n)))
    w = numpy.minimum(numpy.sqrt(area * r), 1.0)
    h = numpy.minimum(numpy.sqrt(area / r), 1.0)
    cx = (gen.uniform(0, 1, (V, n)) * 2 - 1) * (1 - w)
    cy = (gen.uniform(0, 1, (V, n)) * 2 - 1) * (1 - h)
    flip = numpy.where(gen.uniform(0, 1, (V, n)) < 0.5, -1.0, 1.0)
    th = numpy.zeros((V, n, 2, 3))
    th[:, :, 0, 0] = w * flip
    th[:, :, 0, 2] = cx
    th[:, :, 1, 1] = h
    th[:, :, 1, 2] = cy
    return torch.tensor(th, dtype=torch.float32, device=device)


def _boot(d, rng):
    d = numpy.asarray(d, dtype=numpy.float64)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    bs = d[idx].mean(1)
    return {"mean": round(float(d.mean()), 4),
            "ci": [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]}


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"]] for rid in STUDENTS.values()})
    Tm = api.teacher.eval()
    depth_t = len(Tm.blocks)
    xs, ys = [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
    y_all = torch.cat(ys)
    n = len(y_all)
    th = _grids(n, y_all.device)

    # views[b] = (clean x, [(view_type, crop theta or None, teacher_map_view)]); full-teacher log-probs per view
    views, tl = [], {"clean": [], "aug": [[] for _ in range(V)]}
    off = 0
    for x, y in zip(xs, ys):
        b = len(y)
        with torch.no_grad():
            tl["clean"].append(Tm(x).float().cpu())
            out = api.student_forward(Tm, x, attn_layers=list(range(depth_t)))
        tm = _rollout(out.attn)
        vb = [("clean", None, tm)]
        tmap = tm.reshape(b, 1, G, G)
        for v in range(V):
            grid = F.affine_grid(th[v, off:off + b], (b, 3, IMG, IMG), align_corners=False)
            xa = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            ma = F.grid_sample(tmap.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            ma = F.adaptive_avg_pool2d(ma, G).flatten(1)
            with torch.no_grad():
                tl["aug"][v].append(Tm(xa).float().cpu())
            vb.append((v, th[v, off:off + b], ma))
        views.append((x, vb))
        off += b
    yn = y_all.cpu().numpy().astype(int)
    tlp = {"clean": _logp(torch.cat(tl["clean"]).numpy().astype(numpy.float64))}
    for v in range(V):
        tlp[v] = _logp(torch.cat(tl["aug"][v]).numpy().astype(numpy.float64))
    full_acc = {"clean": round(float((tlp["clean"].argmax(1) == yn).mean() * 100), 3),
                "aug": round(float(numpy.mean([(tlp[v].argmax(1) == yn).mean() for v in range(V)]) * 100), 3)}

    kl, acc, s_acc = {}, {}, {}
    for sname, rid in STUDENTS.items():
        S = api.student(rid)
        depth = len(S.blocks)
        gl = {(vt, r, k): [] for vt in ["clean"] + list(range(V)) for r in RULES for k in KS}
        sl = []
        for x, vb in views:
            for vt, theta, tm in vb:
                if theta is None:
                    xv = x
                else:
                    grid = F.affine_grid(theta, (len(x), 3, IMG, IMG), align_corners=False)
                    xv = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                with torch.no_grad():
                    if vt == "clean":
                        sl.append(api.student_forward(S, xv).logits.float().cpu())
                    last = api.student_forward(S, xv, attn_layers=[depth - 1]).attn[depth - 1].float().mean(1)[:, 0, 1:]
                    out = api.student_forward(S, xv, attn_layers=list(range(6, depth)))
                    lr = _rollout([out.attn[i] for i in range(6, depth)])
                    for k in KS:
                        masks = {"maskedkd": _topk_mask(last, k), "lateroll": _topk_mask(lr, k),
                                 "trustfilt": _filt(lr, tm, k), "teacher_topk": _topk_mask(tm, k)}
                        for r in RULES:
                            gl[(vt, r, k)].append(api.teacher_forward_keepmask(Tm, xv, masks[r]).float().cpu())
        del S
        torch.cuda.empty_cache()
        s_acc[sname] = round(float((torch.cat(sl).numpy().argmax(1) == yn).mean() * 100), 3)
        for r in RULES:
            for k in KS:
                glp = _logp(torch.cat(gl[("clean", r, k)]).numpy().astype(numpy.float64))
                kl[(sname, "clean", r, k)] = (numpy.exp(tlp["clean"]) * (tlp["clean"] - glp)).sum(1)
                acc[(sname, "clean", r, k)] = (glp.argmax(1) == yn).astype(numpy.float64) * 100
                ka, aa = [], []
                for v in range(V):
                    glp = _logp(torch.cat(gl[(v, r, k)]).numpy().astype(numpy.float64))
                    ka.append((numpy.exp(tlp[v]) * (tlp[v] - glp)).sum(1))
                    aa.append((glp.argmax(1) == yn).astype(numpy.float64) * 100)
                kl[(sname, "aug", r, k)] = numpy.mean(ka, 0)
                acc[(sname, "aug", r, k)] = numpy.mean(aa, 0)

    rng = numpy.random.default_rng(SEED)
    pairs = [("lateroll", "maskedkd"), ("trustfilt", "maskedkd"), ("trustfilt", "lateroll"),
             ("teacher_topk", "maskedkd"), ("teacher_topk", "lateroll")]
    rec = {}
    for s in STUDENTS:
        for k in KS:
            e = {}
            for vt in ("clean", "aug"):
                e[vt] = {
                    "kl_mean": {r: round(float(kl[(s, vt, r, k)].mean()), 4) for r in RULES},
                    "masked_teacher_acc": {r: round(float(acc[(s, vt, r, k)].mean()), 3) for r in RULES},
                    "kl_diff": {f"{a}-{b}": _boot(kl[(s, vt, a, k)] - kl[(s, vt, b, k)], rng) for a, b in pairs},
                    "acc_diff": {f"{a}-{b}": _boot(acc[(s, vt, a, k)] - acc[(s, vt, b, k)], rng)
                                 for a, b in pairs[:2]},
                }
            e["aug_minus_clean_gap"] = {
                f"{a}-{b}": _boot((kl[(s, "aug", a, k)] - kl[(s, "aug", b, k)])
                                  - (kl[(s, "clean", a, k)] - kl[(s, "clean", b, k)]), rng) for a, b in pairs[:3]}
            rec[f"{s}@K{k}"] = e
    api.write_json("summary", {
        "n_val": int(n), "views_per_image": V, "full_teacher_acc": full_acc,
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_lateroll_minus_maskedkd_aug_kl_full_K29": rec["full@K29"]["aug"]["kl_diff"]["lateroll-maskedkd"],
        "P2_trustfilt_minus_maskedkd_aug_kl_full_K29": rec["full@K29"]["aug"]["kl_diff"]["trustfilt-maskedkd"],
        "record": rec,
        "note": "KL(full teacher on view || keep-only masked teacher on same view), T=1, nats; aug = mean over V "
                "crop+flip views per image; 95% percentile bootstrap over val images (2000 resamples); seed-1 "
                "students only"})
