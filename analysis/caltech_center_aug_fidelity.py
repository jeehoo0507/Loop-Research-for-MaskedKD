"""H-A188 (round 452, analysis only, no training): the training-view (crop+flip) version of H-A187. On Caltech-256
clean val, a fixed centre@29 mask preserved the teacher better than lateroll (H-A187 P1 +0.138 nats), so the
position prior is strong in the clean centred view. Training never shows that view (RandomResizedCrop+flip), and
H-A181 found the lateroll - MaskedKD order holds on crop views. Does centre@K still beat lateroll on crop views,
and does the lateroll - centre gap move in lateroll's favour from clean to crop views?

Students (caltech256, seed 1, same runs as H-A187): full@196 (primary), maskedkd@98 (record).
Views: clean val + V = 2 fixed crop+flip views per val image (harness GpuAugment RandomResizedCrop formula, numpy
default_rng 20261203, the same draws as H-A181; no RandAugment, erasing or Mixup).
Rules, K = 12, 29, 49: maskedkd (last-block CLS attention top-K), lateroll (student rollout blocks 6-11 top-K),
center (K patches nearest the 14x14 grid centre of the view, fixed), random (numpy default_rng(20261211)).
Per image: KL(full teacher on the view || keep-only masked teacher on the same view), T=1, mean over the V views.
P1: KL_lateroll - KL_center, aug views, full student, K=29.
P2: (lateroll - center on aug) - (lateroll - center on clean), same images, full student, K=29.
CI: 95% percentile bootstrap over val images (numpy default_rng(20261212), 2000 resamples, shared index).
Record: K=12/49, maskedkd98 student, other rule pairs, masked-teacher acc, accuracy check.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on Caltech-256 crop+flip views of val, does a fixed centre@K mask still preserve the teacher better "
                "than lateroll at K=29 (H-A187 was clean view only), and does the gap shift from clean to crop?",
    "method": "keep-only masked-teacher KL to the full teacher on RandomResizedCrop+flip views for student-attention, "
              "rollout, fixed-centre and random masks; paired image bootstrap",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL 2020 (rollout); Szegedy et al. 2015 "
             "(RandomResizedCrop); Beyer et al. CVPR 2022 (consistent teaching); Efron & Tibshirani 1993 (bootstrap)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261212
RND_SEED = 20261211
V = 2
IMG = 224
G = 14
KS = (12, 29, 49)
RULES = ("maskedkd", "lateroll", "center", "random")
STUDENTS = {
    "full": "caltech256_final_full_k196_s1_1103_123120_d6de",
    "maskedkd98": "caltech256_final_maskedkd_k98_s1_1103_123120_9e43",
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
    return r[:, 0, 1:]


def _grids(n, device):
    """V crop+flip affine grids per val image, harness GpuAugment RandomResizedCrop formula, fixed draws (H-A181)."""
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


def _view(x, theta):
    if theta is None:
        return x
    grid = F.affine_grid(theta, (len(x), 3, IMG, IMG), align_corners=False)
    return F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"]] for rid in STUDENTS.values()})
    Tm = api.teacher.eval()
    xs, ys = [], []
    for x, y in api.val_batches(batch=BATCH):
        xs.append(x)
        ys.append(y)
    y_all = torch.cat(ys)
    n = len(y_all)
    th = _grids(n, y_all.device)
    vts = ["clean"] + list(range(V))

    # per batch: list of (view type, crop theta or None); full-teacher logits per view
    views, tl = [], {vt: [] for vt in vts}
    off = 0
    for x in xs:
        b = len(x)
        vb = [("clean", None)] + [(v, th[v, off:off + b]) for v in range(V)]
        for vt, theta in vb:
            with torch.no_grad():
                tl[vt].append(Tm(_view(x, theta)).float().cpu())
        views.append(vb)
        off += b
    yn = y_all.cpu().numpy().astype(int)
    tlp = {vt: _logp(torch.cat(tl[vt]).numpy().astype(numpy.float64)) for vt in vts}
    full_acc = {"clean": round(float((tlp["clean"].argmax(1) == yn).mean() * 100), 3),
                "aug": round(float(numpy.mean([(tlp[v].argmax(1) == yn).mean() for v in range(V)]) * 100), 3)}

    pos = torch.arange(G * G)
    c = (G - 1) / 2
    dist = ((pos // G - c) ** 2 + (pos % G - c) ** 2).float().sqrt() / math.sqrt(2 * c * c)
    cen_score = -dist - pos.float() * 1e-6
    gen = numpy.random.default_rng(RND_SEED)
    rnd_all = {vt: [torch.from_numpy(gen.random((len(x), G * G))).float() for x in xs] for vt in vts}

    kl, acc, s_acc = {}, {}, {}
    for sname, rid in STUDENTS.items():
        S = api.student(rid)
        depth = len(S.blocks)
        gl = {(vt, r, k): [] for vt in vts for r in RULES for k in KS}
        sl = []
        for bi, (x, vb) in enumerate(zip(xs, views)):
            for vt, theta in vb:
                xv = _view(x, theta)
                B = len(xv)
                with torch.no_grad():
                    if vt == "clean":
                        sl.append(api.student_forward(S, xv).logits.float().cpu())
                    out = api.student_forward(S, xv, attn_layers=list(range(6, depth)))
                    last = out.attn[depth - 1].float().mean(1)[:, 0, 1:]
                    lr = _rollout([out.attn[i] for i in range(6, depth)])
                    cs = cen_score.to(xv.device).expand(B, G * G)
                    rnd = rnd_all[vt][bi].to(xv.device)
                    for k in KS:
                        masks = {"maskedkd": _topk_mask(last, k), "lateroll": _topk_mask(lr, k),
                                 "center": _topk_mask(cs, k), "random": _topk_mask(rnd, k)}
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
    idx = rng.integers(0, n, size=(N_BOOT, n))

    def boot(dv):
        dv = numpy.asarray(dv, dtype=numpy.float64)
        bs = dv[idx].mean(1)
        return {"mean": round(float(dv.mean()), 4),
                "ci": [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]}

    def gap(s, vt, a, b, k):
        return kl[(s, vt, a, k)] - kl[(s, vt, b, k)]

    pairs = [("lateroll", "center"), ("center", "maskedkd"), ("lateroll", "maskedkd"), ("center", "random"),
             ("maskedkd", "random")]
    p1 = boot(gap("full", "aug", "lateroll", "center", 29))
    p2 = boot(gap("full", "aug", "lateroll", "center", 29) - gap("full", "clean", "lateroll", "center", 29))
    rec = {}
    for s in STUDENTS:
        rec[s] = {"student_acc": s_acc[s]}
        for k in KS:
            e = {}
            for vt in ("clean", "aug"):
                e[vt] = {
                    "kl_mean": {r: round(float(kl[(s, vt, r, k)].mean()), 4) for r in RULES},
                    "masked_teacher_acc": {r: round(float(acc[(s, vt, r, k)].mean()), 3) for r in RULES},
                    "kl_diff": {f"{a}-{b}": boot(gap(s, vt, a, b, k)) for a, b in pairs},
                }
            e["aug_minus_clean_gap"] = {f"{a}-{b}": boot(gap(s, "aug", a, b, k) - gap(s, "clean", a, b, k))
                                        for a, b in pairs[:3]}
            rec[s][f"K{k}"] = e
    api.write_json("summary", {
        "n_val": int(n), "views_per_image": V, "full_teacher_acc": full_acc,
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_kl_lateroll_minus_center_aug_full_K29": p1,
        "P2_aug_minus_clean_lateroll_center_gap_full_K29": p2,
        "record": rec,
        "note": "KL(full teacher on view || keep-only masked teacher on same view), T=1, nats; aug = mean over V "
                "crop+flip views per image (H-A181 draws); center = K patches nearest the view's 14x14 grid centre; "
                "random = numpy default_rng(20261211); 95% percentile bootstrap over val images (2000 resamples, "
                "shared index); seed-1 students only"})
