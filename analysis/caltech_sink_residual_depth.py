"""H-A190 (round 454, analysis only, no training): what closes the rest of the gap after sinks are removed? On
Caltech-256 crop+flip views, excluding student high-norm sink tokens from MaskedKD's K=29 last-block CLS attention
closed 0.55 of the lateroll - MaskedKD masked-teacher KL gap, leaving nosink - lateroll = +0.46 (H-A189). lateroll
differs from nosink in two ways: it uses blocks 6-11 instead of block 11 only, and it multiplies them (rollout with
residual identity). Is the residual from using several blocks (depth pooling, no product), or from the product?

Sinks (same definition as H-A186/189, fixed before results): per view, patch tokens whose L2 norm at the block-10
output exceeds median + 3 * MAD of that view's 196 patch norms.
Students (caltech256, seed 1, same runs as H-A186/189): full@196 (primary), maskedkd@98 (record).
Views: clean val + V = 2 fixed crop+flip views per val image (same draws as H-A181/188/189).
Rules, K = 12, 29, 49 (all decomposition controls, not selectors):
  maskedkd  last-block (11) CLS attention top-K
  nosink    same scores, sinks set to -1 (next tokens fill)
  avgnosink mean over blocks 6-11 of the head-mean CLS->patch attention, sinks set to -1
  blkB      block-B CLS attention, sinks set to -1, B = 6..10 (record)
  lateroll  student rollout blocks 6-11 top-K
  rollnosink lateroll scores, sinks set to -1 (record)
Per image: KL(full teacher on the view || keep-only masked teacher on the same view), T=1, mean over the V views.
P1: KL_avgnosink - KL_nosink, aug views, full student, K=29.
P2: residual closure = mean(KL_nosink - KL_avgnosink) / mean(KL_nosink - KL_lateroll), aug, full, K=29.
CI: 95% percentile bootstrap over val images (numpy default_rng(20261214), 2000 resamples, shared index).
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on Caltech-256 crop+flip views of val, after excluding student high-norm sink tokens, does averaging "
                "CLS attention over blocks 6-11 (no product) close the remaining nosink - lateroll masked-teacher KL "
                "gap at K=29, or is the rollout product needed?",
    "method": "high-norm artifact token detection (per-view median+3MAD of block-10 token norms), per-block and "
              "depth-averaged CLS attention vs attention rollout, keep-only masked-teacher KL on "
              "RandomResizedCrop+flip views, paired image bootstrap",
    "paper": "Darcet et al. ICLR 2024 (ViTs need registers); Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL "
             "2020 (rollout); Szegedy et al. 2015 (RandomResizedCrop); Efron & Tibshirani 1993 (bootstrap)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261214
V = 2
IMG = 224
KS = (12, 29, 49)
MAD_K = 3.0
BLKS = (6, 7, 8, 9, 10)
RULES = ("maskedkd", "nosink", "avgnosink", "lateroll", "rollnosink") + tuple(f"blk{b}" for b in BLKS)
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

    kl, acc, ovl, s_acc = {}, {}, {}, {}
    for sname, rid in STUDENTS.items():
        S = api.student(rid)
        depth = len(S.blocks)
        gl = {(vt, r, k): [] for vt in vts for r in RULES for k in KS}
        ov = {(vt, r, k): [] for vt in vts for r in ("avgnosink", "lateroll") for k in KS}
        sl = []
        for x, vb in zip(xs, views):
            for vt, theta in vb:
                xv = _view(x, theta)
                with torch.no_grad():
                    if vt == "clean":
                        sl.append(api.student_forward(S, xv).logits.float().cpu())
                    out = api.student_forward(S, xv, attn_layers=list(range(6, depth)), token_layers=[depth - 2])
                    cls = {i: out.attn[i].float().mean(1)[:, 0, 1:] for i in range(6, depth)}
                    last = cls[depth - 1]
                    lr = _rollout([out.attn[i] for i in range(6, depth)])
                    nrm = out.tokens[depth - 2][:, 1:].float().norm(dim=-1)
                    med = nrm.median(dim=1, keepdim=True).values
                    mad = (nrm - med).abs().median(dim=1, keepdim=True).values
                    sink = nrm > med + MAD_K * mad
                    avg = torch.stack([cls[i] for i in range(6, depth)]).mean(0)
                    sc = {"maskedkd": last, "nosink": last.masked_fill(sink, -1.0),
                          "avgnosink": avg.masked_fill(sink, -1.0), "lateroll": lr,
                          "rollnosink": lr.masked_fill(sink, -1.0)}
                    for b_ in BLKS:
                        sc[f"blk{b_}"] = cls[b_].masked_fill(sink, -1.0)
                    for k in KS:
                        masks = {r: _topk_mask(sc[r], k) for r in RULES}
                        for r in ("avgnosink", "lateroll"):
                            ov[(vt, r, k)].append(((masks[r] & masks["nosink"]).float().sum(1) / k).cpu())
                        for r in RULES:
                            gl[(vt, r, k)].append(api.teacher_forward_keepmask(Tm, xv, masks[r]).float().cpu())
        del S
        torch.cuda.empty_cache()
        s_acc[sname] = round(float((torch.cat(sl).numpy().argmax(1) == yn).mean() * 100), 3)

        for tag, vt_set in (("clean", ["clean"]), ("aug", list(range(V)))):
            for k in KS:
                for r in ("avgnosink", "lateroll"):
                    ovl[(sname, tag, r, k)] = numpy.mean(
                        [torch.cat(ov[(vt, r, k)]).numpy().astype(numpy.float64) for vt in vt_set], 0)
                for r in RULES:
                    ka, aa = [], []
                    for vt in vt_set:
                        glp = _logp(torch.cat(gl[(vt, r, k)]).numpy().astype(numpy.float64))
                        ka.append((numpy.exp(tlp[vt]) * (tlp[vt] - glp)).sum(1))
                        aa.append((glp.argmax(1) == yn).astype(numpy.float64) * 100)
                    kl[(sname, tag, r, k)] = numpy.mean(ka, 0)
                    acc[(sname, tag, r, k)] = numpy.mean(aa, 0)

    rng = numpy.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))

    def _ci(bs):
        return [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]

    def boot(dv):
        dv = numpy.asarray(dv, dtype=numpy.float64)
        return {"mean": round(float(dv.mean()), 4), "ci": _ci(dv[idx].mean(1))}

    def closure(s, tag, k, r):
        a = kl[(s, tag, "nosink", k)] - kl[(s, tag, r, k)]
        b = kl[(s, tag, "nosink", k)] - kl[(s, tag, "lateroll", k)]
        return {"mean": round(float(a.mean() / b.mean()), 4), "ci": _ci(a[idx].mean(1) / b[idx].mean(1))}

    p1 = boot(kl[("full", "aug", "avgnosink", 29)] - kl[("full", "aug", "nosink", 29)])
    p2 = closure("full", "aug", 29, "avgnosink")
    rec = {}
    for s in STUDENTS:
        rec[s] = {"student_acc": s_acc[s]}
        for tag in ("clean", "aug"):
            e = {}
            for k in KS:
                e[f"K{k}"] = {
                    "kl_mean": {r: round(float(kl[(s, tag, r, k)].mean()), 4) for r in RULES},
                    "masked_teacher_acc": {r: round(float(acc[(s, tag, r, k)].mean()), 3) for r in RULES},
                    "overlap_with_nosink": {r: round(float(ovl[(s, tag, r, k)].mean()), 4)
                                            for r in ("avgnosink", "lateroll")},
                    "kl_diff": {f"{a}-{b}": boot(kl[(s, tag, a, k)] - kl[(s, tag, b, k)]) for a, b in
                                [("avgnosink", "nosink"), ("avgnosink", "lateroll"), ("nosink", "lateroll"),
                                 ("rollnosink", "lateroll")] + [(f"blk{b_}", "nosink") for b_ in BLKS]},
                    "residual_closure_avgnosink": closure(s, tag, k, "avgnosink"),
                }
            rec[s][tag] = e
    api.write_json("summary", {
        "n_val": int(n), "views_per_image": V, "full_teacher_acc": full_acc,
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_kl_avgnosink_minus_nosink_aug_full_K29": p1,
        "P2_residual_closure_avgnosink_aug_full_K29": p2,
        "record": rec,
        "note": "KL(full teacher on view || keep-only masked teacher on same view), T=1, nats; aug = mean over V "
                "crop+flip views per image (H-A181 draws); sinks = block-10 output patch norm > median + 3 MAD per "
                "view; avgnosink = mean over blocks 6-11 of head-mean CLS->patch attention with sinks set to -1; "
                "residual closure = mean(KL_nosink - KL_avgnosink) / mean(KL_nosink - KL_lateroll); 95% percentile "
                "bootstrap over val images (2000 resamples, shared index); seed-1 students only"})
