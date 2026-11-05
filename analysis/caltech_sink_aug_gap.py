"""H-A189 (round 453, analysis only, no training): the training-view (crop+flip) version of H-A186. On Caltech-256
clean val, MaskedKD's K=29 last-block attention spent +0.37 more of its budget on student high-norm sink tokens than
lateroll, and excluding sinks closed 0.65 of the lateroll - MaskedKD masked-teacher KL gap (H-A186). Training only
shows RandomResizedCrop+flip views, and clean-val masked-teacher stats can mislead (H-A77, H-A188). Does the sink
explanation hold on crop views?

Sinks (same definition as H-A186 / H-C10, fixed before results): per view, patch tokens whose L2 norm at the block-10
output exceeds median + 3 * MAD of that view's 196 patch norms.
Students (caltech256, seed 1, same runs as H-A186/188): full@196 (primary), maskedkd@98 (record).
Views: clean val + V = 2 fixed crop+flip views per val image (harness GpuAugment RandomResizedCrop formula, numpy
default_rng 20261203, the same draws as H-A181/188; no RandAugment, erasing or Mixup).
Rules, K = 12, 29, 49: maskedkd (last-block CLS attention top-K), nosink (same scores, sinks set to -1, next tokens
fill), lateroll (student rollout blocks 6-11 top-K).
Per image: KL(full teacher on the view || keep-only masked teacher on the same view), T=1, mean over the V views.
P1: sink share in maskedkd top-29 minus share in lateroll top-29, aug views, full student.
P2: closure = mean(KL_mkd - KL_nosink) / mean(KL_mkd - KL_lateroll), aug views, full student, K=29.
CI: 95% percentile bootstrap over val images (numpy default_rng(20261213), 2000 resamples, shared index).
Record: K=12/49, maskedkd98 student, clean values (check vs H-A186), aug - clean closure change, sink counts,
attention mass on sinks, masked-teacher acc, accuracy check.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on Caltech-256 crop+flip views of val, does MaskedKD's K=29 last-block attention still spend its "
                "budget on student high-norm sink tokens, and does excluding them still close the lateroll - MaskedKD "
                "masked-teacher KL gap (H-A186 was clean view only)?",
    "method": "high-norm artifact token detection (per-view median+3MAD of block-10 token norms), keep-only "
              "masked-teacher KL on RandomResizedCrop+flip views, attention rollout, paired image bootstrap",
    "paper": "Darcet et al. ICLR 2024 (ViTs need registers); Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL "
             "2020 (rollout); Szegedy et al. 2015 (RandomResizedCrop); Efron & Tibshirani 1993 (bootstrap)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261213
V = 2
IMG = 224
KS = (12, 29, 49)
MAD_K = 3.0
RULES = ("maskedkd", "nosink", "lateroll")
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

    kl, acc, share, cnt, mass, s_acc = {}, {}, {}, {}, {}, {}
    for sname, rid in STUDENTS.items():
        S = api.student(rid)
        depth = len(S.blocks)
        gl = {(vt, r, k): [] for vt in vts for r in RULES for k in KS}
        sh = {(vt, r, k): [] for vt in vts for r in ("maskedkd", "lateroll") for k in KS}
        c_l = {vt: [] for vt in vts}
        m_l = {vt: [] for vt in vts}
        sl = []
        for x, vb in zip(xs, views):
            for vt, theta in vb:
                xv = _view(x, theta)
                with torch.no_grad():
                    if vt == "clean":
                        sl.append(api.student_forward(S, xv).logits.float().cpu())
                    out = api.student_forward(S, xv, attn_layers=list(range(6, depth)), token_layers=[depth - 2])
                    last = out.attn[depth - 1].float().mean(1)[:, 0, 1:]
                    lr = _rollout([out.attn[i] for i in range(6, depth)])
                    nrm = out.tokens[depth - 2][:, 1:].float().norm(dim=-1)
                    med = nrm.median(dim=1, keepdim=True).values
                    mad = (nrm - med).abs().median(dim=1, keepdim=True).values
                    sink = nrm > med + MAD_K * mad
                    c_l[vt].append(sink.float().sum(1).cpu())
                    m_l[vt].append(((last * sink.float()).sum(1) / last.sum(1)).cpu())
                    ns = last.masked_fill(sink, -1.0)
                    for k in KS:
                        masks = {"maskedkd": _topk_mask(last, k), "nosink": _topk_mask(ns, k),
                                 "lateroll": _topk_mask(lr, k)}
                        for r in ("maskedkd", "lateroll"):
                            sh[(vt, r, k)].append(((masks[r] & sink).float().sum(1) / k).cpu())
                        for r in RULES:
                            gl[(vt, r, k)].append(api.teacher_forward_keepmask(Tm, xv, masks[r]).float().cpu())
        del S
        torch.cuda.empty_cache()
        s_acc[sname] = round(float((torch.cat(sl).numpy().argmax(1) == yn).mean() * 100), 3)

        def _avg(d, vt_set):
            return numpy.mean([torch.cat(d[vt]).numpy().astype(numpy.float64) for vt in vt_set], 0)

        for tag, vt_set in (("clean", ["clean"]), ("aug", list(range(V)))):
            cnt[(sname, tag)] = _avg(c_l, vt_set)
            mass[(sname, tag)] = _avg(m_l, vt_set)
            for r in ("maskedkd", "lateroll"):
                for k in KS:
                    share[(sname, tag, r, k)] = numpy.mean(
                        [torch.cat(sh[(vt, r, k)]).numpy().astype(numpy.float64) for vt in vt_set], 0)
            for r in RULES:
                for k in KS:
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

    def _closure_parts(s, tag, k):
        a = kl[(s, tag, "maskedkd", k)] - kl[(s, tag, "nosink", k)]
        b = kl[(s, tag, "maskedkd", k)] - kl[(s, tag, "lateroll", k)]
        return a, b

    def closure(s, tag, k):
        a, b = _closure_parts(s, tag, k)
        return {"mean": round(float(a.mean() / b.mean()), 4), "ci": _ci(a[idx].mean(1) / b[idx].mean(1))}

    def closure_change(s, k):
        aa, ba = _closure_parts(s, "aug", k)
        ac, bc = _closure_parts(s, "clean", k)
        bs = aa[idx].mean(1) / ba[idx].mean(1) - ac[idx].mean(1) / bc[idx].mean(1)
        return {"mean": round(float(aa.mean() / ba.mean() - ac.mean() / bc.mean()), 4), "ci": _ci(bs)}

    p1 = boot(share[("full", "aug", "maskedkd", 29)] - share[("full", "aug", "lateroll", 29)])
    p2 = closure("full", "aug", 29)
    rec = {}
    for s in STUDENTS:
        rec[s] = {"student_acc": s_acc[s]}
        for tag in ("clean", "aug"):
            e = {"sink_count_mean": round(float(cnt[(s, tag)].mean()), 3),
                 "last_attn_mass_on_sinks": round(float(mass[(s, tag)].mean()), 4)}
            for k in KS:
                e[f"K{k}"] = {
                    "sink_share": {r: round(float(share[(s, tag, r, k)].mean()), 4) for r in ("maskedkd", "lateroll")},
                    "sink_share_diff_mkd-lateroll": boot(share[(s, tag, "maskedkd", k)]
                                                         - share[(s, tag, "lateroll", k)]),
                    "kl_mean": {r: round(float(kl[(s, tag, r, k)].mean()), 4) for r in RULES},
                    "masked_teacher_acc": {r: round(float(acc[(s, tag, r, k)].mean()), 3) for r in RULES},
                    "kl_diff": {f"{a}-{b}": boot(kl[(s, tag, a, k)] - kl[(s, tag, b, k)]) for a, b in
                                [("nosink", "maskedkd"), ("lateroll", "maskedkd"), ("nosink", "lateroll")]},
                    "closure_kl": closure(s, tag, k),
                }
            rec[s][tag] = e
        rec[s]["aug_minus_clean_closure"] = {f"K{k}": closure_change(s, k) for k in KS}
    api.write_json("summary", {
        "n_val": int(n), "views_per_image": V, "full_teacher_acc": full_acc,
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_sink_share_maskedkd_minus_lateroll_aug_full_K29": p1,
        "P2_closure_kl_aug_full_K29": p2,
        "record": rec,
        "note": "KL(full teacher on view || keep-only masked teacher on same view), T=1, nats; aug = mean over V "
                "crop+flip views per image (H-A181 draws); sinks = block-10 output patch norm > median + 3 MAD per "
                "view; closure = mean(KL_mkd - KL_nosink) / mean(KL_mkd - KL_lateroll); 95% percentile bootstrap "
                "over val images (2000 resamples, shared index); seed-1 students only"})
