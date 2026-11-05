"""H-A186 (round 450, analysis only, no training): on Caltech-256, does MaskedKD's last-block CLS attention spend its
K=29 tokens on the student's high-norm "sink" tokens, and does dropping them close the K=29 lateroll - MaskedKD
masked-teacher KL gap (H-A180 -1.14)? CUB version: H-C10 (closure 0.79 at K=29). No Caltech sink analysis exists.

Sinks (same definition as H-C10, fixed before results): per image, patch tokens whose L2 norm at the block-10 output
(input of the block whose attention MaskedKD reads) exceeds median + 3 * MAD of that image's 196 patch norms.
Students (caltech256, seed 1, same runs as H-A180): full@196 (primary), maskedkd@98 (record).
Rules: maskedkd (last-block CLS attention top-K), nosink (same scores, sinks set to -1, next tokens fill),
       lateroll (student rollout over blocks 6-11 top-K, as H-A180~185).
Per image: KL(full teacher || keep-only masked teacher), T=1; DKD TCKD (record).
P1: share of sinks in maskedkd top-29 minus share in lateroll top-29, full student.
P2: closure = mean(KL_mkd - KL_nosink) / mean(KL_mkd - KL_lateroll), full student, K=29.
CI: 95% percentile bootstrap over val images (numpy default_rng(20261209), 2000 resamples, shared index).
Record: K=12/49, maskedkd98 student, sink count, attention mass on sinks, masked-teacher acc, accuracy check.
"""
import numpy
import torch

META = {
    "question": "on Caltech-256 val, does MaskedKD's K=29 last-block attention pick student high-norm sink tokens, "
                "and does excluding them close the lateroll - MaskedKD masked-teacher KL gap?",
    "method": "high-norm artifact token detection (per-image median+3MAD of block-10 token norms), keep-only "
              "masked-teacher KL and DKD TCKD, attention rollout, paired image bootstrap",
    "paper": "Darcet et al. ICLR 2024 (ViTs need registers); Abnar & Zuidema ACL 2020 (rollout); Zhao et al. CVPR 2022 "
             "(DKD); Son et al. ECCV 2024 (MaskedKD); Efron & Tibshirani 1993 (bootstrap)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261209
KS = (12, 29, 49)
MAD_K = 3.0
EPS = 1e-8
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


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"]] for rid in STUDENTS.values()})
    Tm = api.teacher.eval()
    batches, tl = [], []
    for x, y in api.val_batches(batch=BATCH):
        batches.append((x, y))
        with torch.no_grad():
            tl.append(Tm(x).float().cpu())
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    tlp = _logp(torch.cat(tl).numpy().astype(numpy.float64))
    tp = numpy.exp(tlp)
    pt_f = numpy.clip(tp[numpy.arange(n), y], EPS, 1 - EPS)

    kl, tckd, acc, share, s_acc, cnt, mass = {}, {}, {}, {}, {}, {}, {}
    for sname, rid in STUDENTS.items():
        S = api.student(rid)
        depth = len(S.blocks)
        gl = {(r, k): [] for r in RULES for k in KS}
        sh = {(r, k): [] for r in ("maskedkd", "lateroll") for k in KS}
        sl, c_l, m_l = [], [], []
        for x, _ in batches:
            with torch.no_grad():
                out = api.student_forward(S, x, attn_layers=list(range(6, depth)), token_layers=[depth - 2])
                sl.append(api.student_forward(S, x).logits.float().cpu())
                last = out.attn[depth - 1].float().mean(1)[:, 0, 1:]
                lr = _rollout([out.attn[i] for i in range(6, depth)])
                nrm = out.tokens[depth - 2][:, 1:].float().norm(dim=-1)
                med = nrm.median(dim=1, keepdim=True).values
                mad = (nrm - med).abs().median(dim=1, keepdim=True).values
                sink = nrm > med + MAD_K * mad
                c_l.append(sink.float().sum(1).cpu())
                m_l.append(((last * sink.float()).sum(1) / last.sum(1)).cpu())
                ns = last.masked_fill(sink, -1.0)
                for k in KS:
                    masks = {"maskedkd": _topk_mask(last, k), "nosink": _topk_mask(ns, k),
                             "lateroll": _topk_mask(lr, k)}
                    for r in ("maskedkd", "lateroll"):
                        sh[(r, k)].append(((masks[r] & sink).float().sum(1) / k).cpu())
                    for r in RULES:
                        gl[(r, k)].append(api.teacher_forward_keepmask(Tm, x, masks[r]).float().cpu())
        del S
        torch.cuda.empty_cache()
        s_acc[sname] = round(float((torch.cat(sl).numpy().argmax(1) == y).mean() * 100), 3)
        cnt[sname] = torch.cat(c_l).numpy().astype(numpy.float64)
        mass[sname] = torch.cat(m_l).numpy().astype(numpy.float64)
        for key, v in sh.items():
            share[(sname,) + key] = torch.cat(v).numpy().astype(numpy.float64)
        for (r, k), v in gl.items():
            glp = _logp(torch.cat(v).numpy().astype(numpy.float64))
            gp = numpy.exp(glp)
            pt_m = numpy.clip(gp[numpy.arange(n), y], EPS, 1 - EPS)
            kl[(sname, r, k)] = (tp * (tlp - glp)).sum(1)
            tckd[(sname, r, k)] = pt_f * numpy.log(pt_f / pt_m) + (1 - pt_f) * numpy.log((1 - pt_f) / (1 - pt_m))
            acc[(sname, r, k)] = (glp.argmax(1) == y).astype(numpy.float64) * 100

    rng = numpy.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))

    def boot(dv):
        dv = numpy.asarray(dv, dtype=numpy.float64)
        bs = dv[idx].mean(1)
        return {"mean": round(float(dv.mean()), 4),
                "ci": [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]}

    def closure(src, s, k):
        a = src[(s, "maskedkd", k)] - src[(s, "nosink", k)]
        b = src[(s, "maskedkd", k)] - src[(s, "lateroll", k)]
        bs = a[idx].mean(1) / b[idx].mean(1)
        return {"mean": round(float(a.mean() / b.mean()), 4),
                "ci": [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]}

    p1 = boot(share[("full", "maskedkd", 29)] - share[("full", "lateroll", 29)])
    p2 = closure(kl, "full", 29)
    rec = {}
    for s in STUDENTS:
        rec[s] = {"student_acc": s_acc[s], "sink_count_mean": round(float(cnt[s].mean()), 3),
                  "frac_images_with_sink": round(float((cnt[s] > 0).mean()), 4),
                  "last_attn_mass_on_sinks": round(float(mass[s].mean()), 4)}
        for k in KS:
            rec[s][f"K{k}"] = {
                "sink_share": {r: round(float(share[(s, r, k)].mean()), 4) for r in ("maskedkd", "lateroll")},
                "sink_share_diff_mkd-lateroll": boot(share[(s, "maskedkd", k)] - share[(s, "lateroll", k)]),
                "kl_mean": {r: round(float(kl[(s, r, k)].mean()), 4) for r in RULES},
                "tckd_mean": {r: round(float(tckd[(s, r, k)].mean()), 4) for r in RULES},
                "masked_teacher_acc": {r: round(float(acc[(s, r, k)].mean()), 3) for r in RULES},
                "kl_diff": {f"{a}-{b}": boot(kl[(s, a, k)] - kl[(s, b, k)]) for a, b in
                            [("nosink", "maskedkd"), ("lateroll", "maskedkd"), ("nosink", "lateroll")]},
                "closure_kl": closure(kl, s, k),
                "closure_tckd": closure(tckd, s, k),
            }
    api.write_json("summary", {
        "n_val": int(n), "teacher_acc": round(float((tlp.argmax(1) == y).mean() * 100), 3),
        "student_acc_check": {s: [s_acc[s], runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "P1_sink_share_maskedkd_minus_lateroll_full_K29": p1,
        "P2_closure_kl_full_K29": p2,
        "record": rec,
        "note": "KL(full teacher || keep-only masked teacher), T=1, nats; sinks = block-10 output patch norm > "
                "median + 3 MAD per image; closure = mean(KL_mkd - KL_nosink) / mean(KL_mkd - KL_lateroll); 95% "
                "percentile bootstrap over val images (2000 resamples, shared index); seed-1 students only"})
