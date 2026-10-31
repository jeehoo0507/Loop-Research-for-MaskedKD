"""H-C7: at the same mean teacher token budget, does a per-image K chosen from a student signal keep the masked
teacher's soft label better than a uniform K?

Basis: H-C4 (analysis_cub_proxy_validation_1031_165024_bc29): within the same K, masked-teacher TCKD rank-predicts
CUB student accuracy. H-C5/H-C6 (analysis_cub_token_marginal_1031_165359_d18b,
analysis_cub_token_novelty_1031_165711_d835): no per-token student score explains the add-one TCKD gain beyond
|rho| 0.1, so *which* token is near a ceiling. This asks whether *how many* tokens per image is a lever: the
selector interface allows a per-sample K with batch mean <= budget, and per-image TCKD at a fixed K varies a lot.

Per val image (common student = CUB final full-KD seed 1, tokens ranked by lateroll, keep-only teacher):
  tckd[K] for K in 12, 29, 49.
Student signals (no labels, no position): ent = entropy of the student's prediction, conf = max prob,
  conc = share of lateroll mass in its top-12, margin = top1 - top2 prob.
Two-level allocation at mean budget 29 from K in {12, 49}: the fraction p = (29-12)/(49-12) of images with the
highest "need" get 49, the rest 12. need = ent, -conf, -conc, -margin; oracle need = tckd[12] - tckd[49]
(analysis-only upper bound); random need (numpy default_rng). Compared with uniform K=29 by mean TCKD and by
mean KL, with a paired bootstrap over images. Teacher cost is approximately linear in K at these sizes; the exact
per-image GFLOPs of both policies are reported.
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on CUB, at mean K=29, does allocating K in {12,49} per image by a student signal (entropy, "
                "confidence, attention concentration, margin) keep the masked teacher's TCKD lower than uniform K=29?",
    "method": "keep-only masked teacher at several K per image; budget-matched two-level allocation by student "
              "uncertainty / attention concentration; paired bootstrap against uniform K",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only); Zhao et al. CVPR 2022 (DKD/TCKD); Yin et al. CVPR 2022 "
             "(A-ViT adaptive tokens); Abnar & Zuidema ACL 2020 (rollout); Son et al. ECCV 2024 (MaskedKD)",
}

KS = (12, 29, 49)
LO, MIDK, HI = 12, 29, 49
EPS = 1e-8
TEACHER_DIM = 768
NBOOT = 2000


def _tckd_kl(full_lp, mask_lp, y):
    pf, pm = full_lp.exp(), mask_lp.exp()
    kl = (pf * (full_lp - mask_lp)).sum(-1)
    pt_f = pf.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    pt_m = pm.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()
    return tckd, kl


def _cost(t):
    return 12 * (12 * t * TEACHER_DIM * TEACHER_DIM + 2 * t * t * TEACHER_DIM) / 1e9


def _lateroll(api, S, x):
    depth = len(S.blocks)
    with torch.no_grad():
        out = api.student_forward(S, x, attn_layers=list(range(6, depth)))
        R = None
        for s in range(depth - 1, 5, -1):
            A = out.attn[s].float().mean(1)
            N = A.shape[-1]
            a = 0.5 * A + 0.5 * torch.eye(N, device=x.device)
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else R @ a
        return R[:, 0, 1:].float(), out.logits.float()


def run(api):
    runs = api.runs()
    common_id = next(rid for rid, r in sorted(runs.items()) if r.get("dataset") == "cub" and r["tier"] == "final"
                     and str(r["method"]).split("@")[0] == "full" and r["seed"] == 1)
    S = api.student(common_id)
    T = api.teacher.eval()
    rows = {f"tckd{k}": [] for k in KS} | {f"kl{k}": [] for k in KS} | {s: [] for s in ("ent", "conf", "conc",
                                                                                         "margin")}
    for x, y in api.val_batches(batch=50):
        sc, logits = _lateroll(api, S, x)
        p = logits.softmax(-1)
        top2 = p.topk(2, dim=-1).values
        rows["ent"].append(-(p * (p + EPS).log()).sum(-1))
        rows["conf"].append(top2[:, 0])
        rows["margin"].append(top2[:, 0] - top2[:, 1])
        mass = sc / sc.sum(-1, keepdim=True)
        rows["conc"].append(mass.topk(LO, dim=-1).values.sum(-1))
        order = sc.argsort(dim=-1, descending=True)
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
            for k in KS:
                keep = torch.zeros_like(sc, dtype=torch.bool).scatter_(1, order[:, :k], True)
                lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                t, kl = _tckd_kl(full, lm, y)
                rows[f"tckd{k}"].append(t)
                rows[f"kl{k}"].append(kl)
    d = {k: torch.cat(v).float().cpu().numpy().astype(numpy.float64) for k, v in rows.items()}
    n = len(d["tckd12"])
    gen = numpy.random.default_rng(20261031)
    frac = (MIDK - LO) / (HI - LO)
    n_hi = int(math.floor(frac * n))
    needs = {"ent": d["ent"], "neg_conf": -d["conf"], "neg_conc": -d["conc"], "neg_margin": -d["margin"],
             "oracle": d["tckd12"] - d["tckd49"], "random": gen.random(n)}
    out = {"common_student": common_id, "n_images": n, "n_hi": n_hi, "mean_k_two_level": (n_hi * HI + (n - n_hi) * LO) / n,
           "gflops_teacher_uniform29": round(_cost(MIDK + 1), 4),
           "gflops_teacher_two_level": round((n_hi * _cost(HI + 1) + (n - n_hi) * _cost(LO + 1)) / n, 4),
           "uniform": {f"tckd{k}": round(float(d[f"tckd{k}"].mean()), 4) for k in KS}
           | {f"kl{k}": round(float(d[f"kl{k}"].mean()), 4) for k in KS}}
    sp = {}
    for s in ("ent", "conf", "conc", "margin"):
        a = d[s]
        ra, rb = numpy.argsort(numpy.argsort(a)), numpy.argsort(numpy.argsort(d["tckd12"]))
        gain = d["tckd12"] - d["tckd49"]
        rg = numpy.argsort(numpy.argsort(gain))
        sp[s] = {"spearman_vs_tckd12": round(float(numpy.corrcoef(ra, rb)[0, 1]), 4),
                 "spearman_vs_gain12to49": round(float(numpy.corrcoef(ra, rg)[0, 1]), 4)}
    out["signal_spearman"] = sp
    idx = gen.integers(0, n, size=(NBOOT, n))
    alloc = {}
    for name, need in needs.items():
        hi = numpy.zeros(n, dtype=bool)
        hi[numpy.argsort(-need, kind="stable")[:n_hi]] = True
        res = {}
        for m in ("tckd", "kl"):
            pol = numpy.where(hi, d[f"{m}{HI}"], d[f"{m}{LO}"])
            diff = pol - d[f"{m}{MIDK}"]
            boot = diff[idx].mean(1)
            res[m] = {"policy_mean": round(float(pol.mean()), 4),
                      "minus_uniform29": [round(float(diff.mean()), 4), round(float(numpy.percentile(boot, 2.5)), 4),
                                          round(float(numpy.percentile(boot, 97.5)), 4)]}
        alloc[name] = res
    out["two_level_vs_uniform29"] = alloc
    api.write_json("summary", out)
