"""H-C5: which student-side token score predicts how much one more kept token restores the teacher's soft label?

Basis: H-C4 (analysis_cub_proxy_validation_1031_165024_bc29) found that, within the same K, the masked teacher's
TCKD to the full teacher rank-predicts CUB screen student accuracy (mean within-K Spearman -0.69, permutation
p 0.001, seed 0). So a token is "teachable" to the extent that keeping it brings the masked teacher's target-class
probability back toward the full teacher. H-C3 showed that sharpening attention (lateroll vs rollout) did not
move student accuracy, so this asks whether a non-attention, position-free student signal carries more of that
per-token value.

Ground truth (teacher-side, analysis only): for each val image and a base set S of 12 tokens, the marginal gain
  gain_j = TCKD(S) - TCKD(S + {j})   for every patch j not in S
with two bases: S = lateroll top-12 of the common student (what the best attention score already keeps) and
S = 12 random tokens (numpy default_rng). Masked teacher = keep-only forward (MaskedKD style).

Student-side candidate scores (common student = CUB final full-KD seed 1, no labels: class = student argmax):
  attn_last  last-block CLS attention (MaskedKD)
  rollout    attention rollout over blocks 0..11
  lateroll   rollout over blocks 6..11
  lens_pred  logit lens: log-softmax of head(norm(patch token)) at the student's predicted class
  lens_conf  max softmax prob of the patch token's logit-lens distribution (class-agnostic)
  gradxin    |grad of student log p(pred) w.r.t. block-0 input token . that token| (Simonyan/Shrikumar style)
  tok_norm   L2 norm of the last-block patch token
  cos_cls    cosine of the last-block patch token with the CLS token
  center     minus distance to the image center (position prior, reference only, not claimable)
Per image: Spearman(score, gain) over the candidates, and top-12 overlap with the top-12 gain tokens; averaged
over images. Partial: Spearman(score, gain residual after a linear fit of gain ranks on lateroll ranks).
"""
import math

import numpy
import torch
import torch.nn.functional as F

META = {
    "question": "on CUB, which position-free student token score best predicts the marginal TCKD restoration of the "
                "masked teacher when one token is added to a 12-token base set?",
    "method": "keep-only occlusion (add-one marginal) on the teacher vs student attention, rollout, logit lens, "
              "gradient x input and token-geometry scores; per-image Spearman and top-k overlap",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only); Zhao et al. CVPR 2022 (DKD/TCKD); nostalgebraist 2020 "
             "(logit lens); Abnar & Zuidema ACL 2020 (rollout); Shrikumar et al. ICML 2017 (gradient x input); "
             "Son et al. ECCV 2024 (MaskedKD)",
}

K = 12
SIDE = 14
EPS = 1e-8
CHUNK = 4
SCORES = ("attn_last", "rollout", "lateroll", "lens_pred", "lens_conf", "gradxin", "tok_norm", "cos_cls", "center")


def _rank(v):
    v = numpy.asarray(v, dtype=numpy.float64)
    r = numpy.empty(len(v))
    r[v.argsort(kind="stable")] = numpy.arange(len(v), dtype=numpy.float64)
    for val in numpy.unique(v):
        m = v == val
        r[m] = r[m].mean()
    return r


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _tckd(full_lp, mask_lp, y):
    pt_f = full_lp.exp().gather(-1, y[..., None])[..., 0].clamp(EPS, 1 - EPS)
    pt_m = mask_lp.exp().gather(-1, y[..., None])[..., 0].clamp(EPS, 1 - EPS)
    return pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()


def _teacher_sets(T, emb, sets):
    """emb: [N+1, D] embedded tokens of one image (CLS first, pos added). sets: Long [M, k] patch indices."""
    cls, patches = emb[:1], emb[1:]
    h = torch.cat([cls.expand(sets.shape[0], 1, -1), patches[sets]], dim=1)
    h = T.norm_pre(T.patch_drop(h))
    h = T.norm(T.blocks(h))
    return T.forward_head(h).float().log_softmax(-1)


def _student_scores(api, S, x):
    depth = len(S.blocks)
    with torch.no_grad():
        out = api.student_forward(S, x, attn_layers=list(range(depth)), token_layers=[depth - 1])
        A = [a.float().mean(1) for a in out.attn]
        B, N, _ = A[0].shape
        eye = torch.eye(N, device=x.device).expand(B, N, N)
        R, res = None, {"attn_last": A[-1][:, 0, 1:]}
        for s in range(depth - 1, -1, -1):
            a = 0.5 * A[s] + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else R @ a
            if s == 6:
                res["lateroll"] = R[:, 0, 1:].clone()
        res["rollout"] = R[:, 0, 1:]
        pred = out.logits.argmax(-1)
        tok = S.norm(out.tokens[-1])
        lens = S.head(tok[:, 1:]).float().log_softmax(-1)
        res["lens_pred"] = lens.gather(-1, pred[:, None, None].expand(-1, N - 1, 1))[..., 0]
        res["lens_conf"] = lens.max(-1).values
        res["tok_norm"] = out.tokens[-1][:, 1:].float().norm(dim=-1)
        res["cos_cls"] = F.cosine_similarity(tok[:, 1:].float(), tok[:, :1].float(), dim=-1)
    with torch.enable_grad():
        h = S._pos_embed(S.patch_embed(x))
        h = S.norm_pre(S.patch_drop(h)).detach().requires_grad_(True)
        lp = S.forward_head(S.norm(S.blocks(h))).float().log_softmax(-1)
        sel = lp.gather(1, pred[:, None]).sum()
        g, = torch.autograd.grad(sel, h)
    res["gradxin"] = (g * h.detach()).sum(-1)[:, 1:].abs().float()
    r = torch.arange(N - 1, device=x.device)
    c = (SIDE - 1) / 2
    res["center"] = -(((r // SIDE - c) ** 2 + (r % SIDE - c) ** 2).float()).expand(B, -1)
    return res


def run(api):
    runs = api.runs()
    common_id = next(rid for rid, r in sorted(runs.items()) if r.get("dataset") == "cub" and r["tier"] == "final"
                     and str(r["method"]).split("@")[0] == "full" and r["seed"] == 1)
    S = api.student(common_id)
    T = api.teacher.eval()
    n = api.num_patches
    gen = numpy.random.default_rng(20261031)
    acc = {b: {s: {"rho": [], "top12": [], "partial": []} for s in SCORES} for b in ("lateroll", "random")}
    base_tckd = {"lateroll": [], "random": []}
    best_gain = {"lateroll": [], "random": []}
    for x, y in api.val_batches(batch=CHUNK * 5):
        sc = _student_scores(api, S, x)
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
            emb = T._pos_embed(T.patch_embed(x))
        for i in range(x.shape[0]):
            bases = {"lateroll": sc["lateroll"][i].topk(K).indices,
                     "random": torch.from_numpy(gen.permutation(n)[:K]).to(x.device)}
            for bname, base in bases.items():
                inb = torch.zeros(n, dtype=torch.bool, device=x.device)
                inb[base] = True
                cand = (~inb).nonzero().flatten()
                sets = torch.cat([base.expand(len(cand), K), cand[:, None]], dim=1)
                with torch.no_grad():
                    lp0 = _teacher_sets(T, emb[i], base[None])
                    lps = torch.cat([_teacher_sets(T, emb[i], s) for s in sets.split(64)])
                    t0 = _tckd(full[i:i + 1], lp0, y[i:i + 1])[0]
                    tj = _tckd(full[i:i + 1].expand(len(cand), -1), lps, y[i:i + 1].expand(len(cand)))
                gain = (t0 - tj).float().cpu().numpy().astype(numpy.float64)
                base_tckd[bname].append(float(t0))
                best_gain[bname].append(float(gain.max()))
                rg = _rank(gain)
                rl = _rank(sc["lateroll"][i][cand].float().cpu().numpy())
                slope = _corr(rl, rg) * rg.std() / rl.std() if rl.std() > 0 else 0.0
                resid = rg - rg.mean() - slope * (rl - rl.mean())
                topg = set(numpy.argsort(-gain, kind="stable")[:K].tolist())
                for s in SCORES:
                    v = sc[s][i][cand].float().cpu().numpy()
                    rv = _rank(v)
                    acc[bname][s]["rho"].append(_corr(rv, rg))
                    acc[bname][s]["partial"].append(_corr(rv, _rank(resid)))
                    acc[bname][s]["top12"].append(len(topg & set(numpy.argsort(-v, kind="stable")[:K].tolist())) / K)
    out = {"common_student": common_id, "k_base": K, "n_images": len(base_tckd["random"])}
    for b in acc:
        out[b] = {"base_tckd_mean": round(float(numpy.mean(base_tckd[b])), 4),
                  "best_add_gain_mean": round(float(numpy.mean(best_gain[b])), 4)}
        for s in SCORES:
            d = {}
            for m, v in acc[b][s].items():
                v = numpy.asarray([u for u in v if not math.isnan(u)])
                idx = gen.integers(0, len(v), size=(2000, len(v)))
                boot = v[idx].mean(1)
                d[m] = [round(float(v.mean()), 4), round(float(numpy.percentile(boot, 2.5)), 4),
                        round(float(numpy.percentile(boot, 97.5)), 4)]
            out[b][s] = d
    api.write_json("summary", out)
