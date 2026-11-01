"""H-C17: the KD gradient w.r.t. student logits is linear in the teacher's probabilities (TAU = 1:
d KD / d z_s = p_s - p_t), so over the ~100 epochs in which one image comes back, the student is pulled toward the
*average* masked-teacher distribution, not toward any single masked output. Deterministic top-K repeats the same
bias every visit; stochastic top-K (Gumbel / Plackett-Luce sampling from the student's own lateroll map) could make the
averaged target closer to the full teacher. Does it, on CUB?

Student: CUB final full-KD s1 (common student), epoch 50 (descriptive) and final (verdict).
Score: roll6 = student rollout over blocks 6..11 (harness `lateroll`), normalised per image.
Conditions per K in (49, 29, 12):
  det        roll6 top-K (one mask)
  g{tau}     M masks, Gumbel top-K with keys log(roll6)/tau + Gumbel, tau in (0.5, 1, 2); tau=1 is primary
  random     M uniform random K-subsets (control: random@29 screen accuracy is 74.00, far below lateroll 81.33)
For each stochastic condition: bias = KL(full || mean_m p_masked_m) and TCKD of that mean (what the averaged gradient
sees), and single = mean_m KL(full || p_masked_m) (what one step sees). For det, bias = single.
Paired diffs vs det with image bootstrap 95% CI (2000 draws). All noise (Gumbel, random subsets, bootstrap) comes from
numpy default_rng on the analysis side only. No position prior: every score is a per-image student map.
Limitation: val images are not augmented; in training, crops/flips/Mixup already vary the deterministic mask.
"""
import numpy
import torch

META = {
    "question": "on CUB, does averaging the masked teacher over stochastic (Gumbel top-K from the student's lateroll "
                "map) masks give a KD target closer to the full teacher than the deterministic top-K, and does the "
                "same averaging over uniform random masks stay worse (metric sanity)?",
    "method": "attention rollout (blocks 6..11); Gumbel top-K / Plackett-Luce sampling; keep-only masked teacher "
              "faithfulness (KL, DKD TCKD) of the averaged distribution; image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Kool et al. ICML 2019 (Gumbel top-k); Hinton et al. 2015 (KD "
             "gradient p_s - p_t); Zhao et al. CVPR 2022 (DKD); Son et al. ECCV 2024 (MaskedKD)",
}

KS = (49, 29, 12)
TAUS = (0.5, 1.0, 2.0)
M = 16
SPLIT = 6
EPS = 1e-8
NBOOT = 2000
BATCH = 50
STOCH = tuple(f"g{t}" for t in TAUS) + ("random",)


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _roll6(api, S, x):
    depth = len(S.blocks)
    with torch.no_grad():
        out = api.student_forward(S, x, attn_layers=list(range(SPLIT, depth)))
        A = [a.float().mean(1) for a in out.attn]
        B, N, _ = A[0].shape
        eye = torch.eye(N, device=A[0].device).expand(B, N, N)
        R = None
        for a in reversed(A):
            a = 0.5 * a + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else R @ a
    r = R[:, 0, 1:].float().clamp(min=EPS)
    return r / r.sum(1, keepdim=True)


def _kl_tckd(full_lp, mask_p, y):
    pf = full_lp.exp()
    mask_p = mask_p.clamp(min=EPS)
    kl = (pf * (full_lp - mask_p.log())).sum(-1)
    pt_f = pf.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    pt_m = mask_p.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()
    return kl, tckd


def _ci(v, gen):
    n = len(v)
    bs = numpy.array([v[gen.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return [round(float(v.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4)]


def run(api):
    runs = api.runs()
    rid = None
    for r_id, r in sorted(runs.items()):
        if r.get("dataset") == "cub" and r["tier"] == "final" and str(r["method"]).split("@")[0] == "full" \
                and r["seed"] == 1:
            rid = r_id
    snaps = [e for e in (50,) if e in (runs[rid].get("snapshots") or [])]
    epochs = snaps + ["final"]
    api.write_json("students", {"run": rid, "epochs": epochs})
    T = api.teacher.eval()
    n = api.num_patches
    gen = numpy.random.default_rng(20261101)

    fulls, n_img = [], 0
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            fulls.append(T(x).float().log_softmax(-1))
        n_img += x.shape[0]

    summary = {}
    for ep in epochs:
        S = api.student(rid, epoch=None if ep == "final" else ep)
        g = numpy.random.default_rng(20261101)
        per = {k: {c: {"bias_kl": [], "bias_tckd": [], "single_kl": [], "ov_det": []} for c in ("det",) + STOCH}
               for k in KS}
        for b, (x, y) in enumerate(api.val_batches(batch=BATCH)):
            B = x.shape[0]
            r6 = _roll6(api, S, x)
            full = fulls[b]
            logr = r6.log()
            for k in KS:
                det = _topk(r6, k)
                with torch.no_grad():
                    p_det = api.teacher_forward_keepmask(T, x, det).float().softmax(-1)
                kl, tc = _kl_tckd(full, p_det, y)
                d = per[k]["det"]
                d["bias_kl"].append(kl.cpu()); d["bias_tckd"].append(tc.cpu())
                d["single_kl"].append(kl.cpu()); d["ov_det"].append(torch.ones(B))
                for c in STOCH:
                    psum = torch.zeros_like(p_det)
                    ksum = torch.zeros(B, device=x.device)
                    osum = torch.zeros(B, device=x.device)
                    for _ in range(M):
                        u = torch.from_numpy(g.random((B, n))).float().to(x.device).clamp(EPS, 1 - EPS)
                        if c == "random":
                            keys = u
                        else:
                            tau = float(c[1:])
                            keys = logr / tau - (-u.log()).log()
                        keep = _topk(keys, k)
                        with torch.no_grad():
                            p = api.teacher_forward_keepmask(T, x, keep).float().softmax(-1)
                        psum += p
                        ksum += _kl_tckd(full, p, y)[0]
                        osum += (keep & det).sum(1).float() / k
                    kl, tc = _kl_tckd(full, psum / M, y)
                    s = per[k][c]
                    s["bias_kl"].append(kl.cpu()); s["bias_tckd"].append(tc.cpu())
                    s["single_kl"].append((ksum / M).cpu()); s["ov_det"].append((osum / M).cpu())
        res = {}
        for k in KS:
            v = {c: {m: torch.cat(per[k][c][m]).numpy().astype(numpy.float64) for m in per[k][c]}
                 for c in per[k]}
            r = {"mean": {c: {m: round(float(a.mean()), 4) for m, a in v[c].items()} for c in v}}
            for c in STOCH:
                r[f"d_bias_kl_{c}_minus_det"] = _ci(v[c]["bias_kl"] - v["det"]["bias_kl"], gen)
                r[f"d_bias_tckd_{c}_minus_det"] = _ci(v[c]["bias_tckd"] - v["det"]["bias_tckd"], gen)
                r[f"d_single_kl_{c}_minus_det"] = _ci(v[c]["single_kl"] - v["det"]["single_kl"], gen)
            res[str(k)] = r
        summary[str(ep)] = res
        api.write_json(f"average_e{ep}", res)

    fin = summary["final"]
    verdict = {
        "P1_g1_bias_kl_ci_hi_lt_0_at_29_and_12":
            all(fin[str(k)]["d_bias_kl_g1.0_minus_det"][2] < 0 for k in (29, 12)),
        "P2_g1_gain_at_12_larger_than_at_49":
            fin["12"]["d_bias_kl_g1.0_minus_det"][0] < fin["49"]["d_bias_kl_g1.0_minus_det"][0],
        "P3_random_bias_kl_ci_lo_gt_0_at_29": fin["29"]["d_bias_kl_random_minus_det"][1] > 0,
        "D_best_bias_kl_at_each_k":
            {str(k): min(fin[str(k)]["mean"], key=lambda c: fin[str(k)]["mean"][c]["bias_kl"]) for k in KS},
    }
    api.write_json("summary", {"students": {"run": rid, "epochs": epochs}, "n_images": n_img, "M": M,
                               "average": summary, "verdict": verdict})
