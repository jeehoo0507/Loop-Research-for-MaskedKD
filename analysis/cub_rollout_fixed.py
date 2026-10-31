"""H-C13: why does rollout over student blocks 6..11 (lateroll) keep the masked teacher's soft label better than full
rollout (blocks 0..11)? Hypothesis: the early blocks add an image-independent component (the same token ranking for
every image, i.e. a learned fixed spatial pattern), and following that fixed pattern costs teacher fidelity.

Common student (CUB final full-KD s1) plus the six K=49 screen students of lateroll and rollout (seeds 0, 4, 5).
Scores from the student's own head-averaged attention (0.5 A + 0.5 I, row-normalised):
  roll0 = blocks 0..11 (harness rollout), roll6 = blocks 6..11 (lateroll), early = blocks 0..5 only.
Fixed mask fixed@K = top-K of the dataset-mean of the per-image (sum-normalised) roll0 map: one mask for all images.
Per student, K in (12, 29, 49), score in (roll0, roll6, early): mean overlap |topK & fixed@K| / K, and mean per-image
Spearman of the score map with the mean roll0 map.
Common student only: masked-teacher TCKD / KL for roll0, roll6, early, fixed@K; per image
d_tckd = TCKD(roll0) - TCKD(roll6) and d_ov = overlap(roll0, fixed) - overlap(roll6, fixed); Spearman(d_tckd, d_ov)
and paired means with image bootstrap 95% CI (2000 draws, numpy default_rng 20261031).
Position enters only the analysis-side fixed mask (built from the student's own maps), never a selector.
"""
import numpy
import torch

META = {
    "question": "on CUB, does full rollout carry an image-independent token ranking from the student's early blocks "
                "that late-block rollout drops, and does following it cost masked-teacher fidelity (TCKD)?",
    "method": "attention rollout split by block range; dataset-mean map as the fixed component; keep-only masked "
              "teacher faithfulness (KL, DKD TCKD); image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Zhao et al. CVPR 2022 (DKD); Son et al. ECCV 2024 (MaskedKD)",
}

KS = (12, 29, 49)
SPLIT = 6
EPS = 1e-8
NBOOT = 2000


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _scores(api, S, x):
    depth = len(S.blocks)
    with torch.no_grad():
        out = api.student_forward(S, x, attn_layers=list(range(depth)))
        A = [a.float().mean(1) for a in out.attn]
        B, N, _ = A[0].shape
        eye = torch.eye(N, device=A[0].device).expand(B, N, N)
        R, E, r6 = None, None, None
        for s in range(depth - 1, -1, -1):
            a = 0.5 * A[s] + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else R @ a
            if s == SPLIT:
                r6 = R[:, 0, 1:].clone()
            if s < SPLIT:
                E = a if E is None else E @ a
    return {"roll0": R[:, 0, 1:].float(), "roll6": r6.float(), "early": E[:, 0, 1:].float()}


def _rank(v):
    return numpy.argsort(numpy.argsort(v, axis=-1), axis=-1).astype(numpy.float64)


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(-1, keepdims=True)
    rb -= rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / numpy.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))


def _stats(full_lp, mask_lp, y):
    pf, pm = full_lp.exp(), mask_lp.exp()
    kl = (pf * (full_lp - mask_lp)).sum(-1)
    pt_f = pf.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    pt_m = pm.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()
    return kl, tckd


def _ci(v, gen):
    n = len(v)
    bs = numpy.array([v[gen.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return [round(float(v.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4)]


def run(api):
    runs = api.runs()
    studs = {}
    for rid, r in sorted(runs.items()):
        if r.get("dataset") != "cub":
            continue
        m = str(r["method"]).split("@")[0]
        if r["tier"] == "final" and m == "full" and r["seed"] == 1:
            studs["common_full196_s1"] = rid
        if r["tier"] == "screen" and m in ("lateroll", "rollout") and int(r.get("keep") or 0) == 49 \
                and r["seed"] in (0, 4, 5):
            studs[f"{m}49_s{r['seed']}"] = rid
    api.write_json("students", studs)
    n = api.num_patches

    def collect(S):
        out = {k: [] for k in ("roll0", "roll6", "early")}
        for x, _ in api.val_batches(batch=50):
            for k, v in _scores(api, S, x).items():
                out[k].append(v.cpu())
        return {k: torch.cat(v) for k, v in out.items()}

    desc, common = {}, None
    for name, rid in studs.items():
        sc = collect(api.student(rid))
        mean_map = (sc["roll0"] / sc["roll0"].sum(1, keepdim=True)).mean(0)
        res = {}
        for k in KS:
            fixed = _topk(mean_map[None], k)[0]
            for s, v in sc.items():
                res[f"overlap_fixed_{s}@{k}"] = round(((_topk(v, k) & fixed).sum(1).float() / k).mean().item(), 4)
        for s, v in sc.items():
            rho = _spearman(v.numpy().astype(numpy.float64), mean_map.numpy()[None].astype(numpy.float64))
            res[f"spearman_meanmap_{s}"] = round(float(rho.mean()), 4)
        desc[name] = res
        if name == "common_full196_s1":
            common = (sc, mean_map)
    api.write_json("fixed_component", desc)

    sc, mean_map = common
    T = api.teacher.eval()
    gen = numpy.random.default_rng(20261031)
    per = {k: {m: {"kl": [], "tckd": []} for m in ("roll0", "roll6", "early", "fixed")} for k in KS}
    ovs = {k: {m: [] for m in ("roll0", "roll6")} for k in KS}
    off = 0
    for x, y in api.val_batches(batch=50):
        B = x.shape[0]
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        for k in KS:
            fixed = _topk(mean_map[None], k)[0].to(x.device)
            masks = {m: _topk(sc[m][off:off + B].to(x.device), k) for m in ("roll0", "roll6", "early")}
            masks["fixed"] = fixed.expand(B, n).clone()
            for m, keep in masks.items():
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                kl, tckd = _stats(full, lm, y)
                per[k][m]["kl"].append(kl.cpu())
                per[k][m]["tckd"].append(tckd.cpu())
            for m in ("roll0", "roll6"):
                ovs[k][m].append(((masks[m] & fixed).sum(1).float() / k).cpu())
        off += B

    fid, verdict = {}, {}
    for k in KS:
        t = {m: torch.cat(per[k][m]["tckd"]).numpy().astype(numpy.float64) for m in per[k]}
        kl = {m: torch.cat(per[k][m]["kl"]).numpy().astype(numpy.float64) for m in per[k]}
        ov = {m: torch.cat(ovs[k][m]).numpy().astype(numpy.float64) for m in ovs[k]}
        d_t = t["roll0"] - t["roll6"]
        d_o = ov["roll0"] - ov["roll6"]
        rho = float(_spearman(d_t[None], d_o[None])[0])
        idx = [gen.integers(0, len(d_t), len(d_t)) for _ in range(NBOOT)]
        bs = numpy.array([_spearman(d_t[None, i], d_o[None, i])[0] for i in idx])
        fid[str(k)] = {
            "tckd": {m: round(float(v.mean()), 4) for m, v in t.items()},
            "kl": {m: round(float(v.mean()), 4) for m, v in kl.items()},
            "d_tckd_roll0_minus_roll6": _ci(d_t, gen),
            "d_overlap_roll0_minus_roll6": _ci(d_o, gen),
            "spearman_dtckd_doverlap": [round(rho, 4), round(float(numpy.quantile(bs, 0.025)), 4),
                                        round(float(numpy.quantile(bs, 0.975)), 4)],
        }
        c = desc["common_full196_s1"]
        verdict[str(k)] = {
            "P1_overlap_roll0_minus_roll6_ge_0.05": c[f"overlap_fixed_roll0@{k}"] - c[f"overlap_fixed_roll6@{k}"] >= 0.05,
            "P1_overlap_early_ge_roll0": c[f"overlap_fixed_early@{k}"] >= c[f"overlap_fixed_roll0@{k}"],
            "P2_spearman_ci_lo_gt_0_and_rho_ge_0.1": bool(fid[str(k)]["spearman_dtckd_doverlap"][1] > 0 and rho >= 0.1),
        }
    api.write_json("fidelity_common", fid)
    api.write_json("summary", {"students": studs, "fixed_component": desc, "fidelity_common": fid,
                               "verdict": verdict})
