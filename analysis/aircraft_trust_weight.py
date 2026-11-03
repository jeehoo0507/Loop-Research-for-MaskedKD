"""H-A27 (QUEUE 2026-11-03 17:25, TrustMask v1 diagnosis): the cached teacher map as a trust estimator instead of a
filter. H-A26 showed the filter (trustfilt) makes the masked teacher less faithful than student-only picking at every
epoch, and that at K = 29 the masked teacher's top-1 agrees with the full teacher on only ~69% of val images. So the
weak point of v0 is that the KD target is often wrong, and the teacher map is spent where it hurts.
Question: before the teacher forward, can a free or amortized per-image signal tell how faithful the masked teacher
will be for this image (so its KD term can be weighted down when it is not)?

Students: Aircraft final lateroll@29 seeds 1, 2, 3 at epoch 25, 50 and final (the token rule v1 would keep).
Mask: S = student late-block rollout top-K (blocks 6..11, lateroll), K = 29.
Per-image signals (all computable inside a selector before the teacher forward):
  cov   teacher-map mass inside S (ctx.teacher_map; amortized, interpretation based)
  ov    |S & T| / K, T = teacher-map top-K (interpretation based)
  smass student rollout mass inside S (free, interpretation based: how concentrated the student's evidence is)
  ent   student predictive entropy (free, not interpretation based; the ent_weight baseline's signal)
Targets: fid = KL(teacher full || teacher masked S); agree = top-1 of masked equals top-1 of full.
Weight proxy: with w_i = signal_i / batch mean (mean 1, as a selector's kd_weight would be), wfid = sum w fid / sum w
(lower = KD weight moved to images whose masked target is faithful); for ent the weight is 1/ent normalised the same
way (low entropy = trusted) so every signal is oriented "high = trust".
Seed mean per image over the 3 students, image bootstrap 95% CI (2000 draws, numpy default_rng 20261103).
No position prior: every score is a per-image map.
"""
import math

import numpy
import torch

META = {
    "question": "on Aircraft K=29 with lateroll students, does teacher-map mass inside the kept tokens (or overlap, "
                "student mass, entropy) predict per image how faithful the masked teacher will be, before the "
                "teacher forward?",
    "method": "attention rollout (student late blocks, teacher all blocks); keep-only masked teacher faithfulness "
              "(KL, top-1 agreement); per-image Spearman with image bootstrap; weighted-fidelity proxy",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Hooker et al. NeurIPS 2019 (keep-only); Son et al. ECCV 2024 "
             "(MaskedKD); EGAD arXiv 2605.01732 (entropy-weighted distillation)",
}

K = 29
SPLIT = 6
NBOOT = 2000
BATCH = 50
EPOCHS = (25, 50, "final")
SIGNALS = ("cov", "ov", "smass", "ent")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _rollout(api, M, x, start):
    depth = len(M.blocks)
    with torch.no_grad():
        out = api.student_forward(M, x, attn_layers=list(range(depth)))
    r = None
    for a in out.attn[start:]:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    m = r[:, 0, 1:]
    return m / m.sum(1, keepdim=True), out.logits.float().log_softmax(-1)


def _ci(v, gen):
    n = len(v)
    bs = numpy.array([v[gen.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return [round(float(v.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4)]


def _sp(a, b):
    ra = numpy.argsort(numpy.argsort(a)).astype(float)
    rb = numpy.argsort(numpy.argsort(b)).astype(float)
    return float(numpy.corrcoef(ra, rb)[0, 1])


def _spear_ci(a, b, gen):
    n = len(a)
    bs = numpy.array([_sp(a[i], b[i]) for i in (gen.integers(0, n, n) for _ in range(NBOOT))])
    return [round(_sp(a, b), 4), round(float(numpy.quantile(bs, 0.025)), 4), round(float(numpy.quantile(bs, 0.975)), 4)]


def _spear_diff_ci(a1, a2, b, gen):
    """|rho(a1, b)| - |rho(a2, b)| with the same bootstrap draw for both."""
    n = len(b)
    bs = []
    for _ in range(NBOOT):
        i = gen.integers(0, n, n)
        bs.append(abs(_sp(a1[i], b[i])) - abs(_sp(a2[i], b[i])))
    bs = numpy.array(bs)
    return [round(abs(_sp(a1, b)) - abs(_sp(a2, b)), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4)]


def run(api):
    runs = api.runs()
    studs = {}
    for rid, r in sorted(runs.items()):
        if (r.get("dataset") == "aircraft" and r["tier"] == "final" and r["method"] == "lateroll"
                and r.get("keep") == K and r["seed"] in (1, 2, 3)):
            studs[r["seed"]] = rid
    api.write_json("students", {str(s): [rid, runs[rid]["val_acc_last"], runs[rid]["snapshots"]]
                                for s, rid in sorted(studs.items())})
    T = api.teacher.eval()

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        tm, _ = _rollout(api, T, x, 0)
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        batches.append((x, y, tm, full))

    per = {str(e): {} for e in EPOCHS}
    wraw = {str(e): {} for e in EPOCHS}     # per-batch normalised weights, per seed
    for seed, rid in sorted(studs.items()):
        for ep in EPOCHS:
            S = api.student(rid, epoch=None if ep == "final" else ep)
            acc, wacc = {}, {}
            for x, y, tm, full in batches:
                s, slp = _rollout(api, S, x, SPLIT)
                keep = _topk(s, K)
                ent = -(slp.exp() * slp).sum(-1)
                sig = {"cov": (tm * keep).sum(1), "ov": (keep & _topk(tm, K)).sum(1).float() / K,
                       "smass": (s * keep).sum(1), "ent": ent}
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                rows = dict(sig)
                rows["fid"] = (full.exp() * (full - lm)).sum(-1)
                rows["agree"] = (lm.argmax(-1) == full.argmax(-1)).float()
                rows["s_acc"] = (slp.argmax(-1) == y).float()
                for kname, v in rows.items():
                    acc.setdefault(kname, []).append(v.float().cpu())
                for kname, v in sig.items():
                    t = 1.0 / v.clamp_min(1e-6) if kname == "ent" else v
                    wacc.setdefault(kname, []).append((t / t.mean().clamp_min(1e-12)).float().cpu())
            for kname, v in acc.items():
                per[str(ep)].setdefault(kname, []).append(torch.cat(v).numpy())
            for kname, v in wacc.items():
                wraw[str(ep)].setdefault(kname, []).append(torch.cat(v).numpy())

    gen = numpy.random.default_rng(20261103)
    out = {"students": {str(s): r for s, r in studs.items()}, "k": K,
           "n_val": int(sum(b[1].shape[0] for b in batches))}
    for ep in EPOCHS:
        d = {kname: numpy.mean(numpy.stack(v), 0) for kname, v in per[str(ep)].items()}
        e = {"mean_ci": {kname: _ci(v, gen) for kname, v in d.items()},
             "seedwise_mean": {kname: [round(float(a.mean()), 4) for a in v] for kname, v in per[str(ep)].items()}}
        e["spearman_vs_fid"] = {sg: _spear_ci(d[sg], d["fid"], gen) for sg in SIGNALS}
        e["spearman_vs_agree"] = {sg: _spear_ci(d[sg], d["agree"], gen) for sg in SIGNALS}
        e["absdiff_cov_minus_ent_vs_fid"] = _spear_diff_ci(d["cov"], d["ent"], d["fid"], gen)
        # weighted fidelity: per seed, then seed mean; minus uniform mean fid (negative = weight moved to faithful)
        wf = {}
        for sg in SIGNALS:
            per_img = numpy.mean(numpy.stack([w * f for w, f in zip(wraw[str(ep)][sg], per[str(ep)]["fid"])]), 0)
            wf[sg] = _ci(per_img - d["fid"], gen)      # mean of (w-1)*fid, w has mean 1 per batch
        e["wfid_minus_uniform"] = wf
        out[str(ep)] = e
        api.write_json(f"epoch_{ep}", e)

    fin = out["final"]
    out["verdict"] = {
        "P1_cov_predicts_fid": bool(fin["spearman_vs_fid"]["cov"][2] < 0),
        "P2_cov_beats_ent": bool(fin["absdiff_cov_minus_ent_vs_fid"][1] > 0),
        "P3_cov_weight_moves_to_faithful": bool(fin["wfid_minus_uniform"]["cov"][2] < 0),
        "P4_holds_all_epochs": bool(all(out[str(e)]["spearman_vs_fid"]["cov"][2] < 0 for e in EPOCHS)),
    }
    api.write_json("summary", out)
