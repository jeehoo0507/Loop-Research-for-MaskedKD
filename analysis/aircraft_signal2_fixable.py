"""H-A32 (QUEUE 2026-11-03 18:40 "how many?", signal 2 diagnosis before any v3 is registered): does an image's
masked-teacher error history predict where extra tokens help?
Signal 2 of the human's note gives more tokens to images whose masked teacher keeps failing (Feedback.teacher_ok)
and fewer to images it gets right. That only helps if (i) the failures persist over training (history predicts the
next failure) and (ii) they are fixable by more tokens (not images the full teacher also gets wrong).

Students: Aircraft final trustmask_v1@29 seeds 1, 2, 3 (the current TrustMask token rule: late-block rollout,
blocks 6..11). History = masked-teacher error (top-1 != label) at K=29 with the epoch-25 and epoch-50 students,
err_hist = mean of the two (0, 0.5, 1). Outcome = the final student's masked teacher at K = 19, 29, 39 (2/3, 1, 4/3
of 29, mean 29), keep-only.
Allocation proxies on the val set (per seed, val-set thirds, same mean K = 29): uniform (all 29); sig2 (highest
err_hist third -> 39, lowest -> 19, ties broken at random); sig1 (v2's signal: teacher-map entropy, most spread
third -> 39); random thirds (mean over 50 draws). Metrics: masked teacher accuracy (ok), KL(full || masked) (fid),
top-1 agreement with the full teacher (agree). Seed mean per image, image bootstrap 95% CI (2000 draws, numpy
default_rng 20261103). Also reported on the subset where the full teacher is right (closer to training images,
which the fine-tuned teacher mostly gets right).
No position prior: every score is a per-image map or a per-image outcome.
"""
import numpy
import torch

META = {
    "question": "on Aircraft with v1 students, does the masked teacher's error history at K=29 (epochs 25, 50) "
                "predict the final masked-teacher error, and does moving tokens to high-error images (2/3, 1, 4/3 K "
                "thirds, mean K fixed) raise masked-teacher accuracy and fidelity over uniform and random thirds?",
    "method": "attention rollout (student late blocks, teacher all blocks); keep-only masked teacher accuracy and "
              "KL at three budgets; budget-neutral reallocation proxy; image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Hooker et al. NeurIPS 2019 (keep-only); Son et al. ECCV 2024 "
             "(MaskedKD); human TrustMask note 2026-10-28 ('how many?', signal 2)",
}

K = 29
KS = (19, 29, 39)
SPLIT = 6
NBOOT = 2000
NRAND = 50
BATCH = 50


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
    return m / m.sum(1, keepdim=True)


def _ci(v, gen):
    n = len(v)
    bs = numpy.array([v[gen.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return [round(float(v.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4)]


def _sp(a, b):
    ra = numpy.argsort(numpy.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = numpy.argsort(numpy.argsort(b, kind="stable"), kind="stable").astype(float)
    return float(numpy.corrcoef(ra, rb)[0, 1])


def _spear_ci(a, b, gen):
    n = len(a)
    bs = numpy.array([_sp(a[i], b[i]) for i in (gen.integers(0, n, n) for _ in range(NBOOT))])
    return [round(_sp(a, b), 4), round(float(numpy.quantile(bs, 0.025)), 4), round(float(numpy.quantile(bs, 0.975)), 4)]


def _thirds(score, gen):
    """index into KS per image: lowest third of score -> 0 (19), middle -> 1, highest -> 2 (39); random ties."""
    n = len(score)
    order = numpy.lexsort((gen.random(n), score))
    g = numpy.empty(n, dtype=int)
    g[order] = numpy.minimum(numpy.arange(n) * 3 // n, 2)
    return g


def run(api):
    runs = api.runs()
    studs = {}
    for rid, r in sorted(runs.items()):
        if (r.get("dataset") == "aircraft" and r["tier"] == "final" and r["method"] == "trustmask_v1"
                and r.get("keep") == K and r["seed"] in (1, 2, 3)):
            studs[r["seed"]] = rid
    api.write_json("students", {str(s): [rid, runs[rid]["val_acc_last"], runs[rid]["snapshots"]]
                                for s, rid in sorted(studs.items())})
    T = api.teacher.eval()

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        tm = _rollout(api, T, x, 0)
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        batches.append((x, y, tm, full))
    full_ok = torch.cat([(f.argmax(-1) == y).float().cpu() for _, y, _, f in batches]).numpy()
    tent = torch.cat([(-(tm.clamp_min(1e-12) * tm.clamp_min(1e-12).log()).sum(1)).float().cpu()
                      for _, _, tm, _ in batches]).numpy()

    gen = numpy.random.default_rng(20261103)
    seeds = {}
    for seed, rid in sorted(studs.items()):
        rec = {}
        for ep in (25, 50, "final"):
            S = api.student(rid, epoch=None if ep == "final" else ep)
            ks = KS if ep == "final" else (K,)
            cols = {}
            for x, y, tm, full in batches:
                s = _rollout(api, S, x, SPLIT)
                for k in ks:
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, _topk(s, k)).float().log_softmax(-1)
                    cols.setdefault(("ok", k), []).append((lm.argmax(-1) == y).float().cpu())
                    cols.setdefault(("fid", k), []).append((full.exp() * (full - lm)).sum(-1).float().cpu())
                    cols.setdefault(("agree", k), []).append((lm.argmax(-1) == full.argmax(-1)).float().cpu())
            rec[str(ep)] = {key: torch.cat(v).numpy() for key, v in cols.items()}
        hist = 1.0 - 0.5 * (rec["25"][("ok", K)] + rec["50"][("ok", K)])
        fin = rec["final"]
        alloc = {"sig2": _thirds(hist, gen), "sig1": _thirds(tent, gen)}
        res = {"hist": hist, "fin": fin}
        for name, g in alloc.items():
            res[name] = {m: numpy.choose(g, [fin[(m, k)] for k in KS]) for m in ("ok", "fid", "agree")}
        rnd = {m: numpy.zeros_like(fin[("ok", K)]) for m in ("ok", "fid", "agree")}
        for _ in range(NRAND):
            g = _thirds(gen.random(len(hist)), gen)
            for m in rnd:
                rnd[m] += numpy.choose(g, [fin[(m, k)] for k in KS]) / NRAND
        res["random"] = rnd
        res["g_sig2"] = alloc["sig2"]
        seeds[seed] = res

    def smean(f):
        return numpy.mean(numpy.stack([f(r) for r in seeds.values()]), 0)

    hist = smean(lambda r: r["hist"])
    err_fin = smean(lambda r: 1.0 - r["fin"][("ok", K)])
    out = {"students": {str(s): r for s, r in studs.items()}, "n_val": int(len(full_ok)),
           "full_teacher_acc": round(float(full_ok.mean()), 4)}
    subsets = {"all": numpy.ones_like(full_ok, dtype=bool), "full_right": full_ok > 0.5}
    for sname, msk in subsets.items():
        e = {"n": int(msk.sum())}
        e["spearman_hist_vs_final_err"] = _spear_ci(hist[msk], err_fin[msk], gen)
        # per-seed hist is discrete (0, 0.5, 1); report seed-pooled conditional error
        pooled = {}
        for h in (0.0, 0.5, 1.0):
            num = sum(float(((1.0 - r["fin"][("ok", K)]) * ((r["hist"] == h) & msk)).sum()) for r in seeds.values())
            den = sum(int(((r["hist"] == h) & msk).sum()) for r in seeds.values())
            pooled[str(h)] = [round(num / den, 4) if den else None, den]
        e["final_err_by_hist_pooled"] = pooled
        e["level"] = {f"{m}@{k}": _ci(smean(lambda r, m=m, k=k: r["fin"][(m, k)])[msk], gen)
                      for m in ("ok", "fid", "agree") for k in KS}
        for name in ("sig2", "sig1", "random"):
            for m in ("ok", "fid", "agree"):
                d = smean(lambda r, name=name, m=m: r[name][m] - r["fin"][(m, K)])
                e[f"{name}_minus_uniform_{m}"] = _ci(d[msk], gen)
        for m in ("ok", "fid", "agree"):
            e[f"sig2_minus_random_{m}"] = _ci(smean(lambda r, m=m: r["sig2"][m] - r["random"][m])[msk], gen)
            e[f"sig2_minus_sig1_{m}"] = _ci(smean(lambda r, m=m: r["sig2"][m] - r["sig1"][m])[msk], gen)
        # where the sig2 tokens go: gain in the top third, loss in the bottom third (pooled over seeds)
        top = [(r["fin"][("ok", 39)] - r["fin"][("ok", K)])[(r["g_sig2"] == 2) & msk] for r in seeds.values()]
        bot = [(r["fin"][("ok", K)] - r["fin"][("ok", 19)])[(r["g_sig2"] == 0) & msk] for r in seeds.values()]
        e["sig2_top_gain_ok_39_minus_29"] = round(float(numpy.concatenate(top).mean()), 4)
        e["sig2_bottom_loss_ok_29_minus_19"] = round(float(numpy.concatenate(bot).mean()), 4)
        e["sig2_top_full_teacher_wrong_frac"] = round(float(numpy.mean(numpy.concatenate(
            [(1.0 - full_ok)[(r["g_sig2"] == 2) & msk] for r in seeds.values()]))), 4)
        out[sname] = e
        api.write_json(f"subset_{sname}", e)

    a = out["all"]
    out["verdict"] = {
        "P1_history_persists": bool(a["spearman_hist_vs_final_err"][1] > 0),
        "P2_sig2_beats_uniform_ok": bool(a["sig2_minus_uniform_ok"][1] > 0),
        "P3_sig2_beats_random_ok": bool(a["sig2_minus_random_ok"][1] > 0),
        "P4_sig2_beats_uniform_fid": bool(a["sig2_minus_uniform_fid"][2] < 0),
        "P2b_full_right_sig2_beats_uniform_ok": bool(out["full_right"]["sig2_minus_uniform_ok"][1] > 0),
    }
    api.write_json("summary", out)
