"""H-A183 (round 447, analysis only, no training): does the per-image fidelity of the MaskedKD@98 masked teacher track
where the MaskedKD@98 student falls behind the Full KD student on Caltech-256? The gate failed by a small margin
(MaskedKD@98 83.81 vs Full 84.35). H-A180-182 measured masked-teacher fidelity only; none linked it to the students.

Per val image i (clean view):
  F_i  = KL(full teacher || keep-only masked teacher), mask = maskedkd rule (last-block CLS attention) of the
         maskedkd@98 student itself, K=98 (the mask that student's own rule gives at the end of training).
  G_mk = KL(teacher || maskedkd@98 student) - KL(teacher || full@196 student)
  G_ce = KL(teacher || ce student) - KL(teacher || full@196 student)   (negative control: image difficulty)
P1: Spearman(F, G_mk), 95% percentile bootstrap over val images (numpy default_rng(20261206), 2000 resamples).
P2: Spearman(F, G_mk) - Spearman(F, G_ce), same resample indices.
Record: same with F from the full student's maskedkd mask and from the maskedkd@98 student's lateroll mask,
F-tercile accuracy gaps (full - maskedkd98, full - ce), means. Ranks within a resample break ties by order.
Limit: one seed per student, clean val views, end-of-training mask (not the masks seen during training), correlation.
"""
import numpy
import torch

META = {
    "question": "on Caltech-256 val, are images whose MaskedKD@98 masked teacher is far from the full teacher the "
                "images where the MaskedKD@98 student is farther from the teacher than the Full KD student, beyond "
                "what image difficulty (CE student gap) explains?",
    "method": "per-image keep-only masked-teacher KL vs per-image student-teacher KL gap, Spearman with image "
              "bootstrap and a CE-student negative control",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL "
             "2020 (attention rollout); Hinton et al. 2015 (KD); Efron & Tibshirani 1993 (bootstrap)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261206
K = 98
STUDENTS = {
    "full": "caltech256_final_full_k196_s1_1103_123120_d6de",
    "maskedkd98": "caltech256_final_maskedkd_k98_s1_1103_123120_9e43",
    "ce": "caltech256_final_ce_k0_s1_1103_123120_2db8",
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


def _rank(a):
    return a.argsort(-1).argsort(-1).astype(numpy.float64)


def _spear(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(-1, keepdims=True)
    rb = rb - rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / numpy.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))


def _ci(point, bs):
    return {"mean": round(float(point), 4),
            "ci": [round(float(numpy.percentile(bs, 2.5)), 4), round(float(numpy.percentile(bs, 97.5)), 4)]}


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

    skl, scor, F = {}, {}, {}
    for sname, rid in STUDENTS.items():
        S = api.student(rid)
        sl, g_mk, g_lr = [], [], []
        for x, _ in batches:
            with torch.no_grad():
                sl.append(api.student_forward(S, x).logits.float().cpu())
                if sname == "ce":
                    continue
                last = api.student_forward(S, x, attn_layers=[len(S.blocks) - 1]).attn[-1].float().mean(1)[:, 0, 1:]
                g_mk.append(api.teacher_forward_keepmask(Tm, x, _topk_mask(last, K)).float().cpu())
                out = api.student_forward(S, x, attn_layers=list(range(6, len(S.blocks))))
                lr = _rollout([a for a in out.attn if a is not None])
                g_lr.append(api.teacher_forward_keepmask(Tm, x, _topk_mask(lr, K)).float().cpu())
        del S
        torch.cuda.empty_cache()
        slp = _logp(torch.cat(sl).numpy().astype(numpy.float64))
        skl[sname] = (tp * (tlp - slp)).sum(1)
        scor[sname] = (slp.argmax(1) == y).astype(numpy.float64) * 100
        for tag, v in (("maskedkd", g_mk), ("lateroll", g_lr)):
            if v:
                glp = _logp(torch.cat(v).numpy().astype(numpy.float64))
                F[(sname, tag)] = (tp * (tlp - glp)).sum(1)

    g_mk = skl["maskedkd98"] - skl["full"]
    g_ce = skl["ce"] - skl["full"]
    rng = numpy.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))

    def corr(f, g):
        return _ci(_spear(f, g), _spear(f[idx], g[idx]))

    f0 = F[("maskedkd98", "maskedkd")]
    p1 = corr(f0, g_mk)
    p2 = _ci(_spear(f0, g_mk) - _spear(f0, g_ce),
             _spear(f0[idx], g_mk[idx]) - _spear(f0[idx], g_ce[idx]))

    def terc(f):
        cut = numpy.percentile(f, [100 / 3, 200 / 3])
        lo, hi = f <= cut[0], f > cut[1]
        out = {}
        for nm, other in (("full-maskedkd98", "maskedkd98"), ("full-ce", "ce")):
            dacc = scor["full"] - scor[other]
            out[nm] = {"bottom_third": round(float(dacc[lo].mean()), 3), "top_third": round(float(dacc[hi].mean()), 3)}
        return out

    rec = {}
    for (s, tag), f in F.items():
        rec[f"F_{s}_{tag}"] = {
            "F_mean": round(float(f.mean()), 4),
            "spearman_F_Gmk": corr(f, g_mk), "spearman_F_Gce": corr(f, g_ce),
            "tercile_acc_gap": terc(f)}
    api.write_json("summary", {
        "n_val": int(n), "K": K, "teacher_acc": round(float((tlp.argmax(1) == y).mean() * 100), 3),
        "student_acc_check": {s: [round(float(scor[s].mean()), 3), runs[STUDENTS[s]]["val_acc_last"]] for s in STUDENTS},
        "student_teacher_kl_mean": {s: round(float(v.mean()), 4) for s, v in skl.items()},
        "G_mk_mean": round(float(g_mk.mean()), 4), "G_ce_mean": round(float(g_ce.mean()), 4),
        "P1_spearman_F_maskedkd98own_vs_Gmk": p1,
        "P2_spearman_Gmk_minus_Gce": p2,
        "record": rec,
        "note": "F = KL(full teacher || keep-only masked teacher, K=98), G = KL(teacher||student) - KL(teacher||full "
                "student), T=1, nats; 95% percentile bootstrap over val images (2000 resamples); seed-1 students only"})
