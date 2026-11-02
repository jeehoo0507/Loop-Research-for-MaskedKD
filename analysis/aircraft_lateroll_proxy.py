"""H-A1: before any Aircraft lateroll run, does the CUB proxy result of H-C2c (late-block rollout roll6 keeps the
masked teacher's soft label better than maskedkd's ranking and full rollout) also hold on FGVC-Aircraft?

Nothing here selects or tunes lateroll (selectors/lateroll.py stays as fixed on CUB; QUEUE 2026-11-02 08:50 forbids
choosing its hyperparameters on Aircraft val). This is a descriptive proxy prediction made before the Aircraft
accuracy verdict exists. Same code path as analysis/cub_lateroll_refs.py (H-C2c v2): for the three Aircraft final
seed-1 students (ce, full@196, maskedkd@98, the gate runs), at K = 29 and 12,
  1. per-image KL / TCKD of the masked teacher for roll6 (lateroll), roll0 (= harness rollout), roll11 (= maskedkd
     ranking) and the student-independent random reference;
  2. paired bootstrap (2000 resamples over val images, seeded numpy generator) 95 % CI of TCKD(roll6) - TCKD(ref)
     and of KL.
"""
import numpy
import torch

META = {
    "question": "on FGVC-Aircraft, does roll6 (late-block rollout, selectors/lateroll.py unchanged) keep the masked "
                "teacher's TCKD/KL lower than maskedkd's ranking (roll11), full rollout (roll0) and random at "
                "K = 29 and 12, with paired bootstrap CIs, as it did on CUB (H-C2c)?",
    "method": "keep-only masked-teacher faithfulness (KL, DKD TCKD) with paired bootstrap CIs; attention rollout "
              "start-block variants",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Zhao et al. CVPR 2022 (DKD); Son et al. ECCV 2024 (MaskedKD); "
             "Efron & Tibshirani 1993 (bootstrap)",
}

KS = (29, 12)
EPS = 1e-8
NBOOT = 2000
SCORES = ("roll6", "roll0", "roll11")
STUDENTS = ("ce0", "full196", "maskedkd98")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _hash(i, j):
    return torch.frac(torch.sin(i * 12.9898 + j * 78.233) * 43758.5453)


def _random(B, n, k, dev, off):
    i = torch.arange(off, off + B, device=dev, dtype=torch.float64)[:, None]
    j = torch.arange(n, device=dev, dtype=torch.float64)[None, :]
    return _topk(_hash(i, j).float(), k)


def _rollouts(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad():
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
        A = [a.float().mean(1) for a in out.attn]
        B, N, _ = A[0].shape
        eye = torch.eye(N, device=A[0].device).expand(B, N, N)
        res, R = {}, None
        for s in range(depth - 1, -1, -1):
            a = 0.5 * A[s] + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else R @ a
            if f"roll{s}" in SCORES:
                res[f"roll{s}"] = R[:, 0, 1:].clone()
    return res


def _stats(full_lp, mask_lp, y):
    pf, pm = full_lp.exp(), mask_lp.exp()
    kl = (pf * (full_lp - mask_lp)).sum(-1)
    pt_f = pf.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    pt_m = pm.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()
    return kl, tckd


def _boot_idx(N, dev):
    rng = numpy.random.default_rng(20261102)
    return torch.from_numpy(rng.integers(0, N, size=(NBOOT, N))).to(dev)    # [NBOOT, N], fixed


def _ci(d, idx):
    m = d[idx].mean(1).sort().values
    lo, hi = m[int(0.025 * NBOOT)].item(), m[int(0.975 * NBOOT) - 1].item()
    assert lo <= d.mean().item() <= hi, "bootstrap CI does not contain the mean"
    return [round(d.mean().item(), 4), round(lo, 4), round(hi, 4)]


def run(api):
    studs = {}
    for rid, r in sorted(api.runs().items()):
        if r.get("dataset") != "aircraft" or r["seed"] != 1 or r["tier"] != "final":
            continue
        name = f"{r['method']}{int(r.get('keep') or 0)}"
        if name in STUDENTS:
            studs[f"{name}_s1"] = rid
    T = api.teacher.eval()
    n = api.num_patches
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        batches.append((x, y, full))
    N = sum(b[0].shape[0] for b in batches)
    dev = batches[0][0].device
    idx = _boot_idx(N, dev)

    per = {}                                                      # (name, k) -> (kl [N], tckd [N])

    def add(key, kl, tc):
        o = per.setdefault(key, ([], []))
        o[0].append(kl)
        o[1].append(tc)

    off = 0
    for x, y, full in batches:
        B = x.shape[0]
        for k in KS:
            with torch.no_grad():
                lm = api.teacher_forward_keepmask(T, x, _random(B, n, k, dev, off)).float().log_softmax(-1)
            add(("random", k), *_stats(full, lm, y))
        off += B
    for sname, rid in studs.items():
        student = api.student(rid)
        for x, y, full in batches:
            sc = _rollouts(api, student, x)
            for k in KS:
                for m in SCORES:
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, _topk(sc[m], k)).float().log_softmax(-1)
                    add((f"{sname}:{m}", k), *_stats(full, lm, y))
    per = {key: (torch.cat(v[0]), torch.cat(v[1])) for key, v in per.items()}

    means = {f"{a}@{k}": {"kl": round(v[0].mean().item(), 4), "tckd": round(v[1].mean().item(), 4)}
             for (a, k), v in per.items()}
    api.write_json("means", means)

    out, verdict = {}, {}
    for sname in studs:
        out[sname], verdict[sname] = {}, {}
        for k in KS:
            r6kl, r6tc = per[(f"{sname}:roll6", k)]
            row = {}
            for ref in ("roll0", "roll11", "random"):
                kl, tc = per[(f"{sname}:{ref}", k)] if ref in SCORES else per[(ref, k)]
                row[ref] = {"d_tckd_ci": _ci(r6tc - tc, idx), "d_kl_ci": _ci(r6kl - kl, idx)}
            out[sname][str(k)] = row
            # preregistered reading (PROPOSALS H-A1): CI upper bound of TCKD difference < 0
            verdict[sname][str(k)] = {ref: row[ref]["d_tckd_ci"][2] < 0 for ref in ("roll0", "roll11", "random")}
    api.write_json("paired_ci", out)
    api.write_json("verdict", verdict)
    api.write_json("summary", {"students": studs, "n_val": N, "ks": list(KS), "nboot": NBOOT, "verdict": verdict})
