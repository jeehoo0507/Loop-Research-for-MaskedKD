"""H-A6: on FGVC-Aircraft, how much of the full teacher's soft label does MaskedKD's masked teacher keep at
K = 98, 49, 29, 25, 12, and is it less than on CUB at the same K (CUB H-C1, analysis_cub_softlabel_fidelity)?

Why: on the Aircraft gate MaskedKD@98 sits 0.62 below Full KD (seeds 1-3, uncertain), while on CUB MaskedKD@98
is about Full. If the masked teacher keeps less of the soft label on Aircraft at the same K, the token budget that
is "enough" depends on the dataset. Descriptive only: nothing is selected or tuned, no selector is changed
(QUEUE 2026-11-02 08:50). Same statistics and code path as analysis/cub_softlabel_fidelity.py (H-C1), restricted
to the maskedkd ranking, harness rollout and the hash random reference, for the two Aircraft final seed-1 gate
students (full@196, maskedkd@98), so the numbers line up with the CUB maskedkd98_s1 table.
"""
import torch

META = {
    "question": "on FGVC-Aircraft, how faithful (KL to full teacher, DKD TCKD/NCKD, top-1 agreement) is the "
                "MaskedKD masked teacher at K = 98/49/29/25/12, compared with CUB at the same K?",
    "method": "keep-only masked-teacher faithfulness (KL to full teacher, DKD decomposition, top-1/top-5 "
              "agreement) under last-block CLS attention, attention rollout and random token keeps",
    "paper": "Hinton et al. 2015 (KD); Zhao et al. CVPR 2022 (DKD); Abnar & Zuidema ACL 2020 (rollout); "
             "Hooker et al. NeurIPS 2019 (keep-only evaluation); Son et al. ECCV 2024 (MaskedKD)",
}

KS = (98, 49, 29, 25, 12)
SCORES = ("maskedkd", "rollout", "random")
STUDENTS = ("full196", "maskedkd98")
EPS = 1e-8


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _random(B, n, k, dev, off):
    # same hash noise as cub_softlabel_fidelity.py, so both students see the same draw
    i = torch.arange(off, off + B, device=dev, dtype=torch.float64)[:, None]
    j = torch.arange(n, device=dev, dtype=torch.float64)[None, :]
    u = torch.frac(torch.sin(i * 12.9898 + j * 78.233) * 43758.5453)
    return _topk(u.float(), k)


def _attn_scores(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad():
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
    A = [a.float().mean(1) for a in out.attn]
    B, N, _ = A[0].shape
    eye = torch.eye(N, device=x.device).expand(B, N, N)
    R = None
    for a in A:
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        R = a if R is None else a @ R
    return {"maskedkd": A[-1][:, 0, 1:], "rollout": R[:, 0, 1:]}


def _stats(full_lp, mask_lp, y):
    pf, pm = full_lp.exp(), mask_lp.exp()
    kl = (pf * (full_lp - mask_lp)).sum(-1)
    pt_f = pf.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    pt_m = pm.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()
    mask_y = torch.ones_like(pf, dtype=torch.bool).scatter_(1, y[:, None], False)
    qf = (pf * mask_y) / (1 - pt_f)[:, None]
    qm = (pm * mask_y) / (1 - pt_m)[:, None]
    nkl = torch.where(mask_y, qf * ((qf + EPS).log() - (qm + EPS).log()), torch.zeros_like(qf)).sum(-1)
    nckd = (1 - pt_f) * nkl
    t5f = pf.topk(5, -1).indices
    t5m = pm.topk(5, -1).indices
    top5 = (t5f[:, :, None] == t5m[:, None, :]).any(-1).float().mean(-1)
    return {"kl": kl, "tckd": tckd, "nckd": nckd, "nontarget_kl": nkl,
            "top1_agree": (pf.argmax(-1) == pm.argmax(-1)).float(), "top5_overlap": top5,
            "t_acc": (pm.argmax(-1) == y).float(), "p_true": pt_m}


def run(api):
    runs = api.runs()
    studs = {}
    for rid, r in sorted(runs.items()):
        if r.get("dataset") != "aircraft" or r["seed"] != 1 or r["tier"] != "final":
            continue
        name = f"{r['method']}{int(r.get('keep') or 0)}"
        if name in STUDENTS:
            studs[f"{name}_s1"] = rid
    api.write_json("students", {k: [v, runs[v]["val_acc_last"]] for k, v in studs.items()})

    T = api.teacher.eval()
    n = api.num_patches
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        batches.append((x, y, full))
    N = sum(b[0].shape[0] for b in batches)
    ref = {}
    ref["teacher_full_acc"] = round(100 * sum((b[2].argmax(-1) == b[1]).float().sum().item() for b in batches) / N, 2)
    ref["teacher_full_p_true"] = round(sum(b[2].exp().gather(1, b[1][:, None]).sum().item() for b in batches) / N, 4)
    ref["teacher_full_nontarget_mass"] = round(1 - ref["teacher_full_p_true"], 4)
    api.write_json("reference", ref)

    results = {}
    for sname, rid in studs.items():
        student = api.student(rid)
        acc = {(m, k): {} for m in SCORES for k in KS}
        off = 0
        for x, y, full in batches:
            B = x.shape[0]
            sc = _attn_scores(api, student, x)
            for k in KS:
                for m in SCORES:
                    keep = _random(B, n, k, x.device, off) if m == "random" else _topk(sc[m], k)
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    for s, v in _stats(full, lm, y).items():
                        acc[(m, k)][s] = acc[(m, k)].get(s, 0.0) + v.sum().item()
            off += B
        res = {}
        for (m, k), d in acc.items():
            o = {s: round(v / N, 4) for s, v in d.items()}
            for s in ("top1_agree", "top5_overlap", "t_acc"):
                o[s] = round(100 * o[s], 2)
            res[f"{m}@{k}"] = o
        results[sname] = res
        api.write_json(f"fidelity_{sname}", res)

    # preregistered reading (PROPOSALS H-A6), against CUB H-C1 maskedkd98_s1 maskedkd@98: KL 0.0706, top-1 94.83
    m98 = results.get("maskedkd98_s1", {}).get("maskedkd@98", {})
    verdict = {"F1_kl98_above_cub": m98.get("kl", 0) > 0.0706,
               "F2_top1_98_below_cub": m98.get("top1_agree", 100) < 94.83}
    api.write_json("summary", {"students": studs, "n_val": N, "ks": list(KS), "reference": ref,
                               "maskedkd98_s1": {f"maskedkd@{k}": results.get("maskedkd98_s1", {}).get(f"maskedkd@{k}")
                                                 for k in KS},
                               "verdict": verdict})
