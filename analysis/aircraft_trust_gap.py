"""H-A26 (QUEUE 2026-11-03 16:50 step 3, TrustMask weak points): on FGVC-Aircraft at K = 29,
(a) does the student's attention come to agree with the cached teacher map as training goes on, and is there a point
    where picking by the student alone stops losing (or starts winning) against the teacher filter (trustfilt)?
(b) why do "gap" tokens (teacher-important, student not attending) hurt: does a mask with gap tokens make the masked
    teacher less faithful to the full teacher (input off the teacher's distribution), or is the masked teacher still
    faithful but further from what the student can produce (student cannot follow)?

Students: Aircraft final full@196 seeds 1, 2, 3 at epoch 25, 50 and final (25% / 50% / 100% of training; the API
exposes no epoch-0 pretrained student). Full KD students, so no token rule shaped them.
Teacher map: rollout of the whole val image (harness/teacher_cache.py rule: head mean, 0.5 A + 0.5 I, rows
renormalised, product over all blocks, CLS row). Student score: late-block rollout, blocks 6..11 (lateroll / trustfilt).
Masks, all exactly K = 29 tokens:
  S    student top-K (lateroll)
  F    student top-K inside the teacher's top ceil(1.5 K) = 44 (trustfilt, H-C14)
  T    teacher-map top-K (teacher_topk)
  G50  15 tokens = F's first 15 by student score, 14 = gap tokens by teacher map (teacher top-44 minus student
       top-44), topped up from F's order when an image has fewer than 14 gap tokens
  SW   per batch: images whose top-K overlap |S & T| / K is below the batch median use F, the others S
       (a trust switch a selector could compute: student rollout and cached teacher map only)
  R    random K (numpy default_rng on the analysis side only)
Per image: fid = KL(teacher full || teacher masked); follow = KL(teacher masked || student full) (the KD target's
distance from the student); top-1 agreement of masked with full teacher; overlap@K and Spearman(student, teacher map).
Seed mean per image over the 3 students, then image bootstrap 95% CI (2000 draws, numpy default_rng 20261103).
No position prior: every score is a per-image map.
"""
import math

import numpy
import torch

META = {
    "question": "on Aircraft K=29, does student-teacher map agreement rise with training with a switch point where "
                "student-only picking catches the teacher filter, and do gap tokens hurt through masked-teacher "
                "fidelity or through distance from the student?",
    "method": "attention rollout (student late blocks, teacher all blocks); keep-only masked teacher faithfulness "
              "(KL, top-1 agreement); KD target distance KL(masked teacher || student); image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Hooker et al. NeurIPS 2019 (keep-only); Son et al. ECCV 2024 "
             "(MaskedKD); Cho & Hariharan ICCV 2019 (capacity gap: student cannot follow a far target)",
}

K = 29
WIDE = math.ceil(1.5 * K)
NG = K // 2
SPLIT = 6
NBOOT = 2000
BATCH = 50
EPOCHS = (25, 50, "final")
CONDS = ("S", "F", "T", "G50", "SW", "R")


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


def _rank(v):
    return v.argsort(1).argsort(1).float()


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(1, keepdim=True)
    rb = rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1))


def _masks(s, t, rnd):
    B, n = s.shape
    t_rank = t.argsort(1, descending=True).argsort(1)
    allowed = t_rank < WIDE
    f_score = s.masked_fill(~allowed, -1.0)
    f_order = f_score.argsort(1, descending=True)
    F = torch.zeros_like(s, dtype=torch.bool).scatter_(1, f_order[:, :K], True)
    S = _topk(s, K)
    T = _topk(t, K)
    gap = allowed & ~_topk(s, WIDE)
    head = torch.zeros_like(F).scatter_(1, f_order[:, :K - NG], True)
    g_score = t.masked_fill(~(gap & ~head), -1.0)
    g_order = g_score.argsort(1, descending=True)[:, :NG]
    ok = torch.gather(gap & ~head, 1, g_order)
    G = head | torch.zeros_like(head).scatter_(1, g_order, ok)
    need = K - G.sum(1)                          # top up from F's order to exactly K
    fill_rank = f_score.masked_fill(G, -2.0).argsort(1, descending=True).argsort(1)
    G = G | (fill_rank < need[:, None])
    ov = (S & T).sum(1).float() / K
    low = ov < ov.median()
    SW = torch.where(low[:, None], F, S)
    return {"S": S, "F": F, "T": T, "G50": G, "SW": SW, "R": _topk(rnd, K)}, ov, gap.sum(1).float()


def _ci(v, gen):
    n = len(v)
    bs = numpy.array([v[gen.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return [round(float(v.mean()), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4)]


def _spear_ci(a, b, gen):
    def sp(i):
        ra = numpy.argsort(numpy.argsort(a[i])).astype(float)
        rb = numpy.argsort(numpy.argsort(b[i])).astype(float)
        return float(numpy.corrcoef(ra, rb)[0, 1])
    n = len(a)
    bs = numpy.array([sp(gen.integers(0, n, n)) for _ in range(NBOOT)])
    return [round(sp(numpy.arange(n)), 4), round(float(numpy.quantile(bs, 0.025)), 4),
            round(float(numpy.quantile(bs, 0.975)), 4)]


def run(api):
    runs = api.runs()
    studs = {}
    for rid, r in sorted(runs.items()):
        if r.get("dataset") == "aircraft" and r["tier"] == "final" and r["method"] == "full" and r["seed"] in (1, 2, 3):
            studs[r["seed"]] = rid
    api.write_json("students", {str(s): [rid, runs[rid]["val_acc_last"], runs[rid]["snapshots"]]
                                for s, rid in sorted(studs.items())})
    T = api.teacher.eval()
    n = api.num_patches

    batches = []
    for x, y in api.val_batches(batch=BATCH):
        tm, _ = _rollout(api, T, x, 0)
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        batches.append((x, y, tm, full))

    # per[epoch][stat][cond] -> list over seeds of per-image numpy arrays
    per = {str(e): {} for e in EPOCHS}
    for seed, rid in sorted(studs.items()):
        for ep in EPOCHS:
            S = api.student(rid, epoch=None if ep == "final" else ep)
            g = numpy.random.default_rng(20261103 + seed)
            acc = {}
            for x, y, tm, full in batches:
                B = x.shape[0]
                s, slp = _rollout(api, S, x, SPLIT)
                rnd = torch.from_numpy(g.random((B, n))).float().to(x.device)
                masks, ov, ngap = _masks(s, tm, rnd)
                rows = {"overlap": ov, "spearman": _spearman(s, tm), "n_gap": ngap,
                        "follow_full": (full.exp() * (full - slp)).sum(-1),
                        "s_acc": (slp.argmax(-1) == y).float()}
                for c, keep in masks.items():
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    rows[f"fid_{c}"] = (full.exp() * (full - lm)).sum(-1)
                    rows[f"follow_{c}"] = (lm.exp() * (lm - slp)).sum(-1)
                    rows[f"agree_{c}"] = (lm.argmax(-1) == full.argmax(-1)).float()
                    rows[f"tacc_{c}"] = (lm.argmax(-1) == y).float()
                rows["gapfrac_T"] = (masks["T"] & ~_topk(s, WIDE)).sum(1).float() / K
                for kname, v in rows.items():
                    acc.setdefault(kname, []).append(v.float().cpu())
            for kname, v in acc.items():
                per[str(ep)].setdefault(kname, []).append(torch.cat(v).numpy())

    gen = numpy.random.default_rng(20261103)
    out = {"students": {str(s): r for s, r in studs.items()}, "k": K, "wide": WIDE, "n_gap_tokens_in_G50": NG,
           "n_val": int(sum(b[1].shape[0] for b in batches))}
    for ep in EPOCHS:
        d = {kname: numpy.mean(numpy.stack(v), 0) for kname, v in per[str(ep)].items()}   # seed mean per image
        seedwise = {kname: [round(float(a.mean()), 4) for a in v] for kname, v in per[str(ep)].items()}
        e = {"mean_ci": {kname: _ci(v, gen) for kname, v in d.items()}, "seedwise_mean": seedwise}
        e["diff"] = {
            "fid_S_minus_F": _ci(d["fid_S"] - d["fid_F"], gen),
            "fid_T_minus_F": _ci(d["fid_T"] - d["fid_F"], gen),
            "fid_SW_minus_F": _ci(d["fid_SW"] - d["fid_F"], gen),
            "fid_SW_minus_S": _ci(d["fid_SW"] - d["fid_S"], gen),
            "fid_G50_minus_F": _ci(d["fid_G50"] - d["fid_F"], gen),
            "follow_G50_minus_F": _ci(d["follow_G50"] - d["follow_F"], gen),
            "follow_T_minus_F": _ci(d["follow_T"] - d["follow_F"], gen),
            "follow_S_minus_F": _ci(d["follow_S"] - d["follow_F"], gen),
            "agree_G50_minus_F": _ci(d["agree_G50"] - d["agree_F"], gen),
            "agree_S_minus_F": _ci(d["agree_S"] - d["agree_F"], gen),
        }
        e["spearman_overlap_vs_fidSminusF"] = _spear_ci(d["overlap"], d["fid_S"] - d["fid_F"], gen)
        out[str(ep)] = e
        api.write_json(f"epoch_{ep}", e)

    ov = [out[str(e)]["mean_ci"]["overlap"][0] for e in EPOCHS]
    dsf = [out[str(e)]["diff"]["fid_S_minus_F"] for e in EPOCHS]
    fin = out["final"]["diff"]
    out["verdict"] = {
        "A1_overlap_rises": bool(ov[0] < ov[1] < ov[2]),
        "A2_switch_point": bool(dsf[0][1] > 0 and dsf[2][2] < 0),
        "A3_overlap_flags_filter_images": bool(out["final"]["spearman_overlap_vs_fidSminusF"][2] < 0),
        "B1_gap_hurts_fidelity": bool(fin["fid_G50_minus_F"][1] > 0),
        "B2_gap_farther_from_student": bool(fin["follow_G50_minus_F"][1] > 0),
    }
    api.write_json("summary", out)
