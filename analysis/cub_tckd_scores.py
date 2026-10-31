"""H-C2: on CUB-200, is there a position-free per-token student score that keeps the masked teacher's target
probability (TCKD part of KL) better than attention rollout at the same K, at almost no extra cost?

Basis: analysis_cub_softlabel_fidelity_1031_082711_df82 (H-C1a). On CUB most of KL(full || masked teacher) is TCKD
(the masked teacher loses p_true; NCKD is only 19-43% of KL under MaskedKD), and rollout is the strongest score
without a position prior and without a backward pass. So a new score has to beat rollout on p_true / TCKD.

Candidate: class-specific patch evidence. The student's last-block patch tokens are put through the student's own
final norm and classifier head (logit lens on patch tokens; for a ViT whose head reads the CLS token this is the
CAM analogue, Zhou et al. CVPR 2016 / Chefer et al. CVPR 2021 class-specific relevance), and the logit of the
student's own top-1 class is the score. No labels, no teacher, no position: every input is the student side of
the training step (ctx.out.tokens and ctx.student). Extra cost is one [196 x 192] x [192 x 200] product per image
(about 0.015 GFLOPs), small next to the 7.64 GFLOPs student step.
  patchcam   cam_j = head(norm(h_j))[c_hat],  c_hat = student argmax
  rollcam    log rollout_j + log softmax_j(cam)   (product of the two distributions over patches)
  roll6      rollout over blocks 6..11 only (is the early-block part of rollout needed?)
References: maskedkd (last-block CLS attention), rollout (all blocks), random (no signal). Ks: 98, 49, 25, 12.
"""
import torch

META = {
    "question": "on CUB, does class-specific patch evidence from the student's own head (patch-token logit lens / "
                "CAM analogue), alone or multiplied with attention rollout, keep the masked teacher's target "
                "probability (TCKD) better than rollout at the same K?",
    "method": "keep-only masked-teacher faithfulness (KL to full teacher with DKD TCKD/NCKD split, p_true, top-1 "
              "agreement) under patch-token logit-lens, rollout x logit-lens, partial rollout and attention scores",
    "paper": "Zhou et al. CVPR 2016 (CAM); Chefer et al. CVPR 2021 (class-specific transformer relevance); "
             "nostalgebraist 2020 / Belrose et al. 2023 (logit lens); Abnar & Zuidema ACL 2020 (rollout); "
             "Zhao et al. CVPR 2022 (DKD); Son et al. ECCV 2024 (MaskedKD)",
}

KS = (98, 49, 25, 12)
SCORES = ("maskedkd", "rollout", "roll6", "patchcam", "rollcam", "random")
EPS = 1e-8


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _random(B, n, k, dev, off):
    i = torch.arange(off, off + B, device=dev, dtype=torch.float64)[:, None]
    j = torch.arange(n, device=dev, dtype=torch.float64)[None, :]
    u = torch.frac(torch.sin(i * 12.9898 + j * 78.233) * 43758.5453)
    return _topk(u.float(), k)


def _rollout(A, start):
    B, N, _ = A[0].shape
    eye = torch.eye(N, device=A[0].device).expand(B, N, N)
    R = None
    for a in A[start:]:
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        R = a if R is None else a @ R
    return R[:, 0, 1:]


def _scores(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad():
        out = api.student_forward(student, x, attn_layers=list(range(depth)), token_layers=[depth - 1])
        A = [a.float().mean(1) for a in out.attn]
        c_hat = out.logits.float().argmax(-1)
        hp = student.norm(out.tokens[depth - 1])[:, 1:]                       # [B, 196, D], same norm as CLS
        cam = student.head(hp).float()                                         # [B, 196, C]
        cam = cam.gather(2, c_hat[:, None, None].expand(-1, cam.shape[1], 1))[..., 0]
        roll = _rollout(A, 0)
        rollcam = (roll + EPS).log() + cam.log_softmax(-1)
    return {"maskedkd": A[-1][:, 0, 1:], "rollout": roll, "roll6": _rollout(A, depth // 2),
            "patchcam": cam, "rollcam": rollcam}


def _stats(full_lp, mask_lp, y):
    pf, pm = full_lp.exp(), mask_lp.exp()
    kl = (pf * (full_lp - mask_lp)).sum(-1)
    pt_f = pf.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    pt_m = pm.gather(1, y[:, None])[:, 0].clamp(EPS, 1 - EPS)
    tckd = pt_f * (pt_f / pt_m).log() + (1 - pt_f) * ((1 - pt_f) / (1 - pt_m)).log()
    return {"kl": kl, "tckd": tckd, "nckd": kl - tckd, "p_true": pt_m,
            "top1_agree": (pf.argmax(-1) == pm.argmax(-1)).float(), "t_acc": (pm.argmax(-1) == y).float()}


def run(api):
    runs = api.runs()
    studs = {}
    for rid, r in sorted(runs.items()):
        if r.get("dataset") != "cub" or r["seed"] != 1 or r["tier"] != "final":
            continue
        studs[f"{r['method']}{int(r['keep'])}_s1"] = rid
    api.write_json("students", {k: [v, runs[v]["val_acc_last"]] for k, v in studs.items()})

    T = api.teacher.eval()
    n = api.num_patches
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        batches.append((x, y, full))
    N = sum(b[0].shape[0] for b in batches)

    results, verdict = {}, {}
    for sname, rid in studs.items():
        student = api.student(rid)
        acc = {(m, k): {} for m in SCORES for k in KS}
        ov = {(m, k): 0.0 for m in SCORES for k in KS}
        off = 0
        for x, y, full in batches:
            B = x.shape[0]
            sc = _scores(api, student, x)
            for k in KS:
                keeps = {m: (_random(B, n, k, x.device, off) if m == "random" else _topk(sc[m], k)) for m in SCORES}
                for m, keep in keeps.items():
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    for s, v in _stats(full, lm, y).items():
                        acc[(m, k)][s] = acc[(m, k)].get(s, 0.0) + v.sum().item()
                    ov[(m, k)] += (keep & keeps["rollout"]).sum().item() / k
            off += B
        res = {}
        for (m, k), d in acc.items():
            o = {s: round(v / N, 4) for s, v in d.items()}
            for s in ("top1_agree", "t_acc"):
                o[s] = round(100 * o[s], 2)
            o["overlap_with_rollout"] = round(ov[(m, k)] / N, 4)
            res[f"{m}@{k}"] = o
        results[sname] = res
        api.write_json(f"fidelity_{sname}", res)
        # preregistered reading (PROPOSALS H-C2): rollcam vs rollout at K=25 and 12
        verdict[sname] = {str(k): {"tckd_rollcam_lt_rollout": res[f"rollcam@{k}"]["tckd"] < res[f"rollout@{k}"]["tckd"],
                                   "kl_rollcam_le_rollout": res[f"rollcam@{k}"]["kl"] <= res[f"rollout@{k}"]["kl"],
                                   "tckd": [res[f"rollcam@{k}"]["tckd"], res[f"rollout@{k}"]["tckd"]],
                                   "kl": [res[f"rollcam@{k}"]["kl"], res[f"rollout@{k}"]["kl"]]}
                          for k in (25, 12)}
    api.write_json("verdict", verdict)
    api.write_json("summary", {"students": studs, "n_val": N, "ks": list(KS), "verdict": verdict})
