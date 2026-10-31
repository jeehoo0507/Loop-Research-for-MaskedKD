"""H-C11 decomposition (spatial part, run before any promotion): do the student's high-norm tokens sit at the
image border, so that excluding them (selectors/sinkfree.py) just pushes the selection toward the centre?

Common student (CUB final full-KD s1), 600 val images, K = 12, 29, 49. For maskedkd, sinkfree, rollout, random
(numpy default_rng 20261031) and centre@K: mean normalized distance of the chosen tokens to the image centre
(0 = centre, 1 = corner), share of the choice inside centre@K (overlap), and overlap of sinkfree with maskedkd.
Also the mean distance of sink tokens vs all patch tokens. Sinks as in analysis/cub_sink_check.py.
"""
import math

import numpy
import torch

META = {
    "question": "on CUB, are the student's high-norm tokens at the border, so that sinkfree's selection is just "
                "more central than MaskedKD's?",
    "method": "spatial distribution and centre@K overlap of selected tokens; high-norm artifact detection "
              "(per-image median+3MAD of block-10 token norms)",
    "paper": "Darcet et al. ICLR 2024 (ViTs need registers); Son et al. ECCV 2024 (MaskedKD); Abnar & Zuidema ACL "
             "2020 (rollout)",
}

KS = (12, 29, 49)
SIDE = 14
MAD_K = 3.0


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def run(api):
    runs = api.runs()
    common_id = next(rid for rid, r in sorted(runs.items()) if r.get("dataset") == "cub" and r["tier"] == "final"
                     and str(r["method"]).split("@")[0] == "full" and r["seed"] == 1)
    S = api.student(common_id)
    n = api.num_patches
    depth = len(S.blocks)
    gen = numpy.random.default_rng(20261031)
    pos = torch.arange(n)
    c = (SIDE - 1) / 2
    dist = (((pos // SIDE - c) ** 2 + (pos % SIDE - c) ** 2).float().sqrt() / math.sqrt(2 * c * c))
    acc = {k: {} for k in KS}
    sink_d, all_d, sink_n = [], [], []

    def add(k, key, v):
        acc[k].setdefault(key, []).append(v)

    for x, y in api.val_batches(batch=50):
        B = x.shape[0]
        d = dist.to(x.device).expand(B, n)
        with torch.no_grad():
            out = api.student_forward(S, x, attn_layers=list(range(depth)), token_layers=[depth - 2])
            A = [a.float().mean(1) for a in out.attn]
            N = A[0].shape[-1]
            eye = torch.eye(N, device=x.device).expand(B, N, N)
            R = None
            for s in range(depth - 1, -1, -1):
                a = 0.5 * A[s] + 0.5 * eye
                a = a / a.sum(-1, keepdim=True)
                R = a if R is None else R @ a
            last, roll = A[-1][:, 0, 1:], R[:, 0, 1:]
            nrm = out.tokens[depth - 2][:, 1:].float().norm(dim=-1)
            med = nrm.median(dim=1, keepdim=True).values
            mad = (nrm - med).abs().median(dim=1, keepdim=True).values
            sink = nrm > med + MAD_K * mad
        sink_d.append(((d * sink).sum() / sink.sum().clamp_min(1)).item() * 1.0)
        sink_n.append(sink.sum().item())
        all_d.append(d.mean().item())
        rnd = torch.from_numpy(gen.random((B, n))).float().to(x.device)
        for k in KS:
            masks = {"maskedkd": _topk(last, k), "sinkfree": _topk(last.masked_fill(sink, -1.0), k),
                     "rollout": _topk(roll, k), "random": _topk(rnd, k), "center": _topk(-d - pos.to(x.device) * 1e-6, k)}
            cen = masks["center"]
            for m, mk in masks.items():
                add(k, f"dist_{m}", (d * mk).sum().item() / (B * k))
                add(k, f"in_center_{m}", (mk & cen).sum().item() / (B * k))
            add(k, "overlap_sinkfree_maskedkd", (masks["sinkfree"] & masks["maskedkd"]).sum().item() / (B * k))
            add(k, "overlap_sinkfree_rollout", (masks["sinkfree"] & masks["rollout"]).sum().item() / (B * k))
    res = {"common_student": common_id,
           "sink_mean_dist": round(float(numpy.average(sink_d, weights=sink_n)), 4),
           "all_mean_dist": round(float(numpy.mean(all_d)), 4)}
    for k in KS:
        res[str(k)] = {key: round(float(numpy.mean(v)), 4) for key, v in acc[k].items()}
    api.write_json("summary", res)
