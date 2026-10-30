"""H-W5: are the last block's CLS-attention peaks register-like high-norm tokens, and is that why MaskedKD's
masked teacher fails on the Waterbirds minority groups?

analysis_group_fidelity_1030_143842_1ea0 / _144455_654a: with the same student, block-9 CLS attention top-K
keeps the masked teacher far more correct than block-11 (MaskedKD), worst group 72.18 vs 47.37 at K=49, and
MaskedKD's top-K overlaps the token-norm top-K twice as much as random does. Darcet et al. (ICLR 2024) report
that ViTs park global information in a few high-norm, low-local-information tokens that attract attention in
the late blocks. Here, per student and per block l in 6..11:
  hn_frac_l    share of patch tokens whose output norm is a per-image outlier (z > Z) at block l
  mass_l       CLS attention mass (head mean) of block l on the block-10 outlier tokens
and per K, masked-teacher accuracy by group (as in group_fidelity) for
  L11          MaskedKD
  L11_nohn     block-11 CLS attention top-K after dropping block-10 outlier tokens (refilled in L11 order)
  L9           block-9 CLS attention top-K
  hn_in_L11    share of MaskedKD's K that are outliers; hn_in_L9 likewise
  lowinfo      share of the K whose block-10 token is closer (cosine) to the image's mean token than the median
                patch is (Darcet's "low local information" reading)
Students: screen seed-0 full KD (final) and maskedkd@49 (final).
"""
import torch

META = {
    "question": "do MaskedKD's last-block CLS-attention peaks fall on register-like high-norm tokens, and does "
                "removing them restore masked-teacher worst-group accuracy?",
    "method": "high-norm artifact token detection (token norm outliers) and per-layer CLS attention mass; "
              "group-sliced faithfulness of the masked prediction",
    "paper": "Darcet, Oquab, Mairal & Bojanowski, Vision Transformers Need Registers, ICLR 2024; "
             "Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

KS = (12, 29, 49)
Z = 2.0
LAYERS = (6, 7, 8, 9, 10, 11)


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def run(api):
    runs = api.runs()
    pick = []
    for rid, r in runs.items():
        if r["tier"] == "screen" and r["seed"] == 0 and r["method"] == "full":
            pick.append(("full_final", rid))
        if r["tier"] == "screen" and r["seed"] == 0 and r["method"] == "maskedkd" and r["keep"] == 49:
            pick.append(("mkd49_final", rid))
    api.write_json("students", dict(pick))
    groups = api.val_groups()
    T = api.teacher.eval()
    for sname, rid in pick:
        student = api.student(rid)
        depth = len(student.blocks)
        lay = {"hn_frac": {l: 0.0 for l in LAYERS}, "mass": {l: 0.0 for l in LAYERS}}
        rows = {}
        ng = [0] * 4
        tot = 0
        off = 0
        for x, y in api.val_batches(batch=50):
            B = x.shape[0]
            gb = groups[off:off + B].to(x.device)
            off += B
            tot += B
            for gi in range(4):
                ng[gi] += int((gb == gi).sum().item())
            with torch.no_grad():
                out = api.student_forward(student, x, attn_layers=list(LAYERS), token_layers=list(LAYERS))
                cls = {l: out.attn[l].float().mean(1)[:, 0, 1:] for l in LAYERS}
                norms = {l: out.tokens[l][:, 1:].float().norm(dim=-1) for l in LAYERS}
                hn = {}
                for l in LAYERS:
                    nm = norms[l]
                    z = (nm - nm.mean(1, keepdim=True)) / nm.std(1, keepdim=True).clamp(min=1e-6)
                    hn[l] = z > Z
                    lay["hn_frac"][l] += hn[l].float().mean(1).sum().item()
                h10 = hn[depth - 2]
                for l in LAYERS:
                    lay["mass"][l] += (cls[l] * h10.float()).sum(1).sum().item()
                t10 = out.tokens[depth - 2][:, 1:].float()
                cos = torch.nn.functional.cosine_similarity(t10, t10.mean(1, keepdim=True), dim=-1)
                lowinfo = cos > cos.median(1, keepdim=True).values
                for k in KS:
                    l11 = cls[depth - 1]
                    masks = {"L11": _topk(l11, k), "L9": _topk(cls[9], k),
                             "L11_nohn": _topk(l11.masked_fill(h10, -1.0), k)}
                    for name, keep in masks.items():
                        t_m = api.teacher_forward_keepmask(T, x, keep).float()
                        ok = (t_m.argmax(-1) == y).float()
                        row = rows.setdefault(f"{name}@{k}", {"t_acc": [0.0] * 4, "hn_in": 0.0, "lowinfo_in": 0.0})
                        for gi in range(4):
                            row["t_acc"][gi] += ok[gb == gi].sum().item()
                        row["hn_in"] += ((keep & h10).sum(1).float() / k).sum().item()
                        row["lowinfo_in"] += ((keep & lowinfo).sum(1).float() / k).sum().item()
        out = {"per_layer": {f"L{l}": {"hn_frac": round(lay["hn_frac"][l] / tot, 4),
                                       "cls_mass_on_L10_hn": round(lay["mass"][l] / tot, 4)} for l in LAYERS}}
        for key, row in rows.items():
            acc = [round(100 * a / max(b, 1), 2) for a, b in zip(row["t_acc"], ng)]
            out[key] = {"t_acc": acc, "t_acc_mean": round(100 * sum(row["t_acc"]) / tot, 2),
                        "t_wga": min(acc), "hn_in": round(row["hn_in"] / tot, 4),
                        "lowinfo_in": round(row["lowinfo_in"] / tot, 4)}
        api.write_json(f"reg_{sname}", out)
        del student
        torch.cuda.empty_cache()
