"""Is `center` good because the teacher needs a *compact contiguous* region, and can the student's
attention map *place* that region better than the fixed image centre?

Basis: analysis_signal_fidelity_1030_014040_eb3d (center >> every scattered top-K signal on
KL(T_full || T_masked)) and the epoch-15 val of the K=49 screens (center 19.52 > random 19.20 > grid 18.36
> maskedkd 16.96, interim). Hypothesis H6: an interpretability map used for *localisation*
(weakly-supervised object localisation from attention, e.g. rollout / CLS attention centroids) rather
than for *ranking* tokens.

For K in {25, 49} and each student, selections compared:
  center              K patches nearest the image centre (builtin rule)
  win_<sig>_cent      K patches nearest the attention-weighted centroid of map <sig>
  win_<sig>_peak      K patches nearest the argmax of the Gaussian-blurred map <sig>
  blur_<sig>          top-K of the Gaussian-blurred map (blob-shaped, not disc-shaped)
  blurc_<sig>         top-K of blurred map x centre prior
  shift_rand          centre disc shifted by a random offset in {-2..2}^2 (control: does moving help at all?)
  shift_best          analysis-only upper bound: best of 25 shifted discs per image by fid_kl (uses the teacher)
Metrics as in signal_fidelity: fid_kl, agree, t_acc, gap_kl, overlap with MaskedKD, cdist, plus
shift = mean distance of the window centre from the image centre.
"""
import math

import torch
import torch.nn.functional as F

META = {
    "question": "does a compact window placed at the student's attention centroid preserve the teacher better "
                "than the fixed centre window, and how much headroom does window placement have?",
    "method": "Attention Rollout and CLS-attention maps used for weakly-supervised object localisation "
              "(centroid / blurred-peak), compared with fixed centre crop and best-shift upper bound",
    "paper": "Abnar & Zuidema ACL 2020; Chefer, Gur & Wolf CVPR 2021; Gao et al. TS-CAM ICCV 2021",
}

KS = (25, 49)
N_IMG = 1000
SIDE = 14
SHIFTS = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def _blur(m, sigma=1.5):
    B = m.shape[0]
    r = 3
    g = torch.arange(-r, r + 1, device=m.device).float()
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    img = m.reshape(B, 1, SIDE, SIDE)
    img = F.conv2d(F.pad(img, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
    img = F.conv2d(F.pad(img, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))
    return img.reshape(B, -1)


def _disc(cy, cx, k, rows, cols):
    """K patches nearest (cy, cx) per image; cy, cx: [B] float."""
    d = (rows[None] - cy[:, None]) ** 2 + (cols[None] - cx[:, None]) ** 2
    d = d + torch.arange(d.shape[1], device=d.device).float()[None] * 1e-6
    return _topk(-d, k)


def _maps(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad():
        out = api.student_forward(student, x, attn_layers=list(range(depth)), token_layers=[depth - 2])
    A = [a.float() for a in out.attn]
    B, _, N, _ = A[0].shape
    eye = torch.eye(N, device=x.device).expand(B, N, N)
    R = eye.clone()
    for l in range(depth):
        a = 0.5 * A[l].mean(1) + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        R = a @ R
    m = {"rollout": R[:, 0, 1:], "cls11": A[-1].mean(1)[:, 0, 1:], "cls5": A[5].mean(1)[:, 0, 1:],
         "toknorm": out.tokens[depth - 2].float().norm(dim=-1)[:, 1:]}
    return m, out.logits.float().log_softmax(-1)


def run(api):
    runs = api.runs()
    pick = []
    for rid, r in runs.items():
        if r["tier"] != "screen" or r["seed"] != 0:
            continue
        if r["method"] == "maskedkd" and r["keep"] == 98:
            pick.append(("mkd98_final", rid, None))
        if r["method"] == "maskedkd" and r["keep"] == 49:
            pick += [(f"mkd49_e{e}", rid, e) for e in r["snapshots"][:1]] + [("mkd49_final", rid, None)]
    api.write_json("students", {n: [rid, e] for n, rid, e in pick})
    T = api.teacher.eval()
    n = api.num_patches
    dev = api.device
    rr = torch.arange(n, device=dev)
    rows, cols = (rr // SIDE).float(), (rr % SIDE).float()
    c = (SIDE - 1) / 2
    cd = ((rows - c) ** 2 + (cols - c) ** 2).sqrt()
    prior = torch.exp(-cd ** 2 / (2 * 4.0 ** 2))
    results = {}
    for sname, rid, ep in pick:
        student = api.student(rid, ep)
        acc = {}
        for x, y in api.val_batches(batch=100, limit=N_IMG):
            B = x.shape[0]
            with torch.no_grad():
                t_full = T(x).float().log_softmax(-1)
            maps, s_log = _maps(api, student, x)
            mk_scores = maps["cls11"]
            cc = torch.full((B,), c, device=dev)
            for k in KS:
                sel = {"center": (_disc(cc, cc, k, rows, cols), cc, cc)}
                for nm, mp in maps.items():
                    w = mp / mp.sum(1, keepdim=True)
                    cy, cx = (w * rows).sum(1), (w * cols).sum(1)
                    sel[f"win_{nm}_cent"] = (_disc(cy, cx, k, rows, cols), cy, cx)
                    bl = _blur(mp)
                    am = bl.argmax(1)
                    py, px = rows[am], cols[am]
                    sel[f"win_{nm}_peak"] = (_disc(py, px, k, rows, cols), py, px)
                    sel[f"blur_{nm}"] = (_topk(bl, k), None, None)
                    sel[f"blurc_{nm}"] = (_topk(bl / bl.max(1, keepdim=True).values * prior, k), None, None)
                off = torch.randint(-2, 3, (2, B), device=dev).float()
                sel["shift_rand"] = (_disc(cc + off[0], cc + off[1], k, rows, cols), cc + off[0], cc + off[1])
                # best shift: analysis-only upper bound, picked with the teacher itself
                best_kl = torch.full((B,), float("inf"), device=dev)
                best_keep = None
                best_c = torch.zeros(2, B, device=dev)
                for dy, dx in SHIFTS:
                    keep = _disc(cc + dy, cc + dx, k, rows, cols)
                    with torch.no_grad():
                        t_m = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    kl = _kl(t_full, t_m)
                    better = kl < best_kl
                    best_kl = torch.where(better, kl, best_kl)
                    best_keep = keep if best_keep is None else torch.where(better[:, None], keep, best_keep)
                    best_c[0] = torch.where(better, cc + dy, best_c[0])
                    best_c[1] = torch.where(better, cc + dx, best_c[1])
                sel["shift_best"] = (best_keep, best_c[0], best_c[1])
                mk = _topk(mk_scores, k)
                for name, (keep, cy, cx) in sel.items():
                    with torch.no_grad():
                        t_m = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    row = acc.setdefault(f"{name}@{k}", {"fid_kl": 0.0, "agree": 0.0, "t_acc": 0.0, "gap_kl": 0.0,
                                                         "overlap": 0.0, "cdist": 0.0, "shift": 0.0, "n": 0})
                    row["fid_kl"] += _kl(t_full, t_m).sum().item()
                    row["agree"] += (t_m.argmax(-1) == t_full.argmax(-1)).float().sum().item()
                    row["t_acc"] += (t_m.argmax(-1) == y).float().sum().item()
                    row["gap_kl"] += _kl(t_m, s_log).sum().item()
                    row["overlap"] += ((keep & mk).sum(1).float() / k).sum().item()
                    row["cdist"] += ((keep.float() * cd).sum(1) / k).sum().item()
                    if cy is not None:
                        row["shift"] += ((cy - c) ** 2 + (cx - c) ** 2).sqrt().sum().item()
                    row["n"] += B
        out = {}
        for key, row in acc.items():
            m = row.pop("n")
            out[key] = {a: round(b / m, 4) for a, b in row.items()}
        results[sname] = out
        api.write_json(f"win_{sname}", out)
    rank = {}
    for sname, out in results.items():
        for k in KS:
            rws = sorted(((key.split("@")[0], v["fid_kl"], v["t_acc"], v["shift"]) for key, v in out.items()
                          if key.endswith(f"@{k}")), key=lambda t: t[1])
            rank[f"{sname}@{k}"] = [list(t) for t in rws]
    api.write_json("ranking_by_fid_kl", rank)
