"""Waterbirds: does the masked teacher keep its worst-group behaviour, by signal and K?

The archived signal_fidelity analysis ranked student-side signals by how close the masked teacher stays to
the full teacher on average. On Waterbirds the average hides the shortcut: a teacher fed background tokens
can stay "faithful" on majority groups and be wrong on the minority groups (waterbird on land, landbird on
water). Here every metric is split by val group (2*y + place, analysis only):
  t_acc_g   masked-teacher accuracy in group g        (KD target quality where the shortcut fails)
  agree_g   top-1 agreement masked vs full teacher in group g
  fid_kl_g  KL(teacher_full || teacher_masked) in group g
  overlap   |S ∩ MaskedKD| / K,  ov_oracle  |S ∩ teacher-attention top-K| / K,  cdist (patch units)
Signals: the builtins' rules (maskedkd = last-block CLS attention, rollout, grid, center, random), the
teacher's own attention (oracle, analysis only), other per-token student signals (layer CLS attention,
token norm, gradient-weighted attention) and three diagnostic mixes:
  anti      the K lowest last-block CLS attention tokens (mostly background): what does background alone teach?
  half_grid top K/2 by CLS attention, the rest filled in raster-grid order (does spread add to focus?)
  nms       CLS attention top-K with greedy 3x3 suppression of neighbours (focus without clumping)
Students: screen seed-0 maskedkd@49 (final), maskedkd@98 (final and 25%/50% snapshots). Full KD (final) is in
analysis_group_fidelity_1030_143842_1ea0.
"""
import torch

META = {
    "question": "which student-side signal keeps the masked teacher correct on the Waterbirds minority groups "
                "(where the background shortcut fails), and does background-only input carry the shortcut?",
    "method": "per-layer CLS attention, Attention Rollout, Grad-weighted attention, token norm; group-sliced "
              "faithfulness of the masked prediction (deletion/insertion-style)",
    "paper": "Abnar & Zuidema ACL 2020; Chefer, Gur & Wolf ICCV 2021; Darcet et al. ICLR 2024; "
             "Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

KS = (12, 29, 49, 98)
SIDE = 14


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def _fill(first, order, k):
    """first: bool [B,N] already chosen; add the highest-`order` remaining tokens until k are kept."""
    rest = k - first.sum(1, keepdim=True)
    sc = order.masked_fill(first, float("-inf"))
    ranks = sc.argsort(1, descending=True).argsort(1)
    return first | (ranks < rest)


def _nms(scores, k):
    B, n = scores.shape
    sc = scores.clone()
    keep = torch.zeros_like(scores, dtype=torch.bool)
    blocked = torch.zeros_like(keep)
    rr = torch.arange(n, device=scores.device)
    r, c = rr // SIDE, rr % SIDE
    neigh = ((r[:, None] - r[None]).abs() <= 1) & ((c[:, None] - c[None]).abs() <= 1)   # [N,N]
    for _ in range(k):
        s = sc.masked_fill(blocked | keep, float("-inf"))
        j = s.argmax(1)
        ok = torch.isfinite(s.gather(1, j[:, None])).squeeze(1)
        if not ok.any():
            break
        keep[ok, j[ok]] = True
        blocked[ok] |= neigh[j[ok]]
    # rows that ran out of unblocked tokens: fill by score
    return _fill(keep, scores, k)


def _signals(api, student, x):
    depth = len(student.blocks)
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        out = api.student_forward(student, x, attn_layers=list(range(depth)), token_layers=[depth - 2])
        logits = out.logits.float()
        (g,) = torch.autograd.grad(logits.max(1).values.sum(), out.attn[-1])
    A = [a.detach().float() for a in out.attn]
    s = {}
    for l in (3, 6, 9, 10, 11):
        s[f"cls_attn_L{l}"] = A[l].mean(1)[:, 0, 1:]
    B, _, N, _ = A[0].shape
    eye = torch.eye(N, device=x.device).expand(B, N, N)
    R = eye.clone()
    for l in range(depth):
        a = 0.5 * A[l].mean(1) + 0.5 * eye
        R = (a / a.sum(-1, keepdim=True)) @ R
    s["rollout"] = R[:, 0, 1:]
    s["gradattn_last"] = (g.detach().float() * A[-1]).clamp(min=0).mean(1)[:, 0, 1:]
    s["toknorm_L-2"] = out.tokens[depth - 2].detach().float().norm(dim=-1)[:, 1:]
    s["anti"] = -s["cls_attn_L11"]
    return s, logits.detach()


def run(api):
    runs = api.runs()
    pick = []
    for rid, r in runs.items():
        if r["tier"] == "screen" and r["seed"] == 0 and r["method"] == "maskedkd" and r["keep"] == 98:
            pick += [("mkd98_final", rid, None)] + [(f"mkd98_e{e}", rid, e) for e in r["snapshots"]]
        if r["tier"] == "screen" and r["seed"] == 0 and r["method"] == "maskedkd" and r["keep"] == 49:
            pick.insert(0, ("mkd49_final", rid, None))   # K=49: student WGA fell to 51.88 (group 2)
    # full_final was finished in analysis_group_fidelity_1030_143842_1ea0 before that job hit a CUDA OOM
    api.write_json("students", {n: [rid, e] for n, rid, e in pick})
    groups = api.val_groups()
    T = api.teacher.eval()
    n = api.num_patches
    rr = torch.arange(n)
    c = (SIDE - 1) / 2
    cdist_all = ((rr // SIDE - c) ** 2 + (rr % SIDE - c) ** 2).float().sqrt()

    # reference: full teacher per group, and group sizes
    ref = {"t_full_acc": [0.0] * 4, "n": [0] * 4}
    results = {}
    for si, (sname, rid, ep) in enumerate(pick):
        student = api.student(rid, ep)
        for p in student.parameters():
            p.requires_grad_(False)
        acc = {}
        s_acc = [0.0] * 4
        off = 0
        for x, y in api.val_batches(batch=50):   # 50 (not 100): shares the GPU with training jobs
            B = x.shape[0]
            dev = x.device
            gb = groups[off:off + B].to(dev)
            off += B
            cd = cdist_all.to(dev)
            with torch.no_grad():
                t_full = T(x).float().log_softmax(-1)
                t_attn = api.student_forward(T, x, attn_layers=[-1]).attn[-1].float().mean(1)[:, 0, 1:]
            sig, s_logits = _signals(api, student, x)
            for gi in range(4):
                m = gb == gi
                s_acc[gi] += (s_logits.argmax(-1)[m] == y[m]).float().sum().item()
                if si == 0:
                    ref["t_full_acc"][gi] += (t_full.argmax(-1)[m] == y[m]).float().sum().item()
                    ref["n"][gi] += int(m.sum().item())
            sig["oracle"] = t_attn
            sig["random"] = torch.rand(B, n, device=dev)
            for k in KS:
                local = dict(sig)
                gidx = torch.linspace(0, n - 1, k).round().long()
                gs = torch.zeros(n)
                gs[gidx] = 1.0
                local["grid"] = gs.to(dev).expand(B, n)
                local["center"] = (-cd - rr.to(dev).float() * 1e-6).expand(B, n)
                mk = _topk(sig["cls_attn_L11"], k)
                orc = _topk(t_attn, k)
                masks = {name: _topk(sc.float(), k) for name, sc in local.items()}
                # raster-grid order: the K evenly spaced positions first, then the rest at random
                spread = torch.zeros(n, device=dev)
                for j, pos in enumerate(torch.linspace(0, n - 1, k).round().long().tolist()):
                    spread[pos] = 2.0 * n - j
                spread = spread + torch.rand(n, device=dev)
                masks["half_grid"] = _fill(_topk(sig["cls_attn_L11"], k // 2), spread.expand(B, n), k)
                masks["nms"] = _nms(sig["cls_attn_L11"], k)
                for name, keep in masks.items():
                    with torch.no_grad():
                        t_m = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    row = acc.setdefault(f"{name}@{k}", {"t_acc": [0.0] * 4, "agree": [0.0] * 4,
                                                         "fid_kl": [0.0] * 4, "overlap": 0.0, "ov_oracle": 0.0,
                                                         "cdist": 0.0, "n": 0})
                    ok = (t_m.argmax(-1) == y).float()
                    ag = (t_m.argmax(-1) == t_full.argmax(-1)).float()
                    kl = _kl(t_full, t_m)
                    for gi in range(4):
                        m = gb == gi
                        row["t_acc"][gi] += ok[m].sum().item()
                        row["agree"][gi] += ag[m].sum().item()
                        row["fid_kl"][gi] += kl[m].sum().item()
                    kk = keep.sum(1).float()
                    row["overlap"] += ((keep & mk).sum(1).float() / kk).sum().item()
                    row["ov_oracle"] += ((keep & orc).sum(1).float() / kk).sum().item()
                    row["cdist"] += ((keep.float() * cd).sum(1) / kk).sum().item()
                    row["n"] += B
        ng = ref["n"]
        out = {"student_group_acc": [round(100 * a / max(b, 1), 2) for a, b in zip(s_acc, ng)]}
        for key, row in acc.items():
            m = row.pop("n")
            o = {}
            for a in ("t_acc", "agree", "fid_kl"):
                o[a] = [round((100 if a != "fid_kl" else 1) * v / max(b, 1), 3) for v, b in zip(row[a], ng)]
            o["t_acc_mean"] = round(100 * sum(row["t_acc"]) / m, 2)
            o["t_wga"] = round(min(o["t_acc"]), 2)
            o["fid_kl_mean"] = round(sum(row["fid_kl"]) / m, 4)
            for a in ("overlap", "ov_oracle", "cdist"):
                o[a] = round(row[a] / m, 4)
            out[key] = o
        results[sname] = out
        api.write_json(f"grp_{sname}", out)
    api.write_json("reference", {"group_n": ref["n"],
                                 "t_full_acc": [round(100 * a / max(b, 1), 2) for a, b in zip(ref["t_full_acc"],
                                                                                              ref["n"])]})
    rank = {}
    for sname, out in results.items():
        for k in KS:
            rows = sorted(((key.split("@")[0], v["t_wga"], v["t_acc_mean"], v["fid_kl_mean"])
                           for key, v in out.items() if key.endswith(f"@{k}")), key=lambda t: -t[1])
            rank[f"{sname}@{k}"] = [list(r) for r in rows]
    api.write_json("ranking_by_teacher_wga", rank)
