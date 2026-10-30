"""Which student-side interpretability signal picks the tokens that keep the masked teacher closest to
the full teacher (and to the truth)?

For every signal and K in {12, 25, 49} we feed the teacher only the chosen tokens and measure
  fid_kl   KL(teacher_full || teacher_masked)          (how much of the target survives masking)
  agree    top-1 agreement teacher_masked vs teacher_full
  t_acc    teacher_masked accuracy on val labels         (analysis only)
  gap_kl   KL(teacher_masked || student)                 (size of the KD signal the student would get)
  overlap  |S ∩ MaskedKD| / K
  cdist    mean distance of chosen patches to the image centre (patch units)
Students: final and 25%/50% snapshots of the screen seed-0 maskedkd@98 run, and the final full-KD student.
The builtins (maskedkd/random/grid/center) are included so this proxy can later be checked against their
screen val results at the same K.
"""
import math

import torch

META = {
    "question": "which student-side interpretability signal (layer CLS attention, rollout, value-norm attention, "
                "gradient-weighted attention, GAE) selects tokens that best preserve the teacher's prediction?",
    "method": "Attention Rollout; attention-norm (||alpha f(x)||); Grad-weighted attention / Generic Attention "
              "Explainability; per-layer CLS attention",
    "paper": "Abnar & Zuidema ACL 2020; Kobayashi et al. EMNLP 2020; Chefer, Gur & Wolf ICCV 2021; "
             "Hao et al. AAAI 2021",
}

KS = (12, 25, 49)
N_IMG = 1000
SIDE = 14


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def _signals(api, student, x):
    """All scores [B,196] from one student pass (with grad for the gradient-based ones)."""
    depth = len(student.blocks)
    with torch.enable_grad():
        out = api.student_forward(student, x, attn_layers=list(range(depth)), token_layers=[depth - 2])
        logits = out.logits.float()
        target = logits.max(1).values.sum()      # student's own predicted class, no labels
        grads = torch.autograd.grad(target, out.attn)
    A = [a.detach().float() for a in out.attn]
    G = [g.detach().float() for g in grads]
    s = {}
    for l in range(depth):
        s[f"cls_attn_L{l}"] = A[l].mean(1)[:, 0, 1:]
    for h in range(A[-1].shape[1]):
        s[f"cls_attn_last_h{h}"] = A[-1][:, h, 0, 1:]
    # Attention rollout (Abnar & Zuidema 2020)
    B, _, N, _ = A[0].shape
    eye = torch.eye(N, device=x.device).expand(B, N, N)
    R = eye.clone()
    for l in range(depth):
        a = 0.5 * A[l].mean(1) + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        R = a @ R
    s["rollout"] = R[:, 0, 1:]
    # Generic attention explainability (Chefer et al. 2021): R += mean_h relu(grad*A) @ R
    R = eye.clone()
    for l in range(depth):
        R = R + (G[l] * A[l]).clamp(min=0).mean(1) @ R
    s["gae"] = R[:, 0, 1:]
    s["gradattn_last"] = (G[-1] * A[-1]).clamp(min=0).mean(1)[:, 0, 1:]
    s["absgrad_last"] = G[-1].abs().mean(1)[:, 0, 1:]
    # Attention-norm (Kobayashi et al. 2020) at the last block: || sum_h alpha_h[cls,j] * v_hj W_O^h ||
    blk = student.blocks[-1]
    with torch.no_grad():
        hin = blk.norm1(out.tokens[depth - 2].detach())
        H, hd = blk.attn.num_heads, blk.attn.head_dim
        qkv = blk.attn.qkv(hin).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        v = qkv[2].float()                                   # [B,H,N,hd]
        wv = A[-1][:, :, 0, :, None] * v                     # [B,H,N,hd]
        wv = wv.permute(0, 2, 1, 3).reshape(B, N, H * hd)
        f = wv @ blk.attn.proj.weight.float().t()
        s["attnnorm_last"] = f.norm(dim=-1)[:, 1:]
        s["vnorm_last"] = (v.permute(0, 2, 1, 3).reshape(B, N, H * hd) @ blk.attn.proj.weight.float().t()
                           ).norm(dim=-1)[:, 1:]
        s["toknorm_L-2"] = out.tokens[depth - 2].detach().float().norm(dim=-1)[:, 1:]
    s["anti_maskedkd"] = -s["cls_attn_L11"]
    return s, logits.detach()


def run(api):
    runs = api.runs()
    pick = []
    for rid, r in runs.items():
        if r["tier"] == "screen" and r["seed"] == 0 and r["method"] == "maskedkd" and r["keep"] == 98:
            pick += [(f"mkd98_e{e}", rid, e) for e in r["snapshots"]] + [("mkd98_final", rid, None)]
        if r["tier"] == "screen" and r["seed"] == 0 and r["method"] == "full":
            pick.append(("full_final", rid, None))
    api.write_json("students", {n: [rid, e] for n, rid, e in pick})
    T = api.teacher.eval()
    n = api.num_patches
    rr = torch.arange(n)
    c = (SIDE - 1) / 2
    cdist_all = (((rr // SIDE - c) ** 2 + (rr % SIDE - c) ** 2).float().sqrt())
    results = {}
    for sname, rid, ep in pick:
        student = api.student(rid, ep)
        for p in student.parameters():
            p.requires_grad_(False)
        acc = {}
        for x, y in api.val_batches(batch=100, limit=N_IMG):
            B = x.shape[0]
            dev = x.device
            cd = cdist_all.to(dev)
            with torch.no_grad():
                t_full = T(x).float().log_softmax(-1)
                t_attn = api.student_forward(T, x, attn_layers=[-1]).attn[-1].float().mean(1)[:, 0, 1:]
            sig, s_logits = _signals(api, student, x)
            s_log = s_logits.log_softmax(-1)
            sig["oracle"] = t_attn
            sig["random"] = torch.rand(B, n, device=dev)
            for k in KS:
                local = dict(sig)
                gi = torch.linspace(0, n - 1, k).round().long()
                gs = torch.zeros(n)
                gs[gi] = 1.0
                local["grid"] = gs.to(dev).expand(B, n)
                local["center"] = (-cd - rr.to(dev).float() * 1e-6).expand(B, n)
                mk = _topk(sig["cls_attn_L11"], k)
                for name, sc in local.items():
                    keep = _topk(sc.float(), k)
                    with torch.no_grad():
                        t_m = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    row = acc.setdefault(f"{name}@{k}", {"fid_kl": 0.0, "agree": 0.0, "t_acc": 0.0, "gap_kl": 0.0,
                                                         "overlap": 0.0, "cdist": 0.0, "n": 0})
                    row["fid_kl"] += _kl(t_full, t_m).sum().item()
                    row["agree"] += (t_m.argmax(-1) == t_full.argmax(-1)).float().sum().item()
                    row["t_acc"] += (t_m.argmax(-1) == y).float().sum().item()
                    row["gap_kl"] += _kl(t_m, s_log).sum().item()
                    row["overlap"] += ((keep & mk).sum(1).float() / k).sum().item()
                    row["cdist"] += ((keep.float() * cd).sum(1) / k).sum().item()
                    row["n"] += B
        out = {}
        for key, row in acc.items():
            m = row.pop("n")
            out[key] = {a: round(b / m, 4) for a, b in row.items()}
        results[sname] = out
        api.write_json(f"fid_{sname}", out)
    # compact ranking per student and K by fid_kl (lower is better)
    rank = {}
    for sname, out in results.items():
        for k in KS:
            rows = sorted(((key.split("@")[0], v["fid_kl"], v["t_acc"]) for key, v in out.items()
                           if key.endswith(f"@{k}")), key=lambda t: t[1])
            rank[f"{sname}@{k}"] = [[a, b, c2] for a, b, c2 in rows]
    api.write_json("ranking_by_fid_kl", rank)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(KS), figsize=(5 * len(KS), 3.5))
        for ax, k in zip(axes, KS):
            for sname, out in results.items():
                ys = [out[f"cls_attn_L{l}@{k}"]["fid_kl"] for l in range(12)]
                ax.plot(range(12), ys, marker="o", label=sname)
            ax.set_title(f"CLS attention layer vs teacher fidelity, K={k}")
            ax.set_xlabel("student layer")
            ax.set_ylabel("KL(T_full || T_masked)")
        axes[0].legend(fontsize=7)
        api.write_figure("layer_sweep", fig)
    except Exception as e:  # figure is optional
        api.write_json("figure_error", {"error": str(e)})
