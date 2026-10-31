"""H-C1: on CUB-200, which per-token student score keeps the masked teacher's soft label closest to the full
teacher's, and in which part of the soft label (target vs non-target classes) is the loss?

Motivation (QUEUE 2026-10-31 18:30): the next method's grounding must connect to mean accuracy and FLOPs, not
WGA. The KD signal the student receives is the teacher's soft label; with 200 classes most of it sits in the
non-target classes (Hinton et al. 2015 "dark knowledge"). DKD (Zhao et al. CVPR 2022) splits
KL(p_full || p_mask) exactly into
  TCKD = KL(Bern(p_full[y]) || Bern(p_mask[y]))                     (target vs rest)
  NCKD = (1 - p_full[y]) * KL(p_full[-y]/(1-p_full[y]) || p_mask[-y]/(1-p_mask[y]))   (among non-target classes)
so KL = TCKD + NCKD. This tells which part a selector loses.

Students: the CUB runs that exist now (full@196 and maskedkd@98, final seed 1). All scores are per-token and
come from the student's own computation, so they apply wherever the bird is (no position prior):
  maskedkd   last-block CLS attention (head mean)
  rollout    attention rollout over all blocks (Abnar & Zuidema 2020)
  attnvnorm  last-block CLS attention x ||W_o v_j|| (Kobayashi et al. EMNLP 2020)
  gxi_top1   |<grad, h>| of the student's top-1 logit wrt input patch tokens (grad x input)
  gxi_dark   |<grad, h>| of logsumexp of the student's next 10 logits (ranks 2-11), i.e. the non-target part
             of the student's own soft label
Spatial / reference rules: center, grid, random (no signal); oracle = teacher last-block CLS attention (upper
reference only, never a claim). Ks: 98, 49, 25, 12. Nothing here is validated against students yet: the CUB
baselines needed for a rank correlation do not exist. This is the descriptive half of H-C1.
"""
import torch

META = {
    "question": "on CUB, which position-free per-token student score (CLS attention, rollout, attention x value "
                "norm, grad x input of top-1 or of non-target logits) best preserves the full teacher's soft "
                "label under masking, and is the loss in the target (TCKD) or non-target (NCKD) part?",
    "method": "keep-only masked-teacher faithfulness (KL to full teacher, DKD decomposition, top-1/top-5 "
              "agreement) under attention, rollout, value-norm-weighted attention and grad x input token scores",
    "paper": "Hinton et al. 2015 (KD); Zhao et al. CVPR 2022 (DKD); Abnar & Zuidema ACL 2020 (rollout); "
             "Kobayashi et al. EMNLP 2020 (attention x value norm); Shrikumar et al. 2016 / Ancona et al. "
             "ICLR 2018 (grad x input); Hooker et al. NeurIPS 2019 (keep-only evaluation); Son et al. ECCV 2024 "
             "(MaskedKD)",
}

KS = (98, 49, 25, 12)
SIDE = 14
SCORES = ("maskedkd", "rollout", "attnvnorm", "gxi_top1", "gxi_dark", "oracle", "center", "grid", "random")
EPS = 1e-8


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _fixed(method, B, n, k, dev, off):
    if method == "center":
        r = torch.arange(n, device=dev)
        c = (SIDE - 1) / 2
        dist = ((r // SIDE - c) ** 2 + (r % SIDE - c) ** 2).float() + r.float() * 1e-6
        return _topk((-dist).expand(B, n), k)
    if method == "grid":
        idx = torch.linspace(0, n - 1, k, device=dev).round().long()
        return torch.zeros(B, n, dtype=torch.bool, device=dev).index_fill_(1, idx, True)
    # random: fixed hash noise per (image, token), so both students see the same draw
    i = torch.arange(off, off + B, device=dev, dtype=torch.float64)[:, None]
    j = torch.arange(n, device=dev, dtype=torch.float64)[None, :]
    u = torch.frac(torch.sin(i * 12.9898 + j * 78.233) * 43758.5453)
    return _topk(u.float(), k)


def _attn_scores(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad():
        out = api.student_forward(student, x, attn_layers=list(range(depth)), token_layers=[depth - 2])
    A = [a.float().mean(1) for a in out.attn]
    B, N, _ = A[0].shape
    eye = torch.eye(N, device=x.device).expand(B, N, N)
    R = None
    for a in A:
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        R = a if R is None else a @ R
    # ||W_o v_j|| per head in the last block, from the input of the last block (output of block depth-2)
    blk = student.blocks[-1]
    with torch.no_grad():
        h = blk.norm1(out.tokens[depth - 2])
        att = blk.attn
        qkv = att.qkv(h).reshape(B, N, 3, att.num_heads, att.head_dim).permute(2, 0, 3, 1, 4)
        v = qkv[2].float()                                           # [B, H, N, hd]
        W = att.proj.weight.float().reshape(-1, att.num_heads, att.head_dim)   # [D, H, hd]
        fv = torch.einsum("bhnd,ehd->bhne", v, W).norm(dim=-1)        # [B, H, N]
        last = out.attn[-1].float()                                  # [B, H, N, N]
        avn = (last[:, :, 0, :] * fv).sum(1)                         # sum over heads of ||alpha * W_o v||
    return {"maskedkd": A[-1][:, 0, 1:], "rollout": R[:, 0, 1:], "attnvnorm": avn[:, 1:]}


def _gxi_scores(student, x):
    """grad x input on the patch tokens entering block 0, for the top-1 logit and for the next-10 logsumexp."""
    with torch.enable_grad():
        h0 = student.patch_embed(x).detach().requires_grad_(True)
        h = student._pos_embed(h0)
        h = student.norm_pre(student.patch_drop(h))
        h = student.norm(student.blocks(h))
        logits = student.forward_head(h).float()
        srt = logits.sort(-1, descending=True).values
        top1 = srt[:, 0].sum()
        dark = srt[:, 1:11].logsumexp(-1).sum()
        g1 = torch.autograd.grad(top1, h0, retain_graph=True)[0]
        g2 = torch.autograd.grad(dark, h0)[0]
    return {"gxi_top1": (g1.float() * h0.float()).sum(-1).abs().detach(),
            "gxi_dark": (g2.float() * h0.float()).sum(-1).abs().detach()}


def _oracle(api, T, x):
    with torch.no_grad():
        h = T.patch_embed(x)
        h = T._pos_embed(h)
        h = T.norm_pre(T.patch_drop(h))
        for b in T.blocks[:-1]:
            h = b(h)
        _, w = api.attn_with_weights(T.blocks[-1].attn, T.blocks[-1].norm1(h))
    return w.float().mean(1)[:, 0, 1:]


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
        if r.get("dataset") != "cub" or r["seed"] != 1:
            continue
        if r["method"] == "full" and int(r["keep"]) == 196:
            studs["full196_s1"] = rid
        if r["method"] == "maskedkd" and int(r["keep"]) == 98:
            studs["maskedkd98_s1"] = rid
    api.write_json("students", {k: [v, runs[v]["val_acc_last"]] for k, v in studs.items()})

    T = api.teacher.eval()
    n = api.num_patches
    batches = []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        orc = _oracle(api, T, x)
        batches.append((x, y, full, orc))
    N = sum(b[0].shape[0] for b in batches)
    ref = {}
    ref["teacher_full_acc"] = round(100 * sum((b[2].argmax(-1) == b[1]).float().sum().item() for b in batches) / N, 2)
    ref["teacher_full_p_true"] = round(sum(b[2].exp().gather(1, b[1][:, None]).sum().item() for b in batches) / N, 4)
    ref["teacher_full_nontarget_mass"] = round(1 - ref["teacher_full_p_true"], 4)
    api.write_json("reference", ref)

    results = {}
    overlap = {}
    for sname, rid in studs.items():
        student = api.student(rid)
        acc = {(m, k): {} for m in SCORES for k in KS}
        ov = {(m, k): 0.0 for m in SCORES for k in KS}
        off = 0
        for x, y, full, orc in batches:
            B = x.shape[0]
            sc = _attn_scores(api, student, x)
            sc.update(_gxi_scores(student, x))
            sc["oracle"] = orc
            for k in KS:
                keeps = {}
                for m in SCORES:
                    keeps[m] = _fixed(m, B, n, k, x.device, off) if m in ("center", "grid", "random") else _topk(sc[m], k)
                for m, keep in keeps.items():
                    with torch.no_grad():
                        lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    for s, v in _stats(full, lm, y).items():
                        acc[(m, k)][s] = acc[(m, k)].get(s, 0.0) + v.sum().item()
                    ov[(m, k)] += (keep & keeps["maskedkd"]).sum().item() / k
            off += B
        res = {}
        for (m, k), d in acc.items():
            o = {s: round(v / N, 4) for s, v in d.items()}
            for s in ("top1_agree", "top5_overlap", "t_acc"):
                o[s] = round(100 * o[s], 2)
            o["overlap_with_maskedkd"] = round(ov[(m, k)] / N, 4)
            res[f"{m}@{k}"] = o
        results[sname] = res
        # rank of scores by KL at each K (lower = closer to the full teacher), excluding oracle
        overlap[sname] = {str(k): sorted(((res[f"{m}@{k}"]["kl"], m) for m in SCORES if m != "oracle"))
                          for k in KS}
        api.write_json(f"fidelity_{sname}", res)
    api.write_json("kl_rank", overlap)
    api.write_json("summary", {"students": studs, "reference": ref, "n_val": N, "ks": list(KS),
                               "kl_rank": overlap})
