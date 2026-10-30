"""H-W7: does MaskedKD's block-11 top-K hand the teacher more *place* (background) information than block-9
top-K or rollout, and is that what costs worst-group accuracy?

analysis_register_tokens_1030_153157_efbe: block-11 top-K is enriched in high-norm tokens (hn_in 0.18 at K=49,
0.50 at K=12, base rate ~0.05), but dropping them does not restore the masked teacher's worst group
(L11 48.87 vs L11_nohn 49.62 vs L9 72.18, full student, K=49). So register tokens are not the cause. The
remaining reading of H-W1 is content: the last block's peaks sit on background that carries the Waterbirds
place shortcut. Linear probing (Alain & Bengio 2017) of the masked teacher's pre-logit feature tests that:
  place_bacc   within-class balanced accuracy of a linear probe predicting place (land/water) from the masked
               teacher's pre-logit feature (2-fold CV over val, per class, averaged over the 2 classes).
               Higher = the teacher's representation of the masked image carries more background.
  cos_full     cosine similarity of the masked teacher's pre-logit feature to the full-image one, per group.
  t_acc, t_wga masked-teacher accuracy per group (as in group_fidelity).
Selectors: student CLS attention top-K at blocks 8/9/10/11, student rollout, center; reference: full
teacher (196 tokens). K in 29, 49. Students: screen seed-0 full KD and maskedkd@49 (final).
Groups are used only here (analysis), never by a selector.
"""
import torch

META = {
    "question": "does last-block CLS-attention top-K give the masked teacher more background (place) information "
                "than mid-block top-K or rollout, explaining the worst-group loss?",
    "method": "linear probing of the masked teacher's representation for a nuisance attribute (place), "
              "within class; group-sliced faithfulness of the masked prediction",
    "paper": "Alain & Bengio, Understanding intermediate layers using linear classifier probes, ICLR-W 2017; "
             "Sagawa et al. ICLR 2020 (Waterbirds groups); Abnar & Zuidema ACL 2020 (rollout)",
}

KS = (29, 49)
LAYERS = tuple(range(12))


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _teacher_feat(T, x, keep):
    """Pre-logit feature and logits of the teacher on the kept patches (same path as teacher_forward_masked)."""
    h = T.patch_embed(x)
    h = T._pos_embed(h)
    cls, patches = h[:, :1], h[:, 1:]
    if keep is not None:
        k = int(keep.sum(1)[0].item())
        idx = keep.float().argsort(dim=1, descending=True, stable=True)[:, :k]
        patches = torch.gather(patches, 1, idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
    h = torch.cat([cls, patches], dim=1)
    h = T.patch_drop(h)
    h = T.norm_pre(h)
    h = T.blocks(h)
    h = T.norm(h)
    f = T.forward_head(h, pre_logits=True)
    return f, T.head(f)


def _rollout(attn):
    R = None
    for a in attn:
        a = a.float().mean(1)
        n = a.shape[-1]
        a = 0.5 * a + 0.5 * torch.eye(n, device=a.device)
        a = a / a.sum(-1, keepdim=True)
        R = a if R is None else a @ R
    return R[:, 0, 1:]


def _probe_bacc(F, t, lam=100.0):
    """2-fold CV (index parity) class-balanced ridge probe (closed form, targets +-1);
    balanced accuracy over the two values of t."""
    n = F.shape[0]
    folds = torch.arange(n, device=F.device) % 2
    correct = torch.zeros(2, device=F.device)
    count = torch.zeros(2, device=F.device)
    for f in (0, 1):
        tr, te = folds != f, folds == f
        mu, sd = F[tr].mean(0), F[tr].std(0).clamp(min=1e-6)
        Xtr, Xte = (F[tr] - mu) / sd, (F[te] - mu) / sd
        ttr = t[tr].double() * 2 - 1
        sw = torch.where(ttr > 0, 1.0 / (ttr > 0).sum().clamp(min=1), 1.0 / (ttr < 0).sum().clamp(min=1))
        X = torch.cat([Xtr.double(), torch.ones(Xtr.shape[0], 1, device=F.device, dtype=torch.float64)], 1)
        A = X.T @ (X * sw[:, None]) + lam / Xtr.shape[0] * torch.eye(X.shape[1], device=F.device,
                                                                     dtype=torch.float64)
        wb = torch.linalg.solve(A, X.T @ (sw * ttr))
        w, b = wb[:-1].float(), wb[-1].float()
        with torch.no_grad():
            pred = ((Xte @ w + b) > 0).long()
            for v in (0, 1):
                m = t[te] == v
                correct[v] += (pred[m] == v).sum()
                count[v] += m.sum()
    return (correct / count.clamp(min=1)).mean().item()


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
        feats, oks, ys = {}, {}, []
        off = 0
        for x, y in api.val_batches(batch=25):
            B = x.shape[0]
            off += B
            ys.append(y)
            with torch.no_grad():
                out = api.student_forward(student, x, attn_layers=list(LAYERS))
                cls = {l: out.attn[l].float().mean(1)[:, 0, 1:] for l in (8, 9, 10, 11)}
                roll = _rollout(out.attn)
                cdist = None
                masks = {"full": None}
                for k in KS:
                    for l in (8, 9, 10, 11):
                        masks[f"L{l}@{k}"] = _topk(cls[l], k)
                    masks[f"rollout@{k}"] = _topk(roll, k)
                    if cdist is None:
                        g = int(api.num_patches ** 0.5)
                        r_, c_ = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
                        cdist = (((r_ - (g - 1) / 2) ** 2 + (c_ - (g - 1) / 2) ** 2).flatten().float()
                                 .to(x.device))
                    masks[f"center@{k}"] = _topk((-cdist).expand(B, -1), k)
                for name, keep in masks.items():
                    f, lg = _teacher_feat(T, x, keep)
                    feats.setdefault(name, []).append(f.float())
                    oks.setdefault(name, []).append((lg.argmax(-1) == y).float())
        y = torch.cat(ys)
        g = groups[:len(y)].to(y.device)
        place = g % 2
        Ffull = torch.cat(feats["full"])
        res = {}
        for name in feats:
            F = torch.cat(feats[name])
            ok = torch.cat(oks[name])
            bacc = [_probe_bacc(F[y == c], place[y == c]) for c in (0, 1)]
            cos = torch.nn.functional.cosine_similarity(F, Ffull, dim=-1)
            acc = [round(100 * ok[g == gi].mean().item(), 2) for gi in range(4)]
            res[name] = {"place_bacc": round(100 * sum(bacc) / 2, 2),
                         "place_bacc_by_class": [round(100 * b, 2) for b in bacc],
                         "cos_full": [round(cos[g == gi].mean().item(), 4) for gi in range(4)],
                         "t_acc": acc, "t_wga": min(acc)}
        api.write_json(f"probe_{sname}", res)
        del student, feats
        torch.cuda.empty_cache()
