"""H-W14: why do centre-prior selections (priorroll, center) lose to scattered rollout at K=29 when the clean-image
masked-teacher KL says the opposite?

analysis_proxy_holdout_1030_182326_7254: on each run's own student, self kl_worst at K=29 is priorroll 0.391 <
rollout 0.443 < center 0.517, but student val WGA is rollout 65.41 > center 62.41 > priorroll 60.15 (screen seed 0).
The proxy (picked on 16 runs in analysis_proxy_validation_1030_170914_4668) fails out of sample.

Two candidate causes, each measured on the run's own final student:
 (a) training-input gap: during training every batch is cutmix or mixup. A compact central selection may drop the
     pasted image, so the KD target misses half of the mixed label even though it is faithful on clean images.
     Measured as in analysis_train_condition_gap_1030_162956_067e: kl_worst / agree_worst on cutmix and mixup
     inputs, and the kept-token share inside the cutmix box vs the box area.
 (b) student shortcut: the student itself encodes more background. Measured with a within-class place probe
     (closed-form ridge, 2-fold CV, balanced accuracy) on the student's pre-logit feature of clean val images,
     and the student's own group accuracies.
Runs: center / rollout / priorroll at K=12 and 29, suffdisc@12, screen seed 0.

Pre-stated reading: (a) explains the failure if at K=29 the cutmix/mixup kl_worst (or the box-share deficit)
orders rollout < center <= priorroll, i.e. the same order as student WGA. (b) is a description of the outcome,
not a cause, and is reported only to see whether the WGA gap shows up as more place information in the student.
"""
import math

import torch

META = {
    "question": "is the out-of-sample failure of the clean masked-teacher KL proxy for centre-prior selections "
                "explained by the cutmix/mixup inputs seen in training, and does it show as background (place) "
                "information in the student's own features?",
    "method": "group-sliced keep-only faithfulness of the masked teacher under cutmix/mixup inputs for attention "
              "rollout with and without a centre prior; linear place probe on student features",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Yun et al. ICCV 2019 (CutMix); Zhang et al. ICLR 2018 (mixup); "
             "Alain & Bengio ICLR-W 2017 (linear probes); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

SIDE = 14
LAM_CUT = (0.3, 0.5, 0.7)
LAM_MIX = 0.6
NAMES = ("center@12", "rollout@12", "priorroll@12", "suffdisc@12", "center@29", "rollout@29", "priorroll@29")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def _box(bi):
    lam = LAM_CUT[bi % len(LAM_CUT)]
    s = max(1, int(round(((1 - lam) ** 0.5) * SIDE)))
    r0 = (3 * bi) % (SIDE - s + 1)
    c0 = (5 * bi + 2) % (SIDE - s + 1)
    inside = torch.zeros(SIDE, SIDE, dtype=torch.bool)
    inside[r0:r0 + s, c0:c0 + s] = True
    return r0, c0, s, inside.flatten()


def _geom(n, dev):
    rr = torch.arange(n, device=dev)
    c = (SIDE - 1) / 2
    return rr, (rr // SIDE).float(), (rr % SIDE).float(), c


def _center(B, n, k, dev):
    rr, rows, cols, c = _geom(n, dev)
    d = (rows - c) ** 2 + (cols - c) ** 2 + rr.float() * 1e-6
    return _topk((-d).expand(B, n), k)


def _rollout(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
    R = None
    for a in out.attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        R = a if R is None else a @ R
    return R[:, 0, 1:]


def _priorroll(R, k):
    rr, rows, cols, c = _geom(R.shape[1], R.device)
    d2 = (rows - c) ** 2 + (cols - c) ** 2
    s = math.sqrt(k / math.pi)
    return _topk(R.clamp_min(1e-12).log() - d2 / (2 * s * s), k)


def _masked_logits(model, x, idx):
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    cls, patches = h[:, :1], h[:, 1:]
    D = patches.shape[-1]
    patches = torch.gather(patches, 1, idx.unsqueeze(-1).expand(-1, -1, D))
    h = torch.cat([cls, patches], dim=1)
    h = model.norm_pre(h)
    h = model.blocks(h)
    h = model.norm(h)
    return model.forward_head(h)


def _suffdisc(student, x, k, r=2):
    B, dev, n = x.shape[0], x.device, SIDE * SIDE
    rr, rows, cols, c = _geom(n, dev)
    tie = rr.float() * 1e-6
    with torch.no_grad():
        full = student(x).float().log_softmax(-1)
        best_kl = torch.full((B,), float("inf"), device=dev)
        best_idx = None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d = (rows - (c + dy)) ** 2 + (cols - (c + dx)) ** 2 + tie
                idx = (-d).topk(k).indices.sort().values.expand(B, -1)
                lp = _masked_logits(student, x, idx).float().log_softmax(-1)
                kl = (full.exp() * (full - lp)).sum(-1)
                better = kl < best_kl
                best_kl = torch.where(better, kl, best_kl)
                best_idx = idx.clone() if best_idx is None else torch.where(better[:, None], idx, best_idx)
    return torch.zeros(B, n, dtype=torch.bool, device=dev).scatter_(1, best_idx, True)


def _probe_bacc(F, t, lam=100.0):
    """2-fold CV (index parity) class-balanced ridge probe (closed form, targets +-1); balanced accuracy."""
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
        pred = ((Xte @ w + b) > 0).long()
        for v in (0, 1):
            m = t[te] == v
            correct[v] += (pred[m] == v).sum()
            count[v] += m.sum()
    return (correct / count.clamp(min=1)).mean().item()


def run(api):
    runs = api.runs()
    pick = {}
    for rid, r in sorted(runs.items()):
        if r["tier"] != "screen" or r["seed"] != 0 or r.get("dataset", "waterbirds") != "waterbirds":
            continue
        name = f"{r['method'].split('@')[0]}@{int(r['keep'])}"
        if name in NAMES:
            pick[name] = (rid, r["val_wga_last"])
    api.write_json("targets", pick)

    groups = api.val_groups()
    T = api.teacher.eval()
    n = api.num_patches
    dev = api.device
    batches = []
    off = 0
    for bi, (x, y) in enumerate(api.val_batches(batch=50)):
        B = x.shape[0]
        gb = groups[off:off + B]
        off += B
        x2 = x.roll(1, 0)
        r0, c0, s, inside = _box(bi)
        xc = x.clone()
        xc[:, :, 16 * r0:16 * (r0 + s), 16 * c0:16 * (c0 + s)] = x2[:, :, 16 * r0:16 * (r0 + s), 16 * c0:16 * (c0 + s)]
        xm = LAM_MIX * x + (1 - LAM_MIX) * x2
        ins = {"clean": x, "cutmix": xc, "mixup": xm}
        with torch.no_grad():
            full = {c_: T(v).float().log_softmax(-1) for c_, v in ins.items()}
        batches.append(({c_: v.cpu() for c_, v in ins.items()}, {c_: v.cpu() for c_, v in full.items()},
                        y.cpu(), gb.cpu(), inside))
    gn = [sum(int((b[3] == gi).sum()) for b in batches) for gi in range(4)]
    area = sum(_box(bi)[3].float().mean().item() for bi in range(len(batches))) / len(batches)

    summary = {}
    for name in NAMES:
        if name not in pick:
            continue
        rid, wga = pick[name]
        method, k = name.split("@")[0], int(name.split("@")[1])
        st = api.student(rid)
        for p in st.parameters():
            p.requires_grad_(False)
        st.eval()

        def select(x):
            if method == "center":
                return _center(x.shape[0], n, k, dev)
            if method == "suffdisc":
                return _suffdisc(st, x, k)
            R = _rollout(api, st, x)
            return _topk(R, k) if method == "rollout" else _priorroll(R, k)

        acc = {c_: {"kl": [0.0] * 4, "agree": [0.0] * 4} for c_ in ("clean", "cutmix", "mixup")}
        box_share = 0.0
        feats, ys, gs, s_ok = [], [], [], []
        for ins, full, y, gb, inside in batches:
            ins = {c_: v.to(dev) for c_, v in ins.items()}
            full = {c_: v.to(dev) for c_, v in full.items()}
            y, gb, inside = y.to(dev), gb.to(dev), inside.to(dev)
            for c_ in ("clean", "cutmix", "mixup"):
                keep = select(ins[c_])
                with torch.no_grad():
                    lm = api.teacher_forward_keepmask(T, ins[c_], keep).float().log_softmax(-1)
                kl = _kl(full[c_], lm)
                ag = (lm.argmax(-1) == full[c_].argmax(-1)).float()
                for gi in range(4):
                    m = gb == gi
                    acc[c_]["kl"][gi] += kl[m].sum().item()
                    acc[c_]["agree"][gi] += ag[m].sum().item()
                if c_ == "cutmix":
                    box_share += ((keep & inside).sum(1).float() / k).sum().item()
            with torch.no_grad():
                h = st.forward_features(ins["clean"])
                f = st.forward_head(h, pre_logits=True).float()
                logit = st.forward_head(h).float()
            feats.append(f)
            ys.append(y)
            gs.append(gb)
            s_ok.append((logit.argmax(-1) == y).float())
        F, Y, G, OK = torch.cat(feats), torch.cat(ys), torch.cat(gs), torch.cat(s_ok)
        place = G % 2
        probe = [_probe_bacc(F[Y == c_], place[Y == c_]) for c_ in (0, 1)]
        s_group = [round(100 * OK[G == gi].mean().item(), 2) for gi in range(4)]
        out = {"run": rid, "student_wga": wga, "student_group_acc": s_group,
               "place_probe_bacc": [round(100 * v, 2) for v in probe],
               "place_probe_mean": round(50 * (probe[0] + probe[1]), 2),
               "cutmix_box_share": round(box_share / sum(gn), 4)}
        for c_, row in acc.items():
            kl = [round(v / max(b, 1), 4) for v, b in zip(row["kl"], gn)]
            ag = [round(100 * v / max(b, 1), 2) for v, b in zip(row["agree"], gn)]
            out[c_] = {"kl": kl, "kl_worst": max(kl), "agree": ag, "agree_worst": min(ag)}
        summary[name] = out
        api.write_json("rows", summary)
        del st

    table = {nm: [o["student_wga"], o["clean"]["kl_worst"], o["cutmix"]["kl_worst"], o["mixup"]["kl_worst"],
                  o["cutmix_box_share"], o["place_probe_mean"]] for nm, o in summary.items()}
    api.write_json("summary", {"group_n": gn, "box_area_share_mean": round(area, 4),
                               "cols": ["student_wga", "clean kl_worst", "cutmix kl_worst", "mixup kl_worst",
                                        "cutmix box_share", "student place probe bacc"],
                               "table": table})
