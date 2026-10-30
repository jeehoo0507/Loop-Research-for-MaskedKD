"""H-W9: is `center` strong because the teacher needs one compact contiguous region, and can the student's
attention *place* that region on the object better than the fixed image centre?

Basis (screen seed 0, val acc / WGA): center beats MaskedKD at every K measured (K=49 90.16/60.15 vs
89.24/51.88, K=29 88.99/62.41 vs 87.66/43.61), and at K=12 center is ahead of rollout on WGA in its training log
(epoch 95: 55.64 vs rollout final 45.11, screen_center_k12_s0_1030_135953_c64e, screen_rollout_k12_s0_1030_135953_489a).
Scattered top-K (block-9 / block-11 CLS attention) keeps the teacher accurate on clean val images but did not
carry over to the student (midattn@49 57.89 WGA). A disc is what every center run feeds the teacher.

During training the DeiT RandomResizedCrop moves the bird off the image centre, which is where a window placed
by the student can differ from `center`. Conditions:
  clean    val image as is (centre crop)
  crop     deterministic off-centre crop (area 0.45..0.7 of the image, offset cycled per batch) resized to
           224: a stand-in for RandomResizedCrop, label unchanged
Selections (per condition, K in 12, 29, 49):
  center              K patches nearest the image centre (builtin rule)
  win_<map>_cent      K patches nearest the attention-weighted centroid of <map>
  win_<map>_peak      K patches nearest the argmax of the Gaussian-blurred <map>
  top_<map>           plain top-K of <map> (scattered; L11 = MaskedKD, rollout = builtin)
  shift_best          analysis-only upper bound: best of 25 shifted discs per image by KL to the full teacher
maps: L9, L11 (head-mean CLS attention), rollout (Abnar & Zuidema 2020).
Per group of the val image: masked-teacher accuracy (t_acc, t_wga), KL(full || masked) on the same input,
plus shift (distance of the window centre from the image centre, in patches).
Students: screen seed-0 full@196, maskedkd@49, rollout@49 (final), center@49 (final).
"""
import torch
import torch.nn.functional as F

META = {
    "question": "does a compact window placed at the student's attention centroid / peak keep the masked teacher "
                "accurate on the worst group better than the fixed centre window, on clean and off-centre crops?",
    "method": "attention maps (CLS attention, attention rollout) used for weakly-supervised object localisation "
              "(centroid / blurred peak) instead of per-token ranking; group-sliced masked-teacher faithfulness",
    "paper": "Abnar & Zuidema ACL 2020; Gao et al. TS-CAM ICCV 2021; Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

KS = (12, 29, 49)
SIDE = 14
SHIFTS = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
CROP_AREA = (0.45, 0.55, 0.7)


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
    d = (rows[None] - cy[:, None]) ** 2 + (cols[None] - cx[:, None]) ** 2
    d = d + torch.arange(d.shape[1], device=d.device).float()[None] * 1e-6
    return _topk(-d, k)


def _crop(x, bi):
    """Deterministic off-centre crop resized back to 224 (no RNG)."""
    H = x.shape[-1]
    a = CROP_AREA[bi % len(CROP_AREA)]
    s = int(round(H * a ** 0.5))
    span = H - s
    fy = ((7 * bi) % 5) / 4.0
    fx = ((3 * bi + 1) % 5) / 4.0
    y0, x0 = int(round(fy * span)), int(round(fx * span))
    return F.interpolate(x[:, :, y0:y0 + s, x0:x0 + s], size=(H, H), mode="bilinear", align_corners=False)


def _maps(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
    A = [a.float().mean(1) for a in out.attn]
    B, N, _ = A[0].shape
    eye = torch.eye(N, device=x.device).expand(B, N, N)
    R = eye.clone()
    for a in A:
        a = 0.5 * a + 0.5 * eye
        R = (a / a.sum(-1, keepdim=True)) @ R
    return {"L9": A[9][:, 0, 1:], "L11": A[11][:, 0, 1:], "rollout": R[:, 0, 1:]}


def run(api):
    runs = api.runs()
    pick = []
    for rid, r in runs.items():
        if r["tier"] != "screen" or r["seed"] != 0 or r.get("dataset", "waterbirds") != "waterbirds":
            continue
        if (r["method"], r["keep"]) in (("full", 196), ("maskedkd", 49), ("rollout", 49), ("center", 49)):
            pick.append((f"{r['method']}{r['keep']}_final", rid))
    api.write_json("students", {n: rid for n, rid in pick})
    groups = api.val_groups()
    T = api.teacher.eval()
    n = api.num_patches
    rr = torch.arange(n)
    rows_c, cols_c = (rr // SIDE).float(), (rr % SIDE).float()
    c = (SIDE - 1) / 2
    conds = ("clean", "crop")

    # inputs, full-teacher targets and the student-free selections (center, shift_best) are built once
    batches = []
    ref_n = [0] * 4
    fixed = {}
    off = 0
    for bi, (x, y) in enumerate(api.val_batches(batch=50)):
        dev = x.device
        rows, cols = rows_c.to(dev), cols_c.to(dev)
        B = x.shape[0]
        gb = groups[off:off + B].to(dev)
        off += B
        for gi in range(4):
            ref_n[gi] += int((gb == gi).sum().item())
        ins = {"clean": x, "crop": _crop(x, bi)}
        full = {}
        for cond, xi in ins.items():
            with torch.no_grad():
                full[cond] = T(xi).float().log_softmax(-1)
            cc = torch.full((B,), c, device=dev)
            for k in KS:
                keep = _disc(cc, cc, k, rows, cols)
                with torch.no_grad():
                    t_m = api.teacher_forward_keepmask(T, xi, keep).float().log_softmax(-1)
                _acc(fixed, f"{cond}/center@{k}", t_m, full[cond], y, gb, cc, cc, c)
                best_kl = torch.full((B,), float("inf"), device=dev)
                best_lp = None
                by, bx = cc.clone(), cc.clone()
                for dy, dx in SHIFTS:
                    keep = _disc(cc + dy, cc + dx, k, rows, cols)
                    with torch.no_grad():
                        lp = api.teacher_forward_keepmask(T, xi, keep).float().log_softmax(-1)
                    kl = _kl(full[cond], lp)
                    better = kl < best_kl
                    best_kl = torch.where(better, kl, best_kl)
                    best_lp = lp if best_lp is None else torch.where(better[:, None], lp, best_lp)
                    by = torch.where(better, cc + dy, by)
                    bx = torch.where(better, cc + dx, bx)
                _acc(fixed, f"{cond}/shift_best@{k}", best_lp, full[cond], y, gb, by, bx, c)
        batches.append(({k: v.cpu() for k, v in ins.items()}, {k: v.cpu() for k, v in full.items()},
                        y.cpu(), gb.cpu(), dev))
    api.write_json("fixed", _finish(fixed, ref_n))

    summary = {"group_n": ref_n, "cols": ["t_acc_mean", "t_wga", "kl_mean", "kl_worst", "shift"],
               "fixed": {key: _row(v) for key, v in _finish(fixed, ref_n).items()}}
    for sname, rid in pick:
        student = api.student(rid, None)
        for p in student.parameters():
            p.requires_grad_(False)
        student.eval()
        acc = {}
        for ins, full, y, gb, dev in batches:
            rows, cols = rows_c.to(dev), cols_c.to(dev)
            y, gb = y.to(dev), gb.to(dev)
            for cond in conds:
                xi = ins[cond].to(dev)
                fl = full[cond].to(dev)
                maps = _maps(api, student, xi)
                for mname, mp in maps.items():
                    w = mp / mp.sum(1, keepdim=True)
                    cy, cx = (w * rows).sum(1), (w * cols).sum(1)
                    am = _blur(mp).argmax(1)
                    py, px = rows[am], cols[am]
                    for k in KS:
                        sels = {f"win_{mname}_cent": (_disc(cy, cx, k, rows, cols), cy, cx),
                                f"win_{mname}_peak": (_disc(py, px, k, rows, cols), py, px),
                                f"top_{mname}": (_topk(mp, k), None, None)}
                        for sel, (keep, sy, sx) in sels.items():
                            with torch.no_grad():
                                t_m = api.teacher_forward_keepmask(T, xi, keep).float().log_softmax(-1)
                            _acc(acc, f"{cond}/{sel}@{k}", t_m, fl, y, gb, sy, sx, c)
        out = _finish(acc, ref_n)
        api.write_json(f"win_{sname}", out)
        summary[sname] = {key: _row(v) for key, v in out.items()}
    api.write_json("summary", summary)


def _acc(store, key, t_m, full, y, gb, sy, sx, c):
    row = store.setdefault(key, {"kl": [0.0] * 4, "t_acc": [0.0] * 4, "shift": 0.0, "n": 0})
    kl = _kl(full, t_m)
    ok = (t_m.argmax(-1) == y).float()
    for gi in range(4):
        m = gb == gi
        row["kl"][gi] += kl[m].sum().item()
        row["t_acc"][gi] += ok[m].sum().item()
    if sy is not None:
        row["shift"] += ((sy - c) ** 2 + (sx - c) ** 2).sqrt().sum().item()
    row["n"] += y.shape[0]


def _finish(store, ref_n):
    out = {}
    for key, row in store.items():
        n_img = row["n"]
        t = [round(100 * v / max(b, 1), 2) for v, b in zip(row["t_acc"], ref_n)]
        kl = [round(v / max(b, 1), 4) for v, b in zip(row["kl"], ref_n)]
        out[key] = {"t_acc": t, "t_wga": min(t), "t_acc_mean": round(100 * sum(row["t_acc"]) / n_img, 2),
                    "kl": kl, "kl_mean": round(sum(row["kl"]) / n_img, 4), "kl_worst": max(kl),
                    "shift": round(row["shift"] / n_img, 3)}
    return out


def _row(v):
    return [v["t_acc_mean"], v["t_wga"], v["kl_mean"], v["kl_worst"], v["shift"]]
