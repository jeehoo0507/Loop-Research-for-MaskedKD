"""H-W11: can the *student's own sufficiency* (how well the student's prediction survives when it sees only the
kept tokens) place the teacher's input window, where student attention could not (H-W9)?

Basis:
  * analysis_proxy_validation_1030_170914_4668: across 16 screen runs the masked teacher's KL to the full teacher
    on the worst group (measured with each run's own student) rank-predicts student val WGA (|Spearman| 0.94;
    K=12 1.0, K=29 1.0, K=49 0.89). So teacher fidelity on the minority groups is the target to optimise.
  * analysis_window_fidelity_1030_165418_e276: a centred disc moved by at most 2 patches to the best of 25 positions
    (shift_best, uses the teacher, analysis only) cuts that KL at K=12 from 0.694 (center) to 0.049, clean.
    Attention centroids / peaks of L9, L11, rollout did not find those positions (kl_worst ~0.6 at K=12).
  * the student-side analogue of shift_best is cheap: a DeiT-Ti forward on 12..49 tokens.

Selections (K in 12, 29, 49; conditions clean and off-centre crop as in window_fidelity):
  center          fixed centre disc (builtin rule)
  suff_disc       of the 25 shifted discs, the one with the smallest KL(student full || student masked)
  suff_gate       of {center disc, top-K rollout, top-K L9, suff_disc}, the one with the smallest student KL
  top_rollout     builtin rollout top-K
  shift_best      analysis-only upper bound (teacher KL), for regret
Reported per group: masked-teacher accuracy, KL(full teacher || masked teacher); plus per-image Spearman between
the student KL and the teacher KL over the 25 discs, and how often the student picks the teacher's best disc.
Students (screen seed 0, final): full@196, center@12, rollout@12, maskedkd@12, center@49.
"""
import torch
import torch.nn.functional as F

META = {
    "question": "does the disc that best preserves the student's own prediction (student sufficiency) also keep "
                "the masked teacher faithful on the worst group, better than the fixed centre disc?",
    "method": "keep-only sufficiency / insertion faithfulness of a token subset (ERASER sufficiency, RISE "
              "insertion, ROAR keep) evaluated on the student and transferred to the teacher; attention rollout",
    "paper": "DeYoung et al. ACL 2020 (ERASER sufficiency); Petsiuk et al. BMVC 2018 (RISE); Hooker et al. "
             "NeurIPS 2019 (ROAR); Abnar & Zuidema ACL 2020; Sagawa et al. ICLR 2020",
}

KS = (12, 29, 49)
SIDE = 14
SHIFTS = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
CROP_AREA = (0.45, 0.55, 0.7)
STUDENTS = (("full", 196), ("center", 12), ("rollout", 12), ("maskedkd", 12), ("center", 49))


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def _disc(cy, cx, k, rows, cols):
    d = (rows[None] - cy[:, None]) ** 2 + (cols[None] - cx[:, None]) ** 2
    d = d + torch.arange(d.shape[1], device=d.device).float()[None] * 1e-6
    return _topk(-d, k)


def _crop(x, bi):
    H = x.shape[-1]
    a = CROP_AREA[bi % len(CROP_AREA)]
    s = int(round(H * a ** 0.5))
    span = H - s
    fy = ((7 * bi) % 5) / 4.0
    fx = ((3 * bi + 1) % 5) / 4.0
    y0, x0 = int(round(fy * span)), int(round(fx * span))
    return F.interpolate(x[:, :, y0:y0 + s, x0:x0 + s], size=(H, H), mode="bilinear", align_corners=False)


def _rank(v):
    return v.argsort(1).argsort(1).float()


def _spearman_rows(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(1, keepdim=True)
    rb = rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp_min(1e-8)


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
    return out.logits.float().log_softmax(-1), {"L9": A[9][:, 0, 1:], "rollout": R[:, 0, 1:]}


def run(api):
    runs = api.runs()
    pick = []
    for rid, r in runs.items():
        if r["tier"] != "screen" or r["seed"] != 0 or r.get("dataset", "waterbirds") != "waterbirds":
            continue
        if (r["method"], r["keep"]) in STUDENTS:
            pick.append((f"{r['method']}{r['keep']}", rid))
    pick.sort()
    api.write_json("students", {n: rid for n, rid in pick})
    groups = api.val_groups()
    T = api.teacher.eval()
    rr = torch.arange(api.num_patches)
    rows_c, cols_c = (rr // SIDE).float(), (rr % SIDE).float()
    c = (SIDE - 1) / 2
    conds = ("clean", "crop")

    # teacher side, student-free: full log-probs, and for every disc shift the teacher KL and argmax
    batches = []
    ref_n = [0] * 4
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
        full, tkl, tpred = {}, {}, {}
        cc = torch.full((B,), c, device=dev)
        for cond, xi in ins.items():
            with torch.no_grad():
                full[cond] = T(xi).float().log_softmax(-1)
            for k in KS:
                kls, prs = [], []
                for dy, dx in SHIFTS:
                    keep = _disc(cc + dy, cc + dx, k, rows, cols)
                    with torch.no_grad():
                        lp = api.teacher_forward_keepmask(T, xi, keep).float().log_softmax(-1)
                    kls.append(_kl(full[cond], lp))
                    prs.append(lp.argmax(-1))
                tkl[(cond, k)] = torch.stack(kls, 1).cpu()
                tpred[(cond, k)] = torch.stack(prs, 1).cpu()
        batches.append(({k: v.cpu() for k, v in ins.items()}, {k: v.cpu() for k, v in full.items()},
                        tkl, tpred, y.cpu(), gb.cpu(), dev))

    centre_idx = SHIFTS.index((0, 0))
    summary = {"group_n": ref_n, "cols": ["t_acc_mean", "t_wga", "kl_mean", "kl_worst", "shift", "rho_s_t",
                                          "hit_best", "gate_frac"]}
    fixed = {}
    for _, full, tkl, tpred, y, gb, _dev in batches:
        for cond in conds:
            for k in KS:
                kl, pr = tkl[(cond, k)], tpred[(cond, k)]
                n_img = kl.shape[0]
                ar = torch.arange(n_img)
                ci = torch.full((n_img,), centre_idx, dtype=torch.long)
                bi_ = kl.argmin(1)
                _acc(fixed, f"{cond}/center@{k}", kl[ar, ci], pr[ar, ci] == y, gb, ci)
                _acc(fixed, f"{cond}/shift_best@{k}", kl[ar, bi_], pr[ar, bi_] == y, gb, bi_)
    summary["fixed"] = {key: _row(v) for key, v in _finish(fixed, ref_n).items()}

    for sname, rid in pick:
        student = api.student(rid, None)
        for p in student.parameters():
            p.requires_grad_(False)
        student.eval()
        acc = {}
        for ins, full, tkl, tpred, y, gb, dev in batches:
            rows, cols = rows_c.to(dev), cols_c.to(dev)
            yd, gbd = y.to(dev), gb.to(dev)
            for cond in conds:
                xi = ins[cond].to(dev)
                fl = full[cond].to(dev)
                s_full, maps = _maps(api, student, xi)
                B = xi.shape[0]
                cc = torch.full((B,), c, device=dev)
                for k in KS:
                    skl = []
                    for dy, dx in SHIFTS:
                        keep = _disc(cc + dy, cc + dx, k, rows, cols)
                        with torch.no_grad():
                            lp = api.teacher_forward_keepmask(student, xi, keep).float().log_softmax(-1)
                        skl.append(_kl(s_full, lp))
                    skl = torch.stack(skl, 1)
                    si = skl.argmin(1)
                    kl_t, pr_t = tkl[(cond, k)].to(dev), tpred[(cond, k)].to(dev)
                    ar = torch.arange(B, device=dev)
                    rho = _spearman_rows(skl, kl_t)
                    hit = (si == kl_t.argmin(1)).float()
                    _acc(acc, f"{cond}/suff_disc@{k}", kl_t[ar, si], pr_t[ar, si] == yd, gbd, si,
                         rho=rho, hit=hit)

                    # gate over heterogeneous candidates by student sufficiency
                    sy = torch.tensor([SHIFTS[i][0] for i in si.tolist()], device=dev).float()
                    sx = torch.tensor([SHIFTS[i][1] for i in si.tolist()], device=dev).float()
                    cands = {"center": _disc(cc, cc, k, rows, cols), "top_rollout": _topk(maps["rollout"], k),
                             "top_L9": _topk(maps["L9"], k), "suff_disc": _disc(cc + sy, cc + sx, k, rows, cols)}
                    ckl, clp = [], []
                    for name, keep in cands.items():
                        with torch.no_grad():
                            s_lp = api.teacher_forward_keepmask(student, xi, keep).float().log_softmax(-1)
                            t_lp = api.teacher_forward_keepmask(T, xi, keep).float().log_softmax(-1)
                        ckl.append(_kl(s_full, s_lp))
                        clp.append(t_lp)
                        if name == "top_rollout":
                            _acc(acc, f"{cond}/top_rollout@{k}", _kl(fl, t_lp), t_lp.argmax(-1) == yd, gbd, None)
                    gi_ = torch.stack(ckl, 1).argmin(1)
                    t_sel = torch.stack(clp, 1)[ar, gi_]
                    _acc(acc, f"{cond}/suff_gate@{k}", _kl(fl, t_sel), t_sel.argmax(-1) == yd, gbd, None,
                         gate=gi_)
        out = _finish(acc, ref_n)
        api.write_json(f"suff_{sname}", out)
        summary[sname] = {key: _row(v) for key, v in out.items()}
        api.write_json("summary", summary)
    api.write_json("summary", summary)


def _acc(store, key, kl, ok, gb, si, rho=None, hit=None, gate=None):
    row = store.setdefault(key, {"kl": [0.0] * 4, "t_acc": [0.0] * 4, "shift": 0.0, "n": 0, "rho": 0.0,
                                 "hit": 0.0, "gate": [0.0] * 4})
    kl, ok, gb = kl.float().cpu(), ok.float().cpu(), gb.cpu()
    for gi in range(4):
        m = gb == gi
        row["kl"][gi] += kl[m].sum().item()
        row["t_acc"][gi] += ok[m].sum().item()
    if si is not None:
        sh = torch.tensor([float((SHIFTS[i][0] ** 2 + SHIFTS[i][1] ** 2) ** 0.5) for i in si.tolist()])
        row["shift"] += sh.sum().item()
    if rho is not None:
        row["rho"] += rho.float().sum().item()
        row["hit"] += hit.float().sum().item()
    if gate is not None:
        for j in range(4):
            row["gate"][j] += (gate == j).float().sum().item()
    row["n"] += kl.shape[0]


def _finish(store, ref_n):
    out = {}
    for key, row in store.items():
        n_img = row["n"]
        t = [round(100 * v / max(b, 1), 2) for v, b in zip(row["t_acc"], ref_n)]
        kl = [round(v / max(b, 1), 4) for v, b in zip(row["kl"], ref_n)]
        out[key] = {"t_acc": t, "t_wga": min(t), "t_acc_mean": round(100 * sum(row["t_acc"]) / n_img, 2),
                    "kl": kl, "kl_mean": round(sum(row["kl"]) / n_img, 4), "kl_worst": max(kl),
                    "shift": round(row["shift"] / n_img, 3), "rho_s_t": round(row["rho"] / n_img, 3),
                    "hit_best": round(row["hit"] / n_img, 3),
                    "gate_frac": [round(v / n_img, 3) for v in row["gate"]]}
    return out


def _row(v):
    return [v["t_acc_mean"], v["t_wga"], v["kl_mean"], v["kl_worst"], v["shift"], v["rho_s_t"], v["hit_best"],
            v["gate_frac"]]
