"""Pre-promotion decomposition of suffdisc (H-W11) at K=12, as required before screen -> final.

screen 2-seed means: suffdisc@12 89.62 / WGA 61.28, center@12 88.37 / 54.14, jitdisc@12 (same 25 discs, shift drawn
uniformly at random, parent suffdisc) 88.53 / 53.01. The replacement control is therefore done at student level.
What is still missing is the overlap and spatial part of the decomposition:
  overlap   share of suffdisc's K patches that MaskedKD (block-11 CLS attention), rollout, and the centre disc also
            keep, per training snapshot and per Waterbirds group
  spatial   which of the 25 shifts the student picks (histogram, share of the centre shift, mean |shift|), mean
            patch distance from the image centre (cdist), per group
  noise     the selector runs the student in train mode (drop path); agreement of the train-mode pick with the
            eval-mode pick shows how much of the choice is drop-path noise rather than signal
  fidelity  masked-teacher group accuracy and worst-group KL(full || masked) for suffdisc, center, a fixed
            pseudo-random shift (jitdisc stand-in) and MaskedKD, with each student
Students: suffdisc@12 seed 0 (e25, e50, final), suffdisc@12 seed 1 final, jitdisc@12 seed 0 and 1 final,
center@12 seed 0 and 1 final. Teacher-side numbers are context only (the faithfulness proxy failed out of sample).
"""
import torch

META = {
    "question": "where does the student-sufficiency disc move the teacher's window, how often is that the centre, "
                "how much of it overlaps MaskedKD / rollout tokens, and how much of the pick is drop-path noise?",
    "method": "keep-only sufficiency faithfulness (ERASER sufficiency, RISE insertion) of disc-shaped token "
              "subsets on the student; selection overlap with last-block CLS attention and attention rollout",
    "paper": "DeYoung et al. ACL 2020 (ERASER); Petsiuk, Das & Saenko BMVC 2018 (RISE); Abnar & Zuidema ACL 2020; "
             "Touvron et al. 2021 (MaskedKD baseline signal: CLS attention)",
}

K = 12
SIDE = 14
R = 2
SHIFTS = [(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)]
CENTRE = SHIFTS.index((0, 0))


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def _attn_maps(api, student, x):
    depth = len(student.blocks)
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
    A = [a.float().mean(1) for a in out.attn]
    B, N, _ = A[0].shape
    eye = torch.eye(N, device=x.device).expand(B, N, N)
    Rm = eye.clone()
    for a in A:
        a = 0.5 * a + 0.5 * eye
        Rm = (a / a.sum(-1, keepdim=True)) @ Rm
    return out.logits.float().log_softmax(-1), A[-1][:, 0, 1:], Rm[:, 0, 1:]


def _pick(api, student, x, s_full, discs):
    kls = []
    for keep in discs:
        with torch.no_grad():
            lp = api.teacher_forward_keepmask(student, x, keep.expand(x.shape[0], -1)).float().log_softmax(-1)
        kls.append(_kl(s_full, lp))
    return torch.stack(kls, 1).argmin(1)


def run(api):
    runs = api.runs()

    def find(method, seed):
        for rid, r in sorted(runs.items()):
            if (r["tier"] == "screen" and r["seed"] == seed and r["method"] == method and r["keep"] is not None
                    and int(r["keep"]) == K and r.get("dataset", "waterbirds") == "waterbirds"):
                return rid, r["snapshots"]
        return None, []

    plan = []
    s0, snaps = find("suffdisc", 0)
    for e in snaps:
        plan.append((f"suffdisc_s0_e{e}", s0, e))
    plan.append(("suffdisc_s0_final", s0, None))
    for m, sd in (("suffdisc", 1), ("jitdisc", 0), ("jitdisc", 1), ("center", 0), ("center", 1)):
        rid, _ = find(m, sd)
        if rid:
            plan.append((f"{m}_s{sd}_final", rid, None))
    api.write_json("students", plan)

    groups = api.val_groups()
    T = api.teacher.eval()
    n = api.num_patches
    dev = api.device
    rr = torch.arange(n, device=dev)
    rows, cols = (rr // SIDE).float(), (rr % SIDE).float()
    c = (SIDE - 1) / 2
    dist = ((rows - c) ** 2 + (cols - c) ** 2).sqrt()
    tie = rr.float() * 1e-6
    discs = []
    for dy, dx in SHIFTS:
        d = (rows - (c + dy)) ** 2 + (cols - (c + dx)) ** 2 + tie
        discs.append(_topk(-d[None], K))
    D = torch.cat(discs, 0)  # 25 x n
    shift_norm = torch.tensor([(dy * dy + dx * dx) ** 0.5 for dy, dx in SHIFTS], device=dev)

    # teacher side, student-free
    tfull = []
    for x, _ in api.val_batches(batch=50):
        with torch.no_grad():
            tfull.append(T(x).float().log_softmax(-1))

    out = {}
    names = ("suffdisc", "center", "jitter", "maskedkd")
    for sname, rid, ep in plan:
        st = api.student(rid, ep)
        for p in st.parameters():
            p.requires_grad_(False)
        acc = {m: [] for m in names}
        kl = {m: [] for m in names}
        cd = {m: [] for m in names}
        ov = {"suff_mkd": [], "suff_roll": [], "suff_center": [], "center_mkd": []}
        picks, picks_tr, G_all = [], [], []
        off = 0
        for bi, (x, y) in enumerate(api.val_batches(batch=50)):
            B = x.shape[0]
            gb = groups[off:off + B]
            img = torch.arange(off, off + B, device=dev)
            off += B
            st.eval()
            s_full, s_mkd, s_roll = _attn_maps(api, st, x)
            pi = _pick(api, st, x, s_full, discs)
            st.train()
            with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16,
                                                 enabled=x.device.type == "cuda"):
                s_full_tr = st(x).float().log_softmax(-1)
            pt = _pick(api, st, x, s_full_tr, discs)
            st.eval()
            ji = ((img * 104729 + 7919) % 25).long()
            sel = {"suffdisc": D[pi], "center": D[CENTRE].expand(B, -1), "jitter": D[ji],
                   "maskedkd": _topk(s_mkd, K)}
            ov["suff_mkd"].append((sel["suffdisc"] & sel["maskedkd"]).sum(1).float().cpu() / K)
            ov["suff_roll"].append((sel["suffdisc"] & _topk(s_roll, K)).sum(1).float().cpu() / K)
            ov["suff_center"].append((sel["suffdisc"] & sel["center"]).sum(1).float().cpu() / K)
            ov["center_mkd"].append((sel["center"] & sel["maskedkd"]).sum(1).float().cpu() / K)
            full = tfull[bi]
            with torch.no_grad():
                for m in names:
                    keep = sel[m]
                    lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    acc[m].append((lm.argmax(-1) == y).cpu())
                    kl[m].append(_kl(full, lm).cpu())
                    cd[m].append(((keep.float() * dist).sum(1) / K).cpu())
            picks.append(pi.cpu())
            picks_tr.append(pt.cpu())
            G_all.append(gb.cpu())
        Gc = torch.cat(G_all)
        P, Pt = torch.cat(picks), torch.cat(picks_tr)
        sn = shift_norm.cpu()[P]
        row = {"run": rid, "epoch": ep, "K": K,
               "shift_hist": torch.bincount(P, minlength=25).tolist(),
               "centre_frac": round((P == CENTRE).float().mean().item(), 4),
               "centre_frac_g": [round((P[Gc == g] == CENTRE).float().mean().item(), 4) for g in range(4)],
               "mean_shift": round(sn.mean().item(), 3),
               "mean_shift_g": [round(sn[Gc == g].mean().item(), 3) for g in range(4)],
               "mean_dy_g": [round(torch.tensor([SHIFTS[i][0] for i in P[Gc == g].tolist()]).float().mean()
                                   .item(), 3) for g in range(4)],
               "mean_dx_g": [round(torch.tensor([SHIFTS[i][1] for i in P[Gc == g].tolist()]).float().mean()
                                   .item(), 3) for g in range(4)],
               "train_eval_agree": round((P == Pt).float().mean().item(), 4),
               "train_centre_frac": round((Pt == CENTRE).float().mean().item(), 4)}
        for m in names:
            a, kk, dd = torch.cat(acc[m]).float(), torch.cat(kl[m]), torch.cat(cd[m])
            ga = [round(100 * a[Gc == g].mean().item(), 2) for g in range(4)]
            gk = [round(kk[Gc == g].mean().item(), 4) for g in range(4)]
            row[m] = {"t_acc": round(100 * a.mean().item(), 2), "t_wga": min(ga), "t_acc_g": ga,
                      "kl_worst": max(gk), "kl_g": gk, "cdist": round(dd.mean().item(), 3),
                      "cdist_g": [round(dd[Gc == g].mean().item(), 3) for g in range(4)]}
        for o, v in ov.items():
            v = torch.cat(v)
            row["overlap_" + o] = round(v.mean().item(), 4)
            row["overlap_" + o + "_g"] = [round(v[Gc == g].mean().item(), 4) for g in range(4)]
        out[sname] = row
        api.write_json("rows", out)
        del st
    api.write_json("summary", {
        "note": "eval-mode student on clean val; jitter = fixed pseudo-random shift per image (jitdisc stand-in); "
                "overlap = share of K; cdist = mean kept-patch distance from the centre (patch units); "
                "train_eval_agree = share of images where the drop-path (train-mode) pick equals the eval pick",
        "table": {s: {m: [r[m]["t_acc"], r[m]["t_wga"], r[m]["kl_worst"], r[m]["cdist"]] for m in names}
                  for s, r in out.items()},
        "table_cols": ["t_acc", "t_wga", "kl_worst", "cdist"],
        "pick": {s: [r["centre_frac"], r["mean_shift"], r["train_eval_agree"], r["train_centre_frac"]]
                 for s, r in out.items()},
        "pick_cols": ["centre_frac", "mean_shift", "train_eval_agree", "train_centre_frac"],
        "overlap": {s: [r["overlap_suff_mkd"], r["overlap_suff_roll"], r["overlap_suff_center"],
                        r["overlap_center_mkd"]] for s, r in out.items()},
        "overlap_cols": ["suff_mkd", "suff_roll", "suff_center", "center_mkd"],
    })
