"""H-W20 / teachability of suffdisc at K=12: does the student's sufficiency ranking of the 25 candidate discs agree
with the teacher's own faithfulness ranking of the same discs, and does that agreement grow with suffdisc training
(selector-student co-adaptation) rather than being a fixed property of any student?

For every val image and each of the 25 discs (centre disc shifted by <= 2 patches, as in selectors/suffdisc.py):
  s_kl[d]  KL(student full || student on disc d only)      (what suffdisc minimises; label-free)
  t_kl[d]  KL(teacher full || teacher on disc d only)       (teacher faithfulness of that input; student-free)
Per student we report, per Waterbirds group:
  rho       mean per-image Spearman correlation of s_kl and t_kl over the 25 discs
  top1      share of images where the student's argmin disc is the teacher's argmin disc (chance 0.04)
  t_kl_pick teacher KL at the student's pick (eval mode and drop-path train mode), vs the centre disc, the mean over
            discs and the teacher-best disc
  t_acc_pick masked-teacher accuracy at the student's pick (eval mode) and at the centre disc
Students: suffdisc@12 screen seeds 0-3 (e25, e50, final), center@12 screen seeds 0-3 final, jitdisc@12 seeds 0-1 final.
Pre-registered reading (PROPOSALS round 37): H-W20 is supported if, at final, suffdisc students' rho on the worst
group exceeds the same-seed center student's by >= 0.05 in at least 3 of 4 seeds, and the suffdisc s0-s3 rho rises
from e25 to final on average. Teacher-side numbers are context only (the faithfulness proxy failed out of sample,
H-W13).
"""
import torch

META = {
    "question": "does the student-sufficiency ranking of candidate discs track the teacher's faithfulness ranking, "
                "per group, and does that alignment grow with suffdisc training (co-adaptation)?",
    "method": "keep-only sufficiency faithfulness (ERASER sufficiency, RISE insertion) of disc-shaped token subsets, "
              "computed on the student and on the teacher, compared by per-image rank correlation",
    "paper": "DeYoung et al. ACL 2020 (ERASER); Petsiuk, Das & Saenko BMVC 2018 (RISE); Hooker et al. NeurIPS 2019 "
             "(ROAR)",
}

K = 12
SIDE = 14
R = 2
SHIFTS = [(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)]
CENTRE = SHIFTS.index((0, 0))


def _kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def _ranks(v):
    return v.argsort(1).argsort(1).float()


def _spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean(1, keepdim=True)
    rb = rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp_min(1e-8)


def _disc_kls(api, model, x, full_log, discs):
    out = []
    for keep in discs:
        with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            lp = api.teacher_forward_keepmask(model, x, keep.expand(x.shape[0], -1)).float().log_softmax(-1)
        out.append((_kl(full_log, lp), lp.argmax(-1)))
    return torch.stack([o[0] for o in out], 1), torch.stack([o[1] for o in out], 1)


def run(api):
    runs = api.runs()

    def find(method, seed):
        for rid, r in sorted(runs.items()):
            if (r["tier"] == "screen" and r["seed"] == seed and r["method"] == method and r["keep"] is not None
                    and int(r["keep"]) == K and r.get("dataset", "waterbirds") == "waterbirds"):
                return rid, r["snapshots"]
        return None, []

    plan = []
    for sd in range(4):
        rid, snaps = find("suffdisc", sd)
        if rid:
            for e in snaps:
                plan.append((f"suffdisc_s{sd}_e{e}", rid, e))
            plan.append((f"suffdisc_s{sd}_final", rid, None))
    for m, sds in (("center", range(4)), ("jitdisc", range(2))):
        for sd in sds:
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
    tie = rr.float() * 1e-6
    discs = []
    for dy, dx in SHIFTS:
        d = (rows - (c + dy)) ** 2 + (cols - (c + dx)) ** 2 + tie
        idx = (-d).topk(K).indices
        discs.append(torch.zeros(1, n, dtype=torch.bool, device=dev).scatter_(1, idx[None], True))

    # teacher side, student-free
    t_kl, t_pred, Y = [], [], []
    for x, y in api.val_batches(batch=50):
        with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            tf = T(x).float().log_softmax(-1)
        kl, pr = _disc_kls(api, T, x, tf, discs)
        t_kl.append(kl.cpu())
        t_pred.append(pr.cpu())
        Y.append(y.cpu())
    t_kl, t_pred, Y = torch.cat(t_kl), torch.cat(t_pred), torch.cat(Y)
    G = groups.cpu()
    t_best = t_kl.argmin(1)
    t_correct = (t_pred == Y[:, None]).float()
    img = torch.arange(len(Y))

    def gmean(v):
        return [round(v[G == g].float().mean().item(), 4) for g in range(4)]

    teacher_row = {
        "t_kl_centre_g": gmean(t_kl[:, CENTRE]), "t_kl_meandisc_g": gmean(t_kl.mean(1)),
        "t_kl_best_g": gmean(t_kl[img, t_best]), "t_acc_centre_g": [round(100 * x, 2) for x in gmean(t_correct[:, CENTRE])],
        "t_acc_best_kl_g": [round(100 * x, 2) for x in gmean(t_correct[img, t_best])],
        "t_best_centre_frac": round((t_best == CENTRE).float().mean().item(), 4)}
    api.write_json("teacher", teacher_row)

    out = {}
    for sname, rid, ep in plan:
        st = api.student(rid, ep)
        for p in st.parameters():
            p.requires_grad_(False)
        s_kl, s_kl_tr = [], []
        for x, _ in api.val_batches(batch=50):
            st.eval()
            with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16,
                                                 enabled=x.device.type == "cuda"):
                sf = st(x).float().log_softmax(-1)
            s_kl.append(_disc_kls(api, st, x, sf, discs)[0].cpu())
            st.train()
            with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16,
                                                 enabled=x.device.type == "cuda"):
                sf = st(x).float().log_softmax(-1)
            s_kl_tr.append(_disc_kls(api, st, x, sf, discs)[0].cpu())
            st.eval()
        s_kl, s_kl_tr = torch.cat(s_kl), torch.cat(s_kl_tr)
        pick, pick_tr = s_kl.argmin(1), s_kl_tr.argmin(1)
        rho = _spearman(s_kl, t_kl)
        acc_pick = t_correct[img, pick]
        row = {"run": rid, "epoch": ep,
               "rho_g": gmean(rho), "rho": round(rho.mean().item(), 4),
               "top1_g": gmean((pick == t_best).float()), "top1": round((pick == t_best).float().mean().item(), 4),
               "t_kl_pick_g": gmean(t_kl[img, pick]), "t_kl_pick_train_g": gmean(t_kl[img, pick_tr]),
               "t_acc_pick_g": [round(100 * v, 2) for v in gmean(acc_pick)],
               "t_wga_pick": round(100 * min(gmean(acc_pick)), 2),
               "centre_frac": round((pick == CENTRE).float().mean().item(), 4)}
        out[sname] = row
        api.write_json("rows", out)
        del st

    worst = int(torch.tensor(teacher_row["t_kl_centre_g"]).argmax().item())
    api.write_json("summary", {
        "note": "eval-mode students on clean val; rho = per-image Spearman(student disc KL, teacher disc KL) over 25 "
                "discs; top1 chance = 0.04; worst = group with the largest teacher KL on the centre disc",
        "worst_group": worst,
        "teacher": teacher_row,
        "table_cols": ["rho_all", "rho_worst", "top1_worst", "t_kl_pick_worst", "t_kl_pick_train_worst",
                       "t_wga_pick"],
        "table": {s: [r["rho"], r["rho_g"][worst], r["top1_g"][worst], r["t_kl_pick_g"][worst],
                      r["t_kl_pick_train_g"][worst], r["t_wga_pick"]] for s, r in out.items()},
    })
