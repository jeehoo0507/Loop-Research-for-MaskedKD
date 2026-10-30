"""Why does gradroll@12 WGA swing across final seeds (H-W19)?

final_gradroll_k12_s1_1030_201811_faf0 87.99 / 63.16, s2 final_gradroll_k12_s2_1030_212126_da50 88.41 / 62.41,
s3 final_gradroll_k12_s3_1030_220958_176b 87.57 / 51.88 (worst group 2 in all). Two candidate explanations:
  (a) selection collapse: the s3 student's gradient-weighted rollout picks worse tokens for the teacher
      (lower masked-teacher WGA, drifts towards MaskedKD / away from the object) during training
  (b) student-side: selection is as good as in s1/s2, the s3 student itself just ends weaker on group 2
Per student snapshot (e25 / e50 / final) of each gradroll final seed, plus rollout@12 and center@12 finals of the
same seeds, this measures: the student's own group accuracy (clean val), masked-teacher group accuracy and
worst-group KL for its gradroll / rollout / maskedkd / center selection, overlap with MaskedKD and center, centre
distance, and the cross-seed agreement of gradroll selections on the same images (final students).
Teacher-side numbers are context only: the faithfulness proxy failed out of sample (H-W13/H-W15).
"""
import torch

META = {
    "question": "is the seed-3 WGA drop of gradroll@12 a selection collapse (worse teacher inputs) or a "
                "student-side difference with equally good selections?",
    "method": "gradient-weighted attention rollout vs attention rollout vs last-block CLS attention; keep-only "
              "masked-teacher evaluation per group, selection overlap and cross-seed agreement",
    "paper": "Chefer, Gur & Wolf CVPR 2021; Abnar & Zuidema ACL 2020; Hooker et al. NeurIPS 2019 (keep-only)",
}

SIDE = 14
K = 12


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _scores(api, student, x):
    """student logits, MaskedKD (block-11 CLS attn), plain rollout, gradient-weighted rollout."""
    depth = len(student.blocks)
    with torch.enable_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
        target = out.logits.float().max(1).values.sum()
        grads = torch.autograd.grad(target, list(out.attn))
    A = [a.detach().float() for a in out.attn]
    n = A[0].shape[-1]
    eye = torch.eye(n, device=x.device)
    R = None
    G = eye.unsqueeze(0).expand(x.shape[0], n, n)
    for a, g in zip(A, grads):
        am = a.mean(1)
        am = 0.5 * am + 0.5 * eye
        am = am / am.sum(-1, keepdim=True)
        R = am if R is None else am @ R
        G = G + (g.detach().float() * a).clamp(min=0).mean(1) @ G
    return out.logits.detach().float(), A[-1].mean(1)[:, 0, 1:], R[:, 0, 1:], G[:, 0, 1:]


def run(api):
    runs = api.runs()

    def find(method, seed):
        for rid, r in sorted(runs.items()):
            if (r["tier"] == "final" and r["seed"] == seed and r["method"] == method and r["keep"] is not None
                    and int(r["keep"]) == K and r.get("dataset", "waterbirds") == "waterbirds"):
                return rid, r["snapshots"]
        return None, []

    plan = []
    for s in (1, 2, 3):
        rid, snaps = find("gradroll", s)
        if rid is None:
            continue
        for e in snaps:
            plan.append((f"gradroll_s{s}_e{e}", rid, e, s))
        plan.append((f"gradroll_s{s}_final", rid, None, s))
        for m in ("rollout", "center"):
            r2, _ = find(m, s)
            if r2:
                plan.append((f"{m}_s{s}_final", r2, None, s))
    api.write_json("students", plan)

    groups = api.val_groups()
    T = api.teacher.eval()
    n = api.num_patches
    dev = api.device
    rr = torch.arange(n, device=dev)
    rows_, cols_ = (rr // SIDE).float(), (rr % SIDE).float()
    c = (SIDE - 1) / 2
    dist = ((rows_ - c) ** 2 + (cols_ - c) ** 2).sqrt()
    center_s = -(dist + rr.float() * 1e-6)
    names = ("gradroll", "rollout", "maskedkd", "center")

    out = {}
    final_sel = {}
    for sname, rid, ep, seed in plan:
        st = api.student(rid, ep)
        acc = {m: [] for m in names}
        kl = {m: [] for m in names}
        cd = {m: 0.0 for m in names}
        s_acc = []
        ov = {"grad_mkd": [], "grad_center": [], "grad_roll": []}
        G_all, keep_g = [], []
        tot = 0
        off = 0
        for x, y in api.val_batches(batch=25):
            B = x.shape[0]
            G_all.append(groups[off:off + B].cpu())
            off += B
            logits, s_mkd, s_roll, s_grad = _scores(api, st, x)
            s_acc.append((logits.argmax(-1) == y).cpu())
            sel = {"gradroll": _topk(s_grad, K), "rollout": _topk(s_roll, K), "maskedkd": _topk(s_mkd, K),
                   "center": _topk(center_s.expand(B, n), K)}
            ov["grad_mkd"].append((sel["gradroll"] & sel["maskedkd"]).sum(1).float().cpu() / K)
            ov["grad_center"].append((sel["gradroll"] & sel["center"]).sum(1).float().cpu() / K)
            ov["grad_roll"].append((sel["gradroll"] & sel["rollout"]).sum(1).float().cpu() / K)
            keep_g.append(sel["gradroll"].cpu())
            with torch.no_grad():
                full = T(x).float().log_softmax(-1)
                for m in names:
                    keep = sel[m]
                    lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
                    acc[m].append((lm.argmax(-1) == y).cpu())
                    kl[m].append((full.exp() * (full - lm)).sum(-1).cpu())
                    cd[m] += (keep.float() * dist).sum().item() / K
            tot += B
        Gc = torch.cat(G_all)
        sa = torch.cat(s_acc).float()
        sg = [round(100 * sa[Gc == gi].mean().item(), 2) for gi in range(4)]
        row = {"run": rid, "epoch": ep, "seed": seed,
               "student": {"acc": round(100 * sa.mean().item(), 2), "wga": min(sg), "acc_g": sg}}
        for m in names:
            a, kk = torch.cat(acc[m]).float(), torch.cat(kl[m])
            ga = [round(100 * a[Gc == gi].mean().item(), 2) for gi in range(4)]
            gk = [round(kk[Gc == gi].mean().item(), 4) for gi in range(4)]
            row[m] = {"t_acc": round(100 * a.mean().item(), 2), "t_wga": min(ga), "t_acc_g": ga,
                      "kl_worst": max(gk), "kl_g": gk, "cdist": round(cd[m] / tot, 3)}
        for o, v in ov.items():
            v = torch.cat(v)
            row["overlap_" + o] = round(v.mean().item(), 4)
            row["overlap_" + o + "_g"] = [round(v[Gc == gi].mean().item(), 4) for gi in range(4)]
        if sname.startswith("gradroll") and ep is None:
            final_sel[seed] = torch.cat(keep_g)
        out[sname] = row
        api.write_json("rows", out)
        del st

    cross = {}
    seeds = sorted(final_sel)
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = final_sel[seeds[i]], final_sel[seeds[j]]
            v = (a & b).sum(1).float() / K
            cross[f"s{seeds[i]}_s{seeds[j]}"] = {"mean": round(v.mean().item(), 4),
                                                 "g": [round(v[Gc == gi].mean().item(), 4) for gi in range(4)]}
    api.write_json("summary", {
        "note": "student = the student's own clean-val accuracy; t_* = masked teacher with that student's selection; "
                "overlap = share of K; cdist = mean patch distance from the centre; cross_seed = share of K that two "
                "final gradroll students pick in common on the same image",
        "table": {s: [r["student"]["wga"], r["gradroll"]["t_wga"], r["gradroll"]["kl_worst"],
                      r["gradroll"]["cdist"], r["overlap_grad_mkd"], r["overlap_grad_center"]]
                  for s, r in out.items()},
        "table_cols": ["student_wga", "grad_t_wga", "grad_kl_worst", "grad_cdist", "ov_mkd", "ov_center"],
        "cross_seed": cross,
    })
