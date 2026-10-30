"""H-W8: why did the clean-image proxy over-predict midattn (block-9 CLS attention)?

analysis_group_fidelity_1030_143842_1ea0 / _144455_654a rank selectors by the masked teacher's worst-group
accuracy on *clean* val images with the student in *eval* mode. That ranking matched student WGA for 5 methods
at K=29 and K=49 (seed 0), but midattn@49 (screen_midattn_k49_s0_1030_153152_3b79) reached 89.91 / WGA 57.89,
below rollout@49 64.66 and center@49 60.15, while the proxy put block-9 (72.18) level with rollout (72.93).

During training the selector sees something else: the student is in train mode (drop path 0.1), and every batch
is mixup (lam ~ Beta(0.8,0.8)) or cutmix (box pasted from another image). Here the same fidelity measures are
taken under those conditions:
  clean_eval   clean val image, student.eval()                 (the old proxy)
  clean_train  clean val image, student.train() (drop path on)
  cutmix       a box (aligned to the 16-px patch grid) from the next image in the batch pasted in, train mode
  mixup        0.6 * image + 0.4 * next image, train mode
Per condition, selector (block-9 / block-11 CLS attention, rollout, center) and K (29, 49), split by the val
group of the base image:
  kl_g     KL(teacher_full || teacher_masked) on the same (mixed) input: the KD target error the student gets
  agree_g  top-1 agreement masked vs full teacher
  t_acc_g  masked-teacher accuracy on the base label (clean conditions only)
  box_share (cutmix) kept tokens inside the pasted box / K, vs the box area share: does the selector drop one
           of the two images the full teacher sees?
  stab     |S_train ∩ S_eval| / K on the clean image: how much drop path moves the selection
Students: screen seed-0 midattn@49 and rollout@49 (e25, e50, final), maskedkd@49 final.
"""
import torch

META = {
    "question": "does the clean-image, eval-mode masked-teacher proxy mis-rank block-9 CLS attention because the "
                "selector runs on mixup/cutmix inputs and a drop-path student during training?",
    "method": "group-sliced faithfulness of the masked teacher (KL / agreement to the full teacher) under the "
              "training input distribution; selection stability under stochastic depth",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Yun et al. ICCV 2019 (CutMix); Zhang et al. ICLR 2018 (mixup); "
             "Huang et al. ECCV 2016 (stochastic depth); Sagawa et al. ICLR 2020 (Waterbirds groups)",
}

KS = (29, 49)
SIDE = 14
LAM_CUT = (0.3, 0.5, 0.7)   # kept share of the base image, cycled over batches (deterministic, no RNG seed)
LAM_MIX = 0.6


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


def _scores(api, student, x):
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
        if r["tier"] != "screen" or r["seed"] != 0 or r["keep"] != 49:
            continue
        if r["method"] in ("midattn", "rollout"):
            tag = "mid49" if r["method"] == "midattn" else "roll49"
            pick += [(f"{tag}_e{e}", rid, e) for e in r["snapshots"]] + [(f"{tag}_final", rid, None)]
        if r["method"] == "maskedkd":
            pick.append(("mkd49_final", rid, None))
    api.write_json("students", {n: [rid, e] for n, rid, e in pick})
    groups = api.val_groups()
    T = api.teacher.eval()
    n = api.num_patches
    rr = torch.arange(n)
    c = (SIDE - 1) / 2
    center = -(((rr // SIDE - c) ** 2 + (rr % SIDE - c) ** 2).float().sqrt()) - rr.float() * 1e-6
    conds = ("clean_eval", "clean_train", "cutmix", "mixup")

    # inputs and full-teacher targets do not depend on the student: build once
    batches = []
    off = 0
    for bi, (x, y) in enumerate(api.val_batches(batch=50)):
        B = x.shape[0]
        gb = groups[off:off + B].to(x.device)
        off += B
        x2 = x.roll(1, 0)
        r0, c0, s, inside = _box(bi)
        xc = x.clone()
        xc[:, :, 16 * r0:16 * (r0 + s), 16 * c0:16 * (c0 + s)] = x2[:, :, 16 * r0:16 * (r0 + s), 16 * c0:16 * (c0 + s)]
        xm = LAM_MIX * x + (1 - LAM_MIX) * x2
        ins = {"clean_eval": x, "clean_train": x, "cutmix": xc, "mixup": xm}
        with torch.no_grad():
            full = {k: T(v).float().log_softmax(-1) for k, v in ins.items() if k != "clean_train"}
        full["clean_train"] = full["clean_eval"]
        # kept on the CPU: the GPU is shared with training jobs
        batches.append(({k: v.cpu() for k, v in ins.items() if k != "clean_train"},
                        {k: v.cpu() for k, v in full.items()}, y.cpu(), gb.cpu(), inside, x.device))

    ref_n = [0] * 4
    for _, _, _, gb, _, _ in batches:
        for gi in range(4):
            ref_n[gi] += int((gb == gi).sum().item())

    summary = {}
    for sname, rid, ep in pick:
        student = api.student(rid, ep)
        for p in student.parameters():
            p.requires_grad_(False)
        acc = {}
        stab = {f"{s}@{k}": 0.0 for s in ("L9", "L11", "rollout") for k in KS}
        n_img = 0
        for ins, full, y, gb, inside, dev in batches:
            ins = {k: v.to(dev) for k, v in ins.items()}
            ins["clean_train"] = ins["clean_eval"]
            full = {k: v.to(dev) for k, v in full.items()}
            y, gb, inside = y.to(dev), gb.to(dev), inside.to(dev)
            B = y.shape[0]
            n_img += B
            sc_eval = None
            for cond in conds:
                if cond == "clean_eval":
                    student.eval()
                else:
                    student.train()
                sc = _scores(api, student, ins[cond])
                sc["center"] = center.to(dev).expand(B, n)
                if cond == "clean_eval":
                    sc_eval = sc
                for name, s in sc.items():
                    for k in KS:
                        keep = _topk(s, k)
                        if cond == "clean_train" and name != "center":
                            stab[f"{name}@{k}"] += ((keep & _topk(sc_eval[name], k)).sum(1).float() / k).sum().item()
                        with torch.no_grad():
                            t_m = api.teacher_forward_keepmask(T, ins[cond], keep).float().log_softmax(-1)
                        row = acc.setdefault(f"{cond}/{name}@{k}", {"kl": [0.0] * 4, "agree": [0.0] * 4,
                                                                    "t_acc": [0.0] * 4, "box_share": 0.0})
                        kl = _kl(full[cond], t_m)
                        ag = (t_m.argmax(-1) == full[cond].argmax(-1)).float()
                        ok = (t_m.argmax(-1) == y).float()
                        for gi in range(4):
                            m = gb == gi
                            row["kl"][gi] += kl[m].sum().item()
                            row["agree"][gi] += ag[m].sum().item()
                            row["t_acc"][gi] += ok[m].sum().item()
                        if cond == "cutmix":
                            row["box_share"] += ((keep & inside).sum(1).float() / k).sum().item()
        student.eval()
        out = {"stab": {key: round(v / n_img, 4) for key, v in stab.items()}}
        for key, row in acc.items():
            o = {"kl": [round(v / max(b, 1), 4) for v, b in zip(row["kl"], ref_n)],
                 "agree": [round(100 * v / max(b, 1), 2) for v, b in zip(row["agree"], ref_n)]}
            o["kl_mean"] = round(sum(row["kl"]) / n_img, 4)
            o["kl_worst"] = round(max(o["kl"]), 4)
            o["agree_worst"] = round(min(o["agree"]), 2)
            if key.startswith("clean"):
                o["t_acc"] = [round(100 * v / max(b, 1), 2) for v, b in zip(row["t_acc"], ref_n)]
                o["t_wga"] = round(min(o["t_acc"]), 2)
            if key.startswith("cutmix"):
                o["box_share"] = round(row["box_share"] / n_img, 4)
            out[key] = o
        api.write_json(f"gap_{sname}", out)
        summary[sname] = {key: [v.get("t_wga"), v["kl_worst"], v["agree_worst"], v.get("box_share")]
                          for key, v in out.items() if key != "stab"}
    area = []
    for bi in range(len(batches)):
        area.append(_box(bi)[3].float().mean().item())
    api.write_json("summary", {"cols": ["t_wga", "kl_worst", "agree_worst", "box_share"],
                               "box_area_share_mean": round(sum(area) / len(area), 4),
                               "group_n": ref_n, "rows": summary})
