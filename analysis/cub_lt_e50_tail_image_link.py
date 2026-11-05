"""H-A167 (round 431, analysis only, no training): H-A166 at the e50 snapshot instead of e25. H-A166 compared the
cub_lt tail image-level link C (H-A163) of online maskedkd between e25 and final: +7.65 [-0.51, +15.82], uncertain,
but e25 students reach only 12-19% tail accuracy, so e25 C was measured near the floor. This is a separately
registered, post-H-A166 check at the only other snapshot (50% of training); it does not revise the H-A166 verdict.
Same code as H-A166 except the snapshot epoch and the bootstrap seed.

Runs: cub_lt final K=29 seeds 1-3, arms lateroll, maskedkd_frozen, maskedkd (9 runs, same as H-A165), each at
snapshot e50 and final. For each epoch E and arm a: tw_a,E(i), sw_a,E(i) = seed mean of 1[masked teacher wrong],
1[student wrong], where masks and student predictions both come from the epoch-E student of that run (lateroll uses
its own epoch-E student too). dT = tw_a,E - tw_lateroll,E, dS likewise; groups in the tail sample: G+ = dT > 0,
G0 = dT == 0. C_a,E = mean dS[G+] - mean dS[G0] (%p), same code as H-A165.
P1: C_maskedkd,final - C_maskedkd,e50 on tail images, one shared image bootstrap (numpy default_rng, 2000).
Record: C at both epochs for both arms, (C_mkd - C_frozen) at e50 and its change to final, head/all contrasts,
seed-wise values, masked-teacher and student tail accuracies per run and epoch.
Limit: clean val views (training uses crop+flip, H-A77); frozen-arm mask = maskedkd rule on the frozen-arm
epoch-E student (not its training mask); correlation only; no selector.
"""
import math

import numpy
import torch

META = {
    "question": "on cub_lt K=29 tail images, is the image-level link between maskedkd-rule masked-teacher errors and "
                "maskedkd student errors (H-A163) larger at the final student than at the e50 snapshot?",
    "method": "per-image seed-averaged masked-teacher and student error differences vs lateroll at epoch 50 and final, "
              "tail group contrast with a shared image bootstrap for the epoch difference",
    "paper": "Hooker et al. NeurIPS 2019 (keep-only evaluation); Abnar & Zuidema ACL 2020 (attention rollout); "
             "Son et al. ECCV 2024 (MaskedKD); Arazo et al. IJCNN 2020 (confirmation bias)",
}

BATCH = 50
N_BOOT = 2000
SEED = 20261181
K = 29
START = 6
T_CRIT = 4.302653  # t(0.975, df 2)
EPOCHS = (50, None)  # None = final weights
RUNS = {
    "lateroll": ["cub_lt_final_lateroll_k29_s1_1104_084157_2b15", "cub_lt_final_lateroll_k29_s2_1104_084204_9f09",
                 "cub_lt_final_lateroll_k29_s3_1104_084209_a3f8"],
    "frozen": ["cub_lt_final_maskedkd_frozen_k29_s1_1104_074618_8245",
               "cub_lt_final_maskedkd_frozen_k29_s2_1104_074622_647c",
               "cub_lt_final_maskedkd_frozen_k29_s3_1104_074627_dd2d"],
    "maskedkd": ["cub_lt_final_maskedkd_k29_s1_1103_234609_76d1", "cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c",
                 "cub_lt_final_maskedkd_k29_s3_1103_234704_b28b"],
}


def _topk_mask(scores, k):
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(1, scores.topk(k, dim=1).indices, True)
    return keep


def _rollout(attn):
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _tcrit(d):
    d = numpy.asarray(d, dtype=float)
    h = T_CRIT * d.std(ddof=1) / math.sqrt(len(d))
    return [round(float(d.mean()), 4), round(float(d.mean() - h), 4), round(float(d.mean() + h), 4)]


def _contrast(dT, dS):
    gp, g0 = dT > 0, dT == 0
    if gp.sum() == 0 or g0.sum() == 0:
        return float("nan")
    return float((dS[gp].mean() - dS[g0].mean()) * 100)


def _ci(point, bs):
    bs = numpy.asarray(bs)
    return [round(point, 4), round(float(numpy.nanquantile(bs, 0.025)), 4),
            round(float(numpy.nanquantile(bs, 0.975)), 4)]


def _boot4(d, rng):
    """d: {(arm, tag): (dT, dS)} for arm in frozen/maskedkd, tag in e50/final. One shared image resample per draw."""
    keys = list(d)
    n = len(d[keys[0]][0])
    draws = {k: [] for k in keys}
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n)
        for k in keys:
            draws[k].append(_contrast(d[k][0][ix], d[k][1][ix]))
    pts = {k: _contrast(*d[k]) for k in keys}
    b = {k: numpy.asarray(v) for k, v in draws.items()}
    res = {f"C_{a}_{t}_pp_boot": _ci(pts[(a, t)], b[(a, t)]) for a, t in keys}
    for a in ("maskedkd", "frozen"):
        res[f"C_{a}_final_minus_e50_pp_boot"] = _ci(pts[(a, "final")] - pts[(a, "e50")],
                                                    b[(a, "final")] - b[(a, "e50")])
    for t in ("e50", "final"):
        res[f"C_maskedkd_minus_frozen_{t}_pp_boot"] = _ci(pts[("maskedkd", t)] - pts[("frozen", t)],
                                                          b[("maskedkd", t)] - b[("frozen", t)])
    gap = lambda s: s[("maskedkd", "final")] - s[("frozen", "final")] - s[("maskedkd", "e50")] + s[("frozen", "e50")]
    res["gap_change_final_minus_e50_pp_boot"] = _ci(gap(pts), gap(b))
    return res


def run(api):
    runs = api.runs()
    api.write_json("students", {rid: [runs[rid].get("dataset"), runs[rid]["method"], runs[rid]["keep"],
                                      runs[rid]["seed"], runs[rid]["val_acc_last"], runs[rid].get("val_wga_last"),
                                      runs[rid]["snapshots"]]
                                for arm in RUNS for rid in RUNS[arm]})
    groups = api.val_groups().cpu().numpy()
    Tm = api.teacher.eval()
    batches = [(x, y) for x, y in api.val_batches(batch=BATCH)]
    y = torch.cat([b[1] for b in batches]).cpu().numpy().astype(int)
    n = len(y)
    tail, head = groups[:n] == 1, groups[:n] == 0

    tw, sw, per_run = {}, {}, {}
    for arm, ids in RUNS.items():
        for rid in ids:
            for ep in EPOCHS:
                tag = "final" if ep is None else f"e{ep}"
                S = api.student(rid, epoch=ep)
                depth = len(S.blocks)
                gp, sp = [], []
                for x, _ in batches:
                    with torch.no_grad():
                        sp.append(S(x).float().argmax(1).cpu())
                        if arm == "lateroll":
                            out = api.student_forward(S, x, attn_layers=list(range(START, depth)))
                            keep = _topk_mask(_rollout([a for a in out.attn if a is not None]), K)
                        else:
                            out = api.student_forward(S, x, attn_layers=[depth - 1])
                            keep = _topk_mask(out.attn[depth - 1].float().mean(1)[:, 0, 1:], K)
                        gp.append(api.teacher_forward_keepmask(Tm, x, keep).float().argmax(1).cpu())
                del S
                torch.cuda.empty_cache()
                t = (torch.cat(gp).numpy() != y).astype(float)
                s = (torch.cat(sp).numpy() != y).astype(float)
                tw[(rid, tag)], sw[(rid, tag)] = t, s
                per_run[f"{rid}@{tag}"] = {
                    "arm": arm, "seed": runs[rid]["seed"], "epoch": tag,
                    "student_acc": round(float((1 - s.mean()) * 100), 2),
                    "student_tail_acc": round(float((1 - s[tail].mean()) * 100), 2),
                    "masked_teacher_acc": round(float((1 - t.mean()) * 100), 2),
                    "masked_teacher_tail_acc": round(float((1 - t[tail].mean()) * 100), 2)}

    def arm_mean(d, arm, tag):
        return numpy.mean([d[(r, tag)] for r in RUNS[arm]], axis=0)

    diffs = {}
    for a in ("frozen", "maskedkd"):
        for tag in ("e50", "final"):
            diffs[(a, tag)] = (arm_mean(tw, a, tag) - arm_mean(tw, "lateroll", tag),
                               arm_mean(sw, a, tag) - arm_mean(sw, "lateroll", tag))
    rng = numpy.random.default_rng(SEED)
    out = {}
    for name, m in (("tail", tail), ("head", head), ("all", numpy.ones(n, dtype=bool))):
        out[name] = _boot4({k: (v[0][m], v[1][m]) for k, v in diffs.items()}, rng)
        out[name]["means_pp"] = {f"{a}_{t}": [round(float(v[0][m].mean() * 100), 4),
                                              round(float(v[1][m].mean() * 100), 4)]
                                 for (a, t), v in diffs.items()}

    seedwise = {}
    for a in ("frozen", "maskedkd"):
        for tag in ("e50", "final"):
            seedwise[f"{a}_{tag}"] = [
                _contrast((tw[(r, tag)] - tw[(b, tag)])[tail], (sw[(r, tag)] - sw[(b, tag)])[tail])
                for r, b in zip(RUNS[a], RUNS["lateroll"])]
    sw_change = {a: (numpy.array(seedwise[f"{a}_final"]) - numpy.array(seedwise[f"{a}_e50"])) for a in
                 ("frozen", "maskedkd")}

    def verdict(c):
        return "positive" if c[1] > 0 else ("negative" if c[2] < 0 else "uncertain")

    P1 = out["tail"]["C_maskedkd_final_minus_e50_pp_boot"]
    api.write_json("summary", {
        "n_val": int(n), "n_tail": int(tail.sum()), "n_head": int(head.sum()), "K": K, "n_students": len(per_run),
        "P1_tail_C_maskedkd_final_minus_e50_pp_boot": P1,
        "record": {
            "contrasts": out,
            "tail_seedwise_C_pp": {k: [round(c, 4) for c in v] for k, v in seedwise.items()},
            "tail_seedwise_C_final_minus_e50_t": {a: _tcrit(v) for a, v in sw_change.items()},
        },
        "per_run": per_run,
        "verdict": {"P1": verdict(P1),
                    "frozen_final_minus_e50_tail": verdict(out["tail"]["C_frozen_final_minus_e50_pp_boot"]),
                    "mkd_minus_frozen_e50_tail": verdict(out["tail"]["C_maskedkd_minus_frozen_e50_pp_boot"]),
                    "gap_change_tail": verdict(out["tail"]["gap_change_final_minus_e50_pp_boot"])},
        "note": "CI entries are [point, low, high]; dT, dS = arm - lateroll error rates (seed mean) at the same epoch; "
                "groups by dT sign; tail = val group 1; one rng shared in order tail, head, all; masks and student "
                "predictions from the same epoch-E student of each run"})
