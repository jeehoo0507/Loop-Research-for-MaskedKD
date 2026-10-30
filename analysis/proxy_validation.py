"""H-W10: which masked-teacher quantity predicts student WGA?

The clean-image masked-teacher WGA proxy (analysis_group_fidelity_1030_143842_1ea0) ranked 5 methods correctly
at K=29/49 with the full-KD student, but it (a) over-predicted midattn@49 and (b) at K=12 ranks center (41.35)
below rollout / block-9 top-K (48-60, analysis_window_fidelity_1030_165418_e276) while the students are the
other way round (center@12 WGA 56.39 vs rollout@12 45.11, screen seed 0). analysis_train_condition_gap_1030_
162956_067e showed that (a) disappears when the proxy is measured with the student the selector was trained
with ("self" proxy: L9 on the midattn student falls 66.92 -> 59.40 -> 60.90 over e25/e50/final).

Here every finished screen seed-0 run of maskedkd / rollout / center / grid / random / midattn at K = 12, 29, 49
is scored with its own final student (self) and with the full-KD student (common), on clean val images.
Per val group of the masked teacher's output:
  t_acc    top-1 accuracy on the label
  p_true   mean probability of the true class (what the KD target pulls the student towards)
  cwrong   share of images the masked teacher gets wrong with max-prob > 0.8 (confidently wrong)
  ent      mean entropy (nats)
  kl       KL(full teacher || masked teacher)
Each quantity is reduced to its worst group (min for t_acc / p_true, max for cwrong / kl) and to the minority
mean (groups 1, 2), then Spearman-correlated with the runs' val WGA (runs()[id]['val_wga_last']) over all
(method, K) points and within each K. The question is which reduction ranks the students, not which selector wins.
"""
import torch

META = {
    "question": "which masked-teacher statistic (accuracy, true-class probability, confident errors, entropy, "
                "KL to the full teacher; worst-group or minority mean; self vs common student) rank-predicts "
                "the student's worst-group accuracy across selectors and K?",
    "method": "group-sliced faithfulness and calibration of the masked teacher under each selector's attention "
              "map (CLS attention, attention rollout), validated against trained students by rank correlation",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Hooker et al. NeurIPS 2019 (ROAR-style keep-only evaluation); "
             "Sagawa et al. ICLR 2020 (Waterbirds groups); Guo et al. ICML 2017 (calibration)",
}

KS = (12, 29, 49)
SIDE = 14
METHODS = ("maskedkd", "rollout", "center", "grid", "random", "midattn")
STATS = ("t_acc", "p_true", "cwrong", "ent", "kl")


def _topk(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def _scores(api, student, x, n):
    depth = len(student.blocks)
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = api.student_forward(student, x, attn_layers=list(range(depth)))
    A = [a.float().mean(1) for a in out.attn]
    B, N, _ = A[0].shape
    eye = torch.eye(N, device=x.device).expand(B, N, N)
    R = None
    for a in A:
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        R = a if R is None else a @ R
    return {"maskedkd": A[-1][:, 0, 1:], "midattn": A[9][:, 0, 1:], "rollout": R[:, 0, 1:]}


def _fixed(method, B, n, k, dev):
    if method == "center":
        r = torch.arange(n, device=dev)
        c = (SIDE - 1) / 2
        dist = ((r // SIDE - c) ** 2 + (r % SIDE - c) ** 2).float() + r.float() * 1e-6
        return _topk((-dist).expand(B, n), k)
    if method == "grid":
        idx = torch.linspace(0, n - 1, k, device=dev).round().long()
        return torch.zeros(B, n, dtype=torch.bool, device=dev).index_fill_(1, idx, True)
    return _topk(torch.rand(B, n, device=dev), k)   # random


def _rank(v):
    v = torch.tensor(v, dtype=torch.float64)
    order = v.argsort()
    r = torch.empty_like(v)
    r[order] = torch.arange(len(v), dtype=torch.float64)
    for val in v.unique():          # average ties
        m = v == val
        r[m] = r[m].mean()
    return r


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = (ra.norm() * rb.norm()).item()
    return None if d == 0 else round((ra @ rb).item() / d, 3)


def run(api):
    runs = api.runs()
    targets = []    # (method, K, run id, student WGA, student acc)
    full_rid = None
    for rid, r in sorted(runs.items()):
        if r["tier"] != "screen" or r["seed"] != 0 or r.get("dataset", "waterbirds") != "waterbirds":
            continue
        if r["method"] == "full":
            full_rid = rid
        if r["method"] in METHODS and int(r["keep"]) in KS:
            targets.append((r["method"], int(r["keep"]), rid, r["val_wga_last"], r["val_acc_last"]))
    api.write_json("targets", {"full": full_rid, "runs": targets})

    groups = api.val_groups()
    T = api.teacher.eval()
    n = api.num_patches
    batches = []
    off = 0
    for x, y in api.val_batches(batch=50):
        B = x.shape[0]
        gb = groups[off:off + B]
        off += B
        with torch.no_grad():
            full = T(x).float().log_softmax(-1)
        batches.append((x.cpu(), y.cpu(), gb.cpu(), full.cpu(), x.device))
    gn = [sum(int((b[2] == gi).sum()) for b in batches) for gi in range(4)]

    def score(student, method, k):
        acc = {s: [0.0] * 4 for s in STATS}
        for x, y, gb, full, dev in batches:
            x, y, gb, full = x.to(dev), y.to(dev), gb.to(dev), full.to(dev)
            B = x.shape[0]
            if method in ("center", "grid", "random"):
                keep = _fixed(method, B, n, k, dev)
            else:
                keep = _topk(_scores(api, student, x, n)[method], k)
            with torch.no_grad():
                lm = api.teacher_forward_keepmask(T, x, keep).float().log_softmax(-1)
            pm = lm.exp()
            pred = pm.argmax(-1)
            wrong = pred != y
            vals = {"t_acc": (~wrong).float(),
                    "p_true": pm.gather(1, y[:, None])[:, 0],
                    "cwrong": (wrong & (pm.max(-1).values > 0.8)).float(),
                    "ent": -(pm * lm).sum(-1),
                    "kl": (full.exp() * (full - lm)).sum(-1)}
            for s, v in vals.items():
                for gi in range(4):
                    acc[s][gi] += v[gb == gi].sum().item()
        o = {s: [round(v / max(c, 1), 4) for v, c in zip(acc[s], gn)] for s in STATS}
        o["t_acc"] = [round(100 * v, 2) for v in o["t_acc"]]
        o["cwrong"] = [round(100 * v, 2) for v in o["cwrong"]]
        red = {}
        for s in STATS:
            worst = min(o[s]) if s in ("t_acc", "p_true") else max(o[s])
            red[f"{s}_worst"] = round(worst, 4)
            red[f"{s}_minor"] = round((o[s][1] + o[s][2]) / 2, 4)
        return o, red

    common = api.student(full_rid) if full_rid else None
    rows = {}
    for method, k, rid, wga, acc in targets:
        st = api.student(rid) if method in ("maskedkd", "rollout", "midattn") else None
        if st is not None:
            for p in st.parameters():
                p.requires_grad_(False)
        o_self, r_self = score(st, method, k)
        o_com, r_com = (o_self, r_self) if st is None else score(common, method, k)
        rows[f"{method}@{k}"] = {"run": rid, "student_wga": wga, "student_acc": acc,
                                 "self": o_self, "self_red": r_self, "common": o_com, "common_red": r_com}
        del st
    api.write_json("rows", rows)

    keys = list(next(iter(rows.values()))["self_red"].keys()) if rows else []
    corr = {}
    for src in ("self_red", "common_red"):
        for key in keys:
            sign = 1 if key.split("_")[0] in ("t_acc", "p_true") else -1
            pts = [(sign * r[src][key], r["student_wga"]) for r in rows.values() if r["student_wga"] is not None]
            c = {"all": _spearman([a for a, _ in pts], [b for _, b in pts]), "n_all": len(pts)}
            for k in KS:
                pk = [(sign * r[src][key], r["student_wga"]) for name, r in rows.items()
                      if name.endswith(f"@{k}") and r["student_wga"] is not None]
                c[f"K{k}"] = _spearman([a for a, _ in pk], [b for _, b in pk])
            corr[f"{src.split('_')[0]}/{key}"] = c
    table = {name: [r["student_wga"], r["self_red"]["t_acc_worst"], r["self_red"]["p_true_minor"],
                    r["self_red"]["cwrong_worst"], r["self_red"]["ent_minor"], r["common_red"]["t_acc_worst"]]
             for name, r in rows.items()}
    api.write_json("summary", {"group_n": gn,
                               "note": "Spearman of (sign-adjusted) proxy vs student val WGA; +1 = perfect rank",
                               "corr": corr,
                               "table_cols": ["student_wga", "self t_acc_worst", "self p_true_minor",
                                              "self cwrong_worst", "self ent_minor", "common t_acc_worst"],
                               "table": table})
