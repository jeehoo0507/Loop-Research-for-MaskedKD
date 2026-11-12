"""Surgical study, offline sweep (human, 2026-11-12): which student blocks and heads give token rankings under which
the masked teacher still answers like the full teacher, on this dataset's clean val images. No training.

For up to two trained students of this dataset (the CE run and the Full KD run with the smallest seed; neither was
trained with a token selector, so no arm is favoured by the student having adapted to it), in eval mode:
every arm ranks the 196 patches by attention rollout over its block set (the rule of selectors/layerroll.py and of the
harness Rollout baseline), the teacher gets the top K, K in K_LIST, and three things are recorded per image:
  correct  the masked teacher's top class is the label
  agree    the masked teacher's top class is the full teacher's top class
  kl       KL(full teacher || masked teacher)
Arms: each block alone with the head mean (L0..L11; L11 is MaskedKD), each head of each block (L<l>H<h>), the four
block groups of Surgical Fine-Tuning (B1 0-2, B2 3-5, B3 6-8, B4 9-11), all blocks (rollout), all but the last block,
all but the first group, blocks 6-11 (lateroll), blocks 6-10, the two "no middle" sets, and uniform random tokens.
Summary per student and K: mean of the three measures per arm, accuracy per group where the dataset has groups, and
the paired difference of `correct` to L11 with an image-bootstrap 95% interval (NBOOT draws, one set of draws for all
arms). The numbers are masked-teacher fidelity, not student accuracy: earlier analyses here found fidelity ranks
trained accuracy only coarsely (run-level Spearman about -0.5 to -0.7), so this sweep chooses what to train, it does
not replace training.
"""
import numpy
import torch

META = {
    "question": "which student blocks and heads rank the patches so that the masked teacher keeps the full "
                "teacher's answer, and does the best block set differ between datasets with different kinds of "
                "distribution shift (surgical fine-tuning: input shift -> first blocks, output shift -> last)?",
    "method": "attention rollout over block subsets and single heads; keep-only masked-teacher fidelity (accuracy, "
              "agreement with the full teacher, KL); image bootstrap",
    "paper": "Abnar & Zuidema ACL 2020 (rollout); Lee et al. ICLR 2023 (surgical fine-tuning); Son et al. ECCV 2024 "
             "(MaskedKD); DeYoung et al. ACL 2020 (ERASER sufficiency, keep-only)",
}

K_LIST = (29, 49)
BATCH = 50
NBOOT = 1000
DEPTH = 12
EPS = 1e-8
GROUPS = {"B1 0-2": [0, 1, 2], "B2 3-5": [3, 4, 5], "B3 6-8": [6, 7, 8], "B4 9-11": [9, 10, 11]}
RANGES = {
    "all 0-11 (rollout)": list(range(12)),
    "no last 0-10": list(range(11)),
    "no first 3-11": list(range(3, 12)),
    "late 6-11 (lateroll)": list(range(6, 12)),
    "late no last 6-10": list(range(6, 11)),
    "no middle 0-2,9-11": [0, 1, 2, 9, 10, 11],
    "no middle 0-5,9-11": [0, 1, 2, 3, 4, 5, 9, 10, 11],
}


def _topk(s, k):
    idx = s.topk(k, dim=1).indices
    return torch.zeros_like(s, dtype=torch.bool).scatter_(1, idx, True)


def _scores(attn, layers, head):
    """The ranking of selectors/layerroll.py: one block is read as it is; several are rolled out."""
    pick = lambda i: attn[i].float().mean(1) if head is None else attn[i].float()[:, head]   # noqa: E731
    if len(layers) == 1:
        return pick(layers[0])[:, 0, 1:]
    r = None
    for i in layers:
        a = pick(i)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _arms(heads):
    arms = {f"L{i}": ([i], None) for i in range(DEPTH)}
    for i in range(DEPTH):
        for h in range(heads):
            arms[f"L{i}H{h}"] = ([i], h)
    for name, layers in list(GROUPS.items()) + list(RANGES.items()):
        arms[name] = (layers, None)
    return arms


def _students(api):
    """{label: (run id, model)}: the CE and the Full KD student of this dataset with the smallest seed (final tier
    first) whose weights are on this server (a run recorded in git but trained on another server has none)."""
    got = {}
    runs = api.runs()
    for want in ("ce", "full"):
        cands = sorted((r["tier"] != "final", r["seed"], rid) for rid, r in runs.items()
                       if r.get("dataset") == api.dataset and r["method"] == want and r["tier"] in ("final", "screen"))
        for _, _, rid in cands:
            try:
                got[want] = (rid, api.student(rid))
                break
            except ValueError:
                continue
    return got


def _measure(api, student, gen):
    """Per-image tensors {K: {arm: {"correct","agree","kl"}}} on the val set, plus labels."""
    rows, ys, arms = {k: {} for k in K_LIST}, [], None
    for x, y in api.val_batches(batch=BATCH):
        with torch.no_grad():
            out = api.student_forward(student, x, attn_layers=range(DEPTH))
            full = api.teacher(x).float().log_softmax(1)
            top_full = full.argmax(1)
            if arms is None:
                arms = _arms(out.attn[0].shape[1])
            scores = {name: _scores(out.attn, layers, head) for name, (layers, head) in arms.items()}
            scores["random"] = torch.from_numpy(gen.random((x.shape[0], api.num_patches))).float().to(x.device)
            for k in K_LIST:
                for name, s in scores.items():
                    logp = api.teacher_forward_keepmask(api.teacher, x, _topk(s, k)).float().log_softmax(1)
                    rec = rows[k].setdefault(name, {"correct": [], "agree": [], "kl": []})
                    rec["correct"].append((logp.argmax(1) == y).float().cpu())
                    rec["agree"].append((logp.argmax(1) == top_full).float().cpu())
                    rec["kl"].append((full.exp() * (full - logp)).sum(1).clamp_min(0).cpu())
        ys.append(y.cpu())
    return {k: {n: {m: torch.cat(v).numpy() for m, v in rec.items()} for n, rec in rows[k].items()} for k in K_LIST}, \
        torch.cat(ys).numpy()


def _summarise(per_k, groups, rng):
    out = {}
    n = None
    for k, arms in per_k.items():
        ref = arms["L11"]["correct"]
        n = len(ref)
        draws = rng.integers(0, n, (NBOOT, n))
        block = {}
        for name, rec in arms.items():
            d = rec["correct"] - ref
            boot = d[draws].mean(1)
            row = {"acc": 100 * float(rec["correct"].mean()), "agree": 100 * float(rec["agree"].mean()),
                   "kl": float(rec["kl"].mean()),
                   "acc_minus_L11": 100 * float(d.mean()),
                   "acc_minus_L11_ci": [100 * float(numpy.quantile(boot, 0.025)), 100 * float(numpy.quantile(boot, 0.975))]}
            if groups is not None:
                per = {int(g): 100 * float(rec["correct"][groups == g].mean()) for g in numpy.unique(groups)}
                row["group_acc"] = per
                row["worst_group_acc"] = min(per.values())
                row["acc_minus_worst"] = row["acc"] - row["worst_group_acc"]
            block[name] = row
        out[str(k)] = block
    return out, n


def _curve(block, key):
    """The block-group and single-block curves of one measure, for a quick look at where the peak is."""
    return {"groups": {g: block[g][key] for g in GROUPS}, "blocks": [block[f"L{i}"][key] for i in range(DEPTH)]}


def run(api):
    students = _students(api)
    if not students:
        api.write_json("summary", {"dataset": api.dataset, "error": "no finished CE or Full KD run of this dataset"})
        return
    g = api.val_groups()
    groups = None if g is None else g.numpy()
    rng = numpy.random.default_rng(0)
    summary = {"dataset": api.dataset, "k_list": list(K_LIST), "students": {k: v[0] for k, v in students.items()},
               "results": {}, "curves": {}}
    for label, (rid, student) in students.items():
        gen = numpy.random.default_rng(1)   # the random arm: same draws for every student
        per_k, _ = _measure(api, student, gen)
        res, n = _summarise(per_k, groups, rng)
        summary["results"][label] = res
        summary["n_val"] = n
        key = "worst_group_acc" if groups is not None else "acc"
        summary["curves"][label] = {k: {"measure": key, **_curve(block, key)} for k, block in res.items()}
    api.write_json("summary", summary)
