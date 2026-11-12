"""Results of the surgical study from runs/*/metrics.json (read-only; prints Markdown).

    python -m surgical.table                       # every dataset that has runs
    python -m surgical.table --datasets waterbirds celeba --keep 29

Per dataset: each arm's accuracy and worst-group accuracy over seeds, and its paired difference to MaskedKD
(95% t-interval over shared seeds, harness.report.paired). Then the two readings the plan fixed in advance:
  follow - avoid   the arm that selects with the block group surgical fine-tuning names, minus the arm without it
  loss per arm     what the shift costs each arm: clean CIFAR-10 minus the corrupted set (same images, same seed), or
                   mean accuracy minus worst-group accuracy. The surgical trend carries over if the loss is largest
                   for the block group surgical fine-tuning names (first blocks for input shift, last for output).
Seeds from the final and the screen tier are pooled when they ran the same number of epochs.
"""
import argparse
import statistics

from harness import config as C
from harness.report import load_runs, paired

from .arms import EXTRA, K, REFERENCE, SELECT, SHIFT, arm_of, teacher_sha, usable

ORDER = list(REFERENCE) + list(SELECT) + list(EXTRA)
SKIPPED = {}   # dataset -> KD runs left out because they used another teacher
ANY_TEACHER = False   # --any-teacher: show them anyway (reading the old results on a machine without the checkpoints)


def arms_by_seed(dataset, keep):
    """{arm: {seed: run}}; a final-tier run wins over a screen-tier run of the same seed. KD runs made with a
    teacher other than this server's are left out (counted in SKIPPED)."""
    final_epochs = C.TIER_EPOCHS.get(dataset, {}).get("final")
    sha = teacher_sha(C.TEACHER_SHARED.get(dataset, dataset))
    out = {}
    for r in sorted(load_runs(dataset), key=lambda r: (r["tier"] == "final", r["id"])):
        if r["tier"] not in ("final", "screen") or r.get("epochs") != final_epochs:
            continue
        name = arm_of(r["method"], r.get("sel_args"))
        if name is None or (r["method"] not in ("ce", "full") and float(r["keep"]) != float(keep)):
            continue
        if not ANY_TEACHER and not usable(r, sha):
            SKIPPED[dataset] = SKIPPED.get(dataset, 0) + 1
            continue
        out.setdefault(name, {})[r["seed"]] = r
    return out


def _ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{statistics.mean(vals):.2f}" + (f" ± {statistics.stdev(vals):.2f}" if len(vals) > 1 else "")


def _ci(p):
    return "-" if p is None else f"{p['diff']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}] (n={len(p['seeds'])})"


def _with(runs, key, fn):
    """Copy of {seed: run} with run[key] = fn(run), for paired() on a derived quantity."""
    return {s: {**r, key: fn(r)} for s, r in runs.items()}


def section(dataset, keep):
    SKIPPED.pop(dataset, None)
    arms = arms_by_seed(dataset, keep)
    skipped = SKIPPED.get(dataset, 0)
    if not arms:
        return ([f"## {dataset}", "", f"- {skipped} KD runs exist but were made with another teacher than this server's "
                 "(or there is no teacher here yet): not shown.", ""] if skipped else [])
    grouped = bool(C.NUM_GROUPS[dataset])
    wg = "tail acc" if dataset == "cub_lt" else "worst-group acc"
    kind, follow, avoid = SHIFT.get(dataset, ("?", None, None))
    out = [f"## {dataset} (shift: {kind}, K={keep:g})", "",
           "| arm | seeds | val acc | " + (f"{wg} | " if grouped else "") + "acc − MaskedKD | "
           + (f"{wg} − MaskedKD | " if grouped else "") + "GFLOPs/img |", "|---|---|---|" + ("---|---|" if grouped else "") + "---|---|"]
    ref = arms.get("last (MaskedKD)", {})
    for name in ORDER:
        rs = arms.get(name)
        if not rs:
            continue
        row = [name, ",".join(str(s) for s in sorted(rs)), _ms([r["val_acc_last"] for r in rs.values()])]
        if grouped:
            row.append(_ms([r.get("val_wga_last") for r in rs.values()]))
        row.append("-" if name == "last (MaskedKD)" else _ci(paired(rs, ref)))
        if grouped:
            row.append("-" if name == "last (MaskedKD)" else _ci(paired(rs, ref, key="val_wga_last")))
        fl = [r["flops"]["total_gflops_per_img"] for r in rs.values() if r.get("flops")]
        row.append(f"{statistics.mean(fl):.2f}" if fl else "-")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    if skipped:
        out.append(f"- {skipped} KD runs made with another teacher than this server's are left out.")
    gate = paired(arms.get("Full KD", {}), arms.get("CE", {}))
    out.append(f"- KD gain (Full KD − CE): {_ci(gate)}" + ("" if gate is None or gate["lo"] > 0 else
               "  ← not above 0: this dataset cannot carry a training claim"))
    if follow and arms.get(follow) and arms.get(avoid):
        key = "val_wga_last" if grouped else "val_acc_last"
        out.append(f"- follow − avoid ({follow} − {avoid}, {'worst group' if grouped else 'acc'}): "
                   f"{_ci(paired(arms[follow], arms[avoid], key=key))}")
    if grouped:
        out += ["", f"Loss per arm = val acc − {wg} (larger = the minority pays more):", ""]
        for name in ORDER:
            rs = arms.get(name)
            if rs and all(r.get("val_wga_last") is not None for r in rs.values()):
                out.append(f"- {name}: {_ms([r['val_acc_last'] - r['val_wga_last'] for r in rs.values()])}")
    out.append("")
    return out


def cifar_loss(keep):
    """Clean minus corrupted, per arm, paired by seed (same images in every split)."""
    clean = arms_by_seed("cifar10", keep)
    out = []
    for d in C.CIFAR_C:
        cor = arms_by_seed(d, keep)
        rows = [(n, paired(clean.get(n, {}), cor[n])) for n in ORDER if cor.get(n)]
        rows = [(n, p) for n, p in rows if p is not None]
        if not rows:
            continue
        out += [f"## Input shift: cifar10 − {d} per arm (K={keep:g})", "",
                "Largest for B1 0-2 = the surgical trend carries over to token selection.", ""]
        out += [f"- {n}: {_ci(p)}" for n, p in rows] + [""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS), choices=list(C.DATASETS))
    ap.add_argument("--keep", type=float, default=K)
    ap.add_argument("--any-teacher", action="store_true",
                    help="include KD runs made with another teacher than this server's (never mix them into a claim)")
    a = ap.parse_args()
    global ANY_TEACHER
    ANY_TEACHER = a.any_teacher
    body = ["# Surgical study: token selection by block set", ""]
    for d in a.datasets:
        body += section(d, a.keep)
    body += cifar_loss(a.keep)
    print("\n".join(body))


if __name__ == "__main__":
    main()
