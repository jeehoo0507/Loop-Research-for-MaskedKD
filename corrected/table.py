"""Results of the corrected-distillation study from runs/*/metrics.json (read-only; prints Markdown).

    python -m corrected.table                          # every dataset that has runs of the study
    python -m corrected.table --datasets waterbirds celeba

Per dataset and budget: each arm's accuracy (and worst-group accuracy) over seeds, its paired difference to MaskedKD
and to CE on the dataset's primary metric (worst-group accuracy where the dataset has groups, else accuracy; 95 %
t-interval over shared seeds), and the teacher GFLOPs per image the harness measured. Then the readings fixed in
corrected/README.md before any run, the prediction of the no-training dissection next to what training gave, and
what the tokens + images run measured on its own paired images during training.
"""
import argparse
import json
import statistics

from harness import config as C
from harness.report import load_runs, paired
from surgical.arms import teacher_sha, usable
from surgical.table import _ci, _ms

from .arms import BATCH, BUDGETS, IMAGES, MASKEDKD, REFERENCE, SPLITS, TOKENS, UNBIASED, arm_of, arms

MARKS = (0.0, 0.1, 0.25, 0.5, 1.0)   # where in training the diagnostics are read


def runs_by_arm(dataset):
    """{(budget or None, arm): {seed: run}}: final-tier runs at the current epochs, made with this server's teacher."""
    epochs = C.TIER_EPOCHS.get(dataset, {}).get("final")
    sha = teacher_sha(C.TEACHER_SHARED.get(dataset, dataset))
    out = {}
    for r in sorted(load_runs(dataset), key=lambda r: r["id"]):
        if r["tier"] != "final" or r.get("epochs") != epochs or not usable(r, sha):
            continue
        key = arm_of(r["method"], r["keep"], r.get("sel_args"))
        if key is not None:
            out.setdefault(key, {})[r["seed"]] = r
    return out


def needed_gain(budget):
    """Share of what the free answers leave that the masked answer must explain for tokens + images to add less
    noise than images only, at this budget's splits: both draw without replacement, noise (1/m - 1/B) * residual."""
    m1, (_, m2) = SPLITS[budget]["img"], SPLITS[budget]["two"]
    return 1 - (1 / m1 - 1 / BATCH) / (1 / m2 - 1 / BATCH)


def latest_dissection(dataset):
    """Summary of the newest finished dissection of this dataset, or None."""
    best = None
    for f in sorted((C.ROOT / "analysis_out").glob("analysis_masked_teacher_dissection_*/summary.json")):
        s = json.loads(f.read_text())
        if s.get("dataset") == dataset and s.get("design"):
            best = s
    return best


def primary_student(summary):
    """The student the prediction is read from: the Full KD run a quarter into training if its snapshot is here
    (training is where the designs differ; a finished student already agrees with the teacher), else the CE one."""
    snaps = sorted(k for k in summary["design"] if k.startswith("full_e"))
    return snaps[0] if snaps else "ce" if "ce" in summary["design"] else sorted(summary["design"])[0]


def prediction_lines(dataset, summary):
    who = primary_student(summary)
    lines = []
    for budget in BUDGETS:
        d = summary["design"][who].get(str(budget))
        if d is None:
            continue
        lines.append(f"- budget {budget}: predicted better = **{d['predicted_better']}** (student {who}: added noise "
                     f"over full KD's own, images only {d['images_only_var_over_noise']:.3f}, tokens + images "
                     f"{d['tokens_images_var_over_noise']:.3f}; masked answer explains "
                     f"{100 * d['gain_beyond_free']:.0f} % beyond the free ones, needs {100 * d['gain_needed']:.0f} %; "
                     f"MaskedKD bias^2 / noise {d['maskedkd_bias_sq_over_noise']:.3f})")
    others = {k: {b: v["predicted_better"] for b, v in summary["design"][k].items()} for k in summary["design"] if k != who}
    lines.append(f"- other students, same question: {json.dumps(others)}")
    return lines


def _teacher_gflops(rs):
    v = [r["flops"]["teacher_gflops_per_img"] for r in rs.values() if r.get("flops") and "teacher_gflops_per_img" in r["flops"]]
    return statistics.mean(v) if v else None


def _diag_rows(rs, budget):
    """Per mark of training, over the seeds of the tokens + images run: what its paired images measured."""
    need = needed_gain(budget)
    out = []
    for mark in MARKS:
        gains, ws, res = [], [], []
        for r in rs.values():
            h = [e for e in r["history"] if e.get("tp") and "res_all" in e["tp"]]
            if not h:
                continue
            t = h[min(len(h) - 1, int(round(mark * (len(h) - 1))))]["tp"]
            gains.append(1 - t["res_all"] / max(t["res_free"], 1e-12))
            ws.append(t.get("w_used") or t["w_all"])
            res.append((t["res_student"], t["res_label"], t["res_masked"], t["res_free"], t["res_all"]))
        if gains:
            g = statistics.mean(gains)
            w = [statistics.mean(x) for x in zip(*ws)]
            r5 = [statistics.mean(x) for x in zip(*res)]
            out.append(f"| {100 * mark:.0f} % | {r5[0]:.4f} | {r5[1]:.4f} | {r5[2]:.4f} | {r5[3]:.4f} | {r5[4]:.4f} | "
                       f"{100 * g:.0f} % | {100 * need:.0f} % | {'tokens + images' if g > need else 'images only'} | "
                       f"{w[0]:.2f} / {w[1]:.2f} / {w[2]:.2f} |")
    return out


def section(dataset):
    by = runs_by_arm(dataset)
    if not any(k[0] is not None and k[1] != MASKEDKD and k[1] != "rollout" for k in by):
        return []
    grouped = bool(C.NUM_GROUPS[dataset])
    key = "val_wga_last" if grouped else "val_acc_last"
    wg = "tail acc" if dataset == "cub_lt" else "worst-group acc"
    ce, full = by.get((None, "CE"), {}), by.get((None, "Full KD"), {})
    gate = paired(full, ce, key=key)
    out = [f"## {dataset}", "", f"Primary metric: {wg if grouped else 'val acc'}. KD gain on it (Full KD − CE): {_ci(gate)}"
           + ("" if gate is None or gate["lo"] > 0 else "  ← not above 0: read this dataset as an observation, not a claim"), ""]
    pred = latest_dissection(dataset)
    for budget in BUDGETS:
        A = {name: by.get((budget, name), {}) for name in arms(budget)}
        if not any(A[n] for n in A if n not in (MASKEDKD, "rollout")):
            continue
        ref = A[MASKEDKD]
        ref_fl = _teacher_gflops(ref) if ref else None
        out += [f"### budget {budget} (the teacher compute of MaskedKD at K={budget})", "",
                "| arm | seeds | val acc | " + (f"{wg} | " if grouped else "") + "primary − MaskedKD | primary − CE | "
                "teacher GFLOPs/img | vs MaskedKD |", "|---|---|---|" + ("---|" if grouped else "") + "---|---|---|---|"]
        rows = [(n, rs) for n, rs in (("CE", ce), ("Full KD", full))] + list(A.items())
        for name, rs in rows:
            if not rs:
                continue
            fl = None if name == "CE" else _teacher_gflops(rs)
            row = [name, ",".join(str(s) for s in sorted(rs)), _ms([r["val_acc_last"] for r in rs.values()])]
            if grouped:
                row.append(_ms([r.get("val_wga_last") for r in rs.values()]))
            row += ["-" if name == MASKEDKD else _ci(paired(rs, ref, key=key)), "-" if name == "CE" else _ci(paired(rs, ce, key=key)),
                    "-" if fl is None else f"{fl:.2f}", "-" if not (fl and ref_fl) else f"{fl / ref_fl:.3f}"]
            out.append("| " + " | ".join(row) + " |")
        out += ["", "Readings fixed before the runs (corrected/README.md):", ""]
        for n in (IMAGES, TOKENS):
            out.append(f"- R1 unbiased vs MaskedKD: {n} − MaskedKD = {_ci(paired(A[n], ref, key=key))}")
        diff = paired(A[TOKENS], A[IMAGES], key=key)
        said = None if pred is None else pred["design"][primary_student(pred)].get(str(budget), {}).get("predicted_better")
        got = None if diff is None else "no difference" if diff["lo"] <= 0 <= diff["hi"] else "tokens + images" if diff["diff"] > 0 else "images only"
        out.append(f"- R2 tokens or images: {TOKENS} − {IMAGES} = {_ci(diff)}; predicted better: {said or '-'}; "
                   f"trained better: {got or '-'}")
        out.append(f"- R3 the weight B/m: images − images, unweighted = "
                   f"{_ci(paired(A['images'], A['images, unweighted (rand_dyn)'], key=key))}")
        floor = [f"{n} {_ci(p)}" for n in UNBIASED if (p := paired(A.get(n, {}), ce, key=key)) is not None]
        out.append("- R4 no worse than no teacher (arm − CE): " + ("; ".join(floor) or "-"))
        out.append(f"- R4, MaskedKD − CE: {_ci(paired(ref, ce, key=key))}")
        diag = _diag_rows(A[TOKENS], budget)
        if diag:
            out += ["", f"Measured during training by {TOKENS} on its own paired images (mean |T − h|² per cheap answer h; "
                    "free = best mix of student and label, all = with the recalibrated masked answer):", "",
                    "| training done | student | label | masked | free | all | masked explains beyond free | needed | "
                    "says | weights student / label / masked |", "|---|---|---|---|---|---|---|---|---|---|"] + diag
        out.append("")
    if pred is not None:
        out += ["Prediction of the dissection (no training):", ""] + prediction_lines(dataset, pred) + [""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS), choices=list(C.DATASETS))
    a = ap.parse_args()
    body = ["# Corrected distillation: the same teacher compute, spent on tokens or on images", ""]
    for d in a.datasets:
        body += section(d)
    print("\n".join(body))


if __name__ == "__main__":
    main()
