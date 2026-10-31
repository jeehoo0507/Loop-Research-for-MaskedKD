"""Decisions agent/bootstrap.sh makes by rule (so nobody has to type numbers into config.py).

    python -m harness.boot jobs-from-bench   # MAX_JOBS from state/bench.json (epochs are fixed by the recipe: 100)
    python -m harness.boot pick-keeps        # stage-2 baseline budgets from the stage-1 baselines
    python -m harness.boot lr-pick <dataset> # student lr from the CE sweep (submit --lr-sweep) -> state/recipe.json

Rules (agreed in planning):
  jobs-from-bench: the parallelism bench found best (aggregate img/s), written to state/epochs.json (locked).
  pick-keeps: the two budgets where MaskedKD falls furthest below Full KD on stage-1 screen seed 0 (accuracy),
  i.e. where token choice matters; default 49, 29.
  lr-pick: the CE run (screen, seed 0, 100 epochs) with the best final val accuracy among config.STUDENT_LRS
  (aim-lab-test-5's rule; ties -> smaller lr). CE and every KD method then use it.
"""
import json
import sys

from . import config as C


def _load():
    return json.loads(C.EPOCHS_FILE.read_text()) if C.EPOCHS_FILE.exists() else {"epochs": {}}


def _save(d):
    C.STATE.mkdir(exist_ok=True)
    C.EPOCHS_FILE.write_text(json.dumps(d, indent=1) + "\n")


def jobs_from_bench():
    rec = json.loads((C.STATE / "bench.json").read_text())["recommend"]
    d = _load()
    d["max_jobs"] = int(rec["parallel_jobs"])
    d["rule"] = "epochs fixed at 100 (aim-lab-test-5 recipe); max_jobs = best bench parallelism"
    _save(d)
    print(json.dumps(d))


def lr_pick(dataset):
    scores = {}
    for m in C.RUNS.glob("*/metrics.json"):
        r = json.loads(m.read_text())
        if (r.get("status") == "done" and r.get("dataset") == dataset and r["method"] == "ce"
                and r.get("lr_override") is not None and r["tier"] == "screen"):
            scores[float(r["lr_override"])] = r["val_acc_last"]
    if set(scores) != set(C.STUDENT_LRS):
        raise SystemExit(f"lr sweep incomplete for {dataset}: have {sorted(scores)}, need {C.STUDENT_LRS}")
    lr = max(sorted(scores), key=lambda l: (scores[l], -l))
    d = json.loads(C.RECIPE_FILE.read_text()) if C.RECIPE_FILE.exists() else {}
    d[dataset] = {"student_lr": lr, "swept": True, "scores": {f"{k:g}": v for k, v in sorted(scores.items())},
                  "source": "CE sweep here (screen seed 0, final val acc; ties -> smaller lr)"}
    C.RECIPE_FILE.write_text(json.dumps(d, indent=1) + "\n")
    print(json.dumps(d[dataset]))


def pick_keeps():
    from .report import load_runs
    runs = load_runs(C.STAGE1)

    def acc(m, k):
        v = [r["val_acc_last"] for r in runs if r["tier"] == "screen" and r["method"] == m and float(r["keep"]) == k
             and r["seed"] == C.SCREEN_SEED]
        return v[-1] if v else None
    full = acc("full", 196.0)
    gaps = [(full - acc("maskedkd", k), k) for k in C.BASELINE_KEEPS
            if full is not None and acc("maskedkd", k) is not None]
    keeps = [k for _, k in sorted(gaps, reverse=True)[:2]] or [49, 29]
    print(",".join(f"{k:g}" for k in sorted(keeps, reverse=True)))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "jobs-from-bench":
        jobs_from_bench()
    elif cmd == "lr-pick":
        lr_pick(sys.argv[2])
    elif cmd == "pick-keeps":
        pick_keeps()
    else:
        raise SystemExit(f"unknown command {cmd}")
