"""Decisions agent/bootstrap.sh makes by rule (so nobody has to type numbers into config.py).

    python -m harness.boot jobs-from-bench   # MAX_JOBS from state/bench.json (epochs are fixed by the recipe: 100)
    python -m harness.boot pick-keeps        # stage-2 baseline budgets from the stage-1 baselines

Rules (agreed in planning):
  jobs-from-bench: the parallelism bench found best (aggregate img/s), written to state/epochs.json (locked).
  pick-keeps: the two budgets where MaskedKD falls furthest below Full KD on stage-1 screen seed 0 (accuracy),
  i.e. where token choice matters; default 49, 29.
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
    elif cmd == "pick-keeps":
        pick_keeps()
    else:
        raise SystemExit(f"unknown command {cmd}")
