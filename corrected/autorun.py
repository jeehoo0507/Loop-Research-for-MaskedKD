"""The corrected-distillation study, unattended (a human starts it once; see corrected/README.md).

    tmux new -d -s corrected '.venv/bin/python -m corrected.autorun 2>&1 | tee -a corrected/autorun.log'
    .venv/bin/python -m corrected.autorun --datasets cub_lt          # one more dataset, later (safe to run again)
    .venv/bin/python -m corrected.autorun --dry-run                  # what would be queued, nothing queued

Per dataset, in this order, so that no prediction is written after a result it is about:
1. base     CE and Full KD (seeds 1 2 3) and MaskedKD at K=29 (seed 1), unless they are already here with this
            server's teacher. A dataset without data or teacher is set up only after every other dataset is done
            (the teacher fine-tune needs the whole GPU), with `--no-setup` not at all.
2. dissect  analysis/masked_teacher_dissection.py on the seed-1 students (no training, minutes). Its "design" block
            says, per budget, which of the two designs should win: corrected/PREDICTIONS.md, with the time.
3. grid     the arms of corrected/arms.py: budget 29 on --seeds (seed 1 first), budget 49 on --seeds-more.
End: corrected_results.md (corrected.table). State: corrected/autorun_state.json. Status: corrected/STATUS.md.
"""
import argparse
import json
import subprocess
import sys
import time

from harness import config as C
from harness import submit
from harness.report import load_runs
from surgical import autorun as S
from surgical.arms import teacher_sha, usable
from surgical.submit_grid import existing

from .arms import BUDGETS, REFERENCE, arms

DEFAULT = ("waterbirds", "cifar10c_gaussian_noise", "celeba", "cifar10")
STATE = C.ROOT / "corrected" / "autorun_state.json"
STATUS = C.ROOT / "corrected" / "STATUS.md"
PREDICTIONS = C.ROOT / "corrected" / "PREDICTIONS.md"
ANALYSIS = "masked_teacher_dissection"
PY = sys.executable
S.STATUS = STATUS   # surgical.autorun's helpers (teachers, scheduler_up) report here too
say = S.say


def tier_of(seed):
    return "final" if seed in C.FINAL_SEEDS else "screen"


def ensure(dataset, method, keep, seed, sel_args, note, priority, dry):
    """Id of the run of this arm: one that exists (finished with this server's teacher, or queued), else a new job."""
    sha = teacher_sha(C.TEACHER_SHARED.get(dataset, dataset))
    have = existing(method, keep, seed, tier_of(seed), sel_args, dataset, sha)
    if have or dry:
        return have, False
    return submit.make_job(method, keep, seed, tier_of(seed), sel_args, f"corrected: {note}", "human", priority, True,
                           dataset)["id"], True


def queue_base(dataset, seeds, dry=False):
    new = 0
    rows = [(name, m, k, a, seed) for seed in seeds for name, (m, k, a) in REFERENCE.items()]
    rows.append(("MaskedKD (for the dissection)", "maskedkd", 29, {}, seeds[0]))
    for name, method, keep, args, seed in rows:
        got, made = ensure(dataset, method, keep, seed, args, name, 16, dry)
        new += made
        if dry:
            print(f"{dataset} seed {seed} base: {name:32s} = {method} keep {keep:g}{'  (exists: ' + got + ')' if got else '  NEW'}")
    return new


def queue_grid(dataset, seeds, seeds_more, dry=False):
    new = 0
    for budget, ss in zip(BUDGETS, (seeds, seeds_more)):
        for seed in ss:
            first = seed == ss[0]
            for name, (method, keep, args) in arms(budget).items():
                prio = (12 if first else 6) - (budget != BUDGETS[0])
                got, made = ensure(dataset, method, keep, seed, args, f"{name} @{budget}", prio, dry)
                new += made
                if dry:
                    print(f"{dataset} seed {seed} budget {budget}: {name:32s} = {method} keep {keep:g} {args or ''}"
                          f"{'  (exists: ' + got + ')' if got else '  NEW'}")
    return new


def students_ready(dataset, seed):
    """True when the seed's CE, Full KD and MaskedKD@29 runs are finished here with their weights (the dissection
    reads them)."""
    sha = teacher_sha(C.TEACHER_SHARED.get(dataset, dataset))
    need = {("ce", 0.0), ("full", float(C.NUM_PATCHES)), ("maskedkd", 29.0)}
    for r in load_runs(dataset):
        if r["seed"] == seed and r["tier"] == tier_of(seed) and usable(r, sha) and not r.get("sel_args") \
                and (C.RUNS / r["id"] / "student.pt").exists():
            need.discard((r["method"], float(r["keep"])))
    return not need


def dissection(job_id):
    """None while the analysis job runs, else its summary (a dict with "error" if it failed)."""
    out = C.ROOT / "analysis_out" / job_id
    if (out / "summary.json").exists() and (out / "meta.json").exists():
        return json.loads((out / "summary.json").read_text())
    if (out / "meta.json").exists() or (submit.queue_dirs()["done"] / f"{job_id}.json").exists():
        err = out / "error.txt"
        return {"error": err.read_text().strip()[-300:] if err.exists() else "no summary written"}
    return None


def write_prediction(dataset, summary):
    from .table import prediction_lines
    lines = prediction_lines(dataset, summary)
    with open(PREDICTIONS, "a") as f:
        f.write(f"\n## {dataset}, written {time.strftime('%Y-%m-%d %H:%M %Z')}, before any run of the grid was queued "
                "for this dataset\n\n" + "\n".join(lines) + "\n")
    say(f"{dataset}: prediction written (corrected/PREDICTIONS.md): " + "; ".join(l.lstrip("- ") for l in lines[:2]))


def advance(dataset, st, a):
    """One dataset one stage further if it can go. Stages: base -> dissect -> grid."""
    seed = a.seeds[0]
    if st["stage"] == "base" and students_ready(dataset, seed):
        job = submit.make_analysis(ANALYSIS, f"corrected: dissection {dataset}", "human", 30, dataset)
        st.update(stage="dissect", analysis=job["id"])
        say(f"{dataset}: base students are here; dissection queued ({job['id']})")
    elif st["stage"] == "dissect":
        s = dissection(st["analysis"])
        if s is None:
            return
        if "error" in s or not s.get("design"):
            say(f"{dataset}: the dissection gave no prediction ({s.get('error', 'no design block')}); the grid runs anyway")
            with open(PREDICTIONS, "a") as f:
                f.write(f"\n## {dataset}, {time.strftime('%Y-%m-%d %H:%M %Z')}\n\n- no prediction: the dissection failed\n")
        else:
            write_prediction(dataset, s)
        n = queue_grid(dataset, a.seeds, a.seeds_more)
        st["stage"] = "grid"
        say(f"{dataset}: grid queued ({n} new runs)")


def watch(datasets, state, a):
    while True:
        S.scheduler_up()
        for d in datasets:
            advance(d, state["datasets"][d], a)
        STATE.write_text(json.dumps(state, indent=1))
        q = submit.queue_dirs()
        busy = len(list(q["pending"].glob("*.json"))) + len(S.running())
        if busy == 0:
            stuck = [d for d in datasets if state["datasets"][d]["stage"] != "grid"]
            for d in stuck:   # a base run failed: its students will not come
                say(f"{d}: stopped at stage '{state['datasets'][d]['stage']}' with nothing running (see runs/*/error.txt)")
            return
        time.sleep(300)


def ready(dataset):
    t = C.TEACHER_SHARED.get(dataset, dataset)
    return (C.DATA_ROOT / dataset / "meta.json").exists() and teacher_sha(t) is not None


def set_up(datasets):
    """Data and teachers for datasets that have none here. Returns the ones that are ready afterwards."""
    for d in sorted({x for d in datasets for x in (d, C.TEACHER_SHARED.get(d, d))}):
        if not (C.DATA_ROOT / d / "meta.json").exists():
            say(f"{d}: preparing data")
            if subprocess.run([PY, "-m", "harness.prepare_data", "--datasets", d], cwd=C.ROOT).returncode != 0:
                say(f"{d}: data preparation FAILED (corrected/autorun.log)")
    have = [d for d in datasets if all((C.DATA_ROOT / x / "meta.json").exists() for x in (d, C.TEACHER_SHARED.get(d, d)))]
    failed = S.teachers(have) if have else []
    return [d for d in have if C.TEACHER_SHARED.get(d, d) not in failed and ready(d)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT), choices=list(C.DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.FINAL_SEEDS), help="budget 29")
    ap.add_argument("--seeds-more", type=int, nargs="+", default=[C.FINAL_SEEDS[0]], help="budget 49")
    ap.add_argument("--no-setup", action="store_true", help="leave out datasets without data or teacher here")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if C.BATCH != 128:
        raise SystemExit("corrected/arms.py splits the budgets for a batch of 128")
    now = [d for d in a.datasets if ready(d)]
    later = [d for d in a.datasets if d not in now]
    if a.dry_run:
        for d in now:
            queue_base(d, a.seeds, dry=True)
            queue_grid(d, a.seeds, a.seeds_more, dry=True)
            print(f"{d}: students for the dissection here: {students_ready(d, a.seeds[0])}")
        print(f"not set up here (data or teacher missing): {later or 'none'}")
        return 0
    state = json.loads(STATE.read_text()) if STATE.exists() else {"datasets": {}}
    say(f"start: {a.datasets}; ready now {now}; to set up after them {[] if a.no_setup else later}")
    S.scheduler_up()
    for label, group in (("now", now), ("later", [] if a.no_setup else later)):
        if not group:
            continue
        if label == "later":   # only now: the teacher fine-tune holds the queue and takes the whole GPU
            group = set_up(group)
            say(f"set up: {group}; left out {[d for d in later if d not in group] or 'none'}")
        for d in group:
            st = state["datasets"].setdefault(d, {"stage": "base"})
            if st["stage"] == "base":
                say(f"{d}: {queue_base(d, a.seeds)} new base runs queued")
        STATE.write_text(json.dumps(state, indent=1))
        watch(group, state, a)
    done = [d for d in a.datasets if state["datasets"].get(d, {}).get("stage") == "grid"]
    out = subprocess.run([PY, "-m", "corrected.table", "--datasets", *done], cwd=C.ROOT, capture_output=True, text=True)
    (C.ROOT / "corrected_results.md").write_text(out.stdout)
    say(f"done: corrected_results.md for {done}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
