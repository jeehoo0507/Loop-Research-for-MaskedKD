"""The whole surgical study, unattended, from the smoke test to the results table (a human starts it once).

    tmux new -d -s surgical '.venv/bin/python -m surgical.autorun 2>&1 | tee -a surgical/autorun.log'
    (bash surgical/setup_server.sh --go does exactly this)

    .venv/bin/python -m surgical.autorun --datasets waterbirds cifar10 --skip-smoke   # other datasets, no smoke

Order, so the GPU is never idle and nothing is decided after seeing a result:
1. smoke    surgical.smoke on the datasets. A dataset whose smoke part failed is dropped (said in the log); the
            rest go on. If every dataset failed, it stops.
2. teachers every dataset without a teacher on this server is fine-tuned, one after another, with job starts held
            (agent/.state/sched_hold) and no job running; then the lock is renewed (state/recipe.json is locked).
3. queue    for every dataset at once: the KD-gain gate (CE and Full KD, seeds 1 2 3, priority 20) and the 11 arms
            at seed 1 (priority 10). Runs already finished with this server's teacher are reused.
4. watch    every 5 minutes. When a dataset's gate runs are done: the gate is read with the rule fixed in the plan
            (Full KD - CE: mean > 0 and at least 2 of 3 seeds > 0), the offline layer sweep is queued, and if the gate
            held, the 11 arms at seeds 2 and 3 (priority 5). If it did not hold, the dataset keeps its seed-1 grid as
            an observation and gets no more seeds.
5. end      when nothing is pending or running: surgical_results.md (surgical.table) and a last status line.
State: surgical/autorun_state.json (safe to restart: decided gates are not read again). Status: surgical/STATUS.md.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from harness import config as C
from harness import submit
from harness.report import paired

from .arms import teacher_sha
from .submit_grid import existing, plan

DEFAULT = ("waterbirds", "cifar10", "cifar10c_gaussian_noise", "celeba")
STATE = C.ROOT / "surgical" / "autorun_state.json"
STATUS = C.ROOT / "surgical" / "STATUS.md"
HOLD = C.ROOT / "agent" / ".state" / "sched_hold"
PY = sys.executable


def say(msg):
    line = f"[autorun {time.strftime('%m-%d %H:%M')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} {msg}\n")


def sh(*cmd, check=True):
    r = subprocess.run(list(cmd), cwd=C.ROOT)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited {r.returncode}")
    return r.returncode


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"gate": {}, "dropped": {}}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1))


def scheduler_up():
    if subprocess.run(["tmux", "has-session", "-t", "sched"], capture_output=True).returncode != 0:
        sh("bash", "agent/start.sh", "sched")
        say("GPU scheduler was not running: started it")


def running():
    return list(submit.queue_dirs()["running"].glob("*.json"))


def queue(dataset, seeds, priority, only=None):
    """The arms of `plan` that are neither finished here nor queued. Returns the new job ids."""
    sha = teacher_sha(C.TEACHER_SHARED.get(dataset, dataset))
    new = []
    for name, method, k, seed, tier, sel_args in plan(seeds, 29, False, only):
        if existing(method, k, seed, tier, sel_args, dataset, sha):
            continue
        j = submit.make_job(method, k, seed, tier, sel_args, f"surgical: {name}", "human", priority, True, dataset)
        new.append(j["id"])
    return new


# ---------------------------------------------------------------- steps

def smoke(datasets):
    rc = sh(PY, "-m", "surgical.smoke", "--datasets", *datasets, check=False)
    rep = json.loads((C.ROOT / "surgical_smoke_report.json").read_text())
    ok, bad = [], {}
    for d in datasets:
        errs = [f"{p}: {rep.get(d, {}).get(p, {}).get('error')}" for p in ("data", "speed", "e2e")
                if "error" in rep.get(d, {}).get(p, {})]
        (bad.__setitem__(d, "; ".join(errs)) if errs else ok.append(d))
    say(f"smoke exit {rc}: ok {ok}" + (f", dropped {bad}" if bad else ""))
    return ok, bad


def teachers(datasets):
    need = sorted({C.TEACHER_SHARED.get(d, d) for d in datasets} - {t for t in
                  {C.TEACHER_SHARED.get(d, d) for d in datasets} if teacher_sha(t)})
    if not need:
        return []
    failed = []
    HOLD.parent.mkdir(parents=True, exist_ok=True)
    HOLD.touch()
    try:
        while running():
            say(f"teachers {need}: waiting for {len(running())} running job(s) to finish")
            time.sleep(120)
        for t in need:
            say(f"teacher {t}: fine-tuning (whole GPU)")
            log = open(C.ROOT / "surgical" / f"teacher_{t}.log", "a")
            r = subprocess.run([PY, "-m", "harness.finetune_teacher", "--dataset", t], cwd=C.ROOT, stdout=log,
                               stderr=subprocess.STDOUT)
            if r.returncode != 0 or not teacher_sha(t):
                failed.append(t)
                say(f"teacher {t}: FAILED (surgical/teacher_{t}.log); its datasets are dropped")
                continue
            meta = json.loads((C.STATE / ("teacher.json" if t == "waterbirds" else f"teacher_{t}.json")).read_text())
            say(f"teacher {t}: val acc {meta['val_acc']}")
        # recipe.json (locked) gained the new datasets' student lr. Tests on the CPU: the GPU may hold jobs (OOM)
        if subprocess.run(["bash", "agent/relock.sh"], cwd=C.ROOT, env={**__import__("os").environ,
                          "CUDA_VISIBLE_DEVICES": ""}).returncode != 0:
            raise RuntimeError("bash agent/relock.sh failed")
        say("lock renewed after the teacher fine-tunes")
    finally:
        HOLD.unlink(missing_ok=True)
    return failed


def gate_verdict(dataset):
    """None while the gate runs are not all done, else (held, text). Rule fixed in the plan, section 0.3."""
    from .table import arms_by_seed
    arms = arms_by_seed(dataset, 29)
    ce, full = arms.get("CE", {}), arms.get("Full KD", {})
    seeds = [s for s in C.FINAL_SEEDS if s in ce and s in full]
    if len(seeds) < len(C.FINAL_SEEDS):
        return None
    diffs = [full[s]["val_acc_last"] - ce[s]["val_acc_last"] for s in seeds]
    held = sum(diffs) / len(diffs) > 0 and sum(d > 0 for d in diffs) >= 2
    p = paired(full, ce)
    return held, (f"Full KD - CE per seed {[round(d, 2) for d in diffs]}, mean {p['diff']:+.2f} "
                  f"[{p['lo']:+.2f}, {p['hi']:+.2f}]")


def watch(datasets, state):
    while True:
        scheduler_up()
        for d in datasets:
            if d in state["gate"]:
                continue
            v = gate_verdict(d)
            if v is None:
                continue
            held, text = v
            state["gate"][d] = {"held": held, "text": text}
            save_state(state)
            a = submit.make_analysis("surgical_layer_sweep", f"surgical: offline sweep {d}", "human", 15, d)
            say(f"{d} gate {'HELD' if held else 'NOT HELD'}: {text}; offline sweep queued ({a['id']})")
            if held:
                new = queue(d, [2, 3], 5)
                say(f"{d}: seeds 2 3 queued ({len(new)} new runs)")
            else:
                say(f"{d}: no more seeds; its seed-1 grid stays an observation, not a training claim")
        q = submit.queue_dirs()
        busy = len(list(q["pending"].glob("*.json"))) + len(running())
        decided = all(d in state["gate"] for d in datasets)
        if busy == 0 and not decided:   # a gate run failed: nothing will come; say so instead of waiting forever
            for d in datasets:
                if d not in state["gate"]:
                    state["gate"][d] = {"held": None, "text": "gate incomplete: a CE or Full KD run failed (see runs/*/error.txt)"}
                    say(f"{d}: {state['gate'][d]['text']}")
            save_state(state)
            decided = True
        if decided and busy == 0:
            return
        time.sleep(300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT), choices=list(C.DATASETS))
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--retry-dropped", action="store_true", help="give datasets dropped earlier another try")
    a = ap.parse_args()
    state = load_state()
    if a.retry_dropped:
        state["dropped"] = {}
        save_state(state)
    say(f"start: {a.datasets}")
    datasets = [d for d in a.datasets if d not in state["dropped"]]
    if not a.skip_smoke and not state.get("smoke_done"):
        datasets, bad = smoke(datasets)
        state["dropped"].update(bad)
        state["smoke_done"] = True
        save_state(state)
    for d in list(datasets):   # also when the smoke was skipped: a teacher fine-tune needs the data on disk
        if not (C.DATA_ROOT / d / "meta.json").exists():
            say(f"{d}: preparing data")
            if subprocess.run([PY, "-m", "harness.prepare_data", "--datasets", d], cwd=C.ROOT).returncode != 0:
                datasets.remove(d)
                state["dropped"][d] = "data preparation failed (see surgical/autorun.log)"
                say(f"{d}: data preparation FAILED; dropped")
    save_state(state)
    if not datasets:
        say("no dataset left (surgical_smoke_report.md, surgical/autorun.log): stopping")
        return 1
    scheduler_up()
    for t in teachers(datasets):
        for d in [d for d in datasets if C.TEACHER_SHARED.get(d, d) == t]:
            datasets.remove(d)
            state["dropped"][d] = f"teacher {t} failed"
    save_state(state)
    for d in datasets:
        if d in state["gate"]:
            continue
        g = queue(d, list(C.FINAL_SEEDS), 20, only=["CE", "Full KD"])
        s1 = queue(d, [1], 10)
        say(f"{d}: gate {len(g)} + seed-1 grid {len(s1)} new runs queued")
    watch(datasets, state)
    out = subprocess.run([PY, "-m", "surgical.table", "--datasets", *datasets], cwd=C.ROOT, capture_output=True, text=True)
    (C.ROOT / "surgical_results.md").write_text(out.stdout)
    say(f"done: surgical_results.md; gates {json.dumps({d: g['held'] for d, g in state['gate'].items()})}; "
        f"dropped {state['dropped'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
