"""Runs queued jobs on the GPU. Started by a human in its own tmux session (see README).

    python -m harness.scheduler

Concurrency: at most config.MAX_JOBS, and a job starts only if free VRAM >= the largest peak VRAM
measured by harness.bench (or 8 GiB before a bench exists) + VRAM_MARGIN_GIB.
Refuses to start jobs while the harness lock does not match. Jobs whose process died (pod restart)
go back to pending and resume from their checkpoint.
"""
import json
import os
import subprocess
import sys
import time

from . import config as C
from .lock import check
from .submit import queue_dirs


def free_vram_gib():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                      text=True)
        return float(out.strip().splitlines()[0]) / 1024
    except Exception:
        return None


def need_vram_gib():
    b = C.STATE / "bench.json"
    if b.exists():
        return max(r["peak_vram_gib"] for r in json.loads(b.read_text())["single"].values())
    return 8.0


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def code_changed(since):
    """True when a harness file changed after this process started (a pull, then the lock adopted it)."""
    return any(f.stat().st_mtime > since for f in (C.ROOT / "harness").glob("*.py"))


def loop_end_date():
    """END_DATE default from agent/loop.sh (the loop's last day, exclusive)."""
    import re
    m = re.search(r'END_DATE="\$\{END_DATE:-([0-9-]+)\}"', (C.ROOT / "agent" / "loop.sh").read_text())
    return m.group(1) if m else None


def revive_loop():
    """The agent loop and the scheduler watch each other (loop.sh brings back a missing scheduler). A loop that
    exited on its own (an older END_DATE, a crash) is started again unless a human left agent/STOP."""
    try:
        end = loop_end_date()
        if (C.ROOT / "agent" / "STOP").exists() or end is None or time.strftime("%Y-%m-%d") >= end:
            return
        # only after bootstrap has started the research loop for the current setting (not during a switch/reset)
        st = (C.ROOT / "agent" / "BOOT_STATUS.md")
        lines = st.read_text().splitlines() if st.exists() else []
        started = max((i for i, l in enumerate(lines) if "research loop started" in l), default=-1)
        if started < 0 or started < max((i for i, l in enumerate(lines) if "SWITCH:" in l), default=-1):
            return
        if subprocess.run(["tmux", "has-session", "-t", "agent"], capture_output=True).returncode == 0:
            return
        aud = C.ROOT / "agent" / ".state" / "boot_auditor"
        env = {**os.environ, "AUDITOR": aud.read_text().strip() if aud.exists() else "claude"}
        subprocess.run(["bash", "agent/start.sh"], cwd=C.ROOT, env=env, capture_output=True, timeout=120)
        print("[sched] agent loop was not running; started it (agent/start.sh)", flush=True)
    except Exception as e:  # never let the watchdog stop the scheduler
        print(f"[sched] revive_loop failed: {e}", flush=True)


def main():
    t0 = time.time()
    last_revive = 0.0
    q = queue_dirs()
    procs = {}
    started = {}
    # recover after restart
    for f in q["running"].glob("*.json"):
        pidf = f.with_suffix(".pid")
        pid = int(pidf.read_text()) if pidf.exists() else -1
        if pid > 0 and alive(pid):
            procs[f.stem] = (None, pid)
        else:
            f.rename(q["pending"] / f.name)
            pidf.unlink(missing_ok=True)
            print(f"[sched] requeued {f.stem} (process gone)", flush=True)
    last_msg = ""
    while True:
        for jid, (p, pid) in list(procs.items()):
            rc = p.poll() if p is not None else (None if alive(pid) else -1)
            if rc is None:
                jj = json.loads((q["running"] / f"{jid}.json").read_text())
                limit = C.time_limit_h(jj.get("dataset", C.STAGE1), jj.get("tier", "screen"))
                if p is not None and time.time() - started.get(jid, time.time()) > limit * 3600 * 1.3:
                    p.kill()   # hard stop (train jobs normally stop themselves at the tier limit)
                    print(f"[sched] killed {jid}: wall-clock limit", flush=True)
                continue
            f = q["running"] / f"{jid}.json"
            j = json.loads(f.read_text())
            j["returncode"] = rc
            j["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            (q["done"] / f.name).write_text(json.dumps(j, indent=1))
            f.unlink()
            (q["running"] / f"{jid}.pid").unlink(missing_ok=True)
            del procs[jid]
            print(f"[sched] {jid} finished rc={rc}", flush=True)

        if time.time() - last_revive > 600:
            last_revive = time.time()
            revive_loop()
        ok, msg = check()
        if code_changed(t0):
            # a scheduler kept running on old code compares the new lock with its old rule and never starts a job
            # again (2026-10-31 20:15 UTC: 9 h idle). Re-exec loads the new code; running jobs are kept (recover).
            print("[sched] harness changed on disk, restarting on the new code", flush=True)
            os.execv(sys.executable, [sys.executable, "-m", "harness.scheduler"])
        pending = sorted(q["pending"].glob("*.json"),
                         key=lambda f: (-json.loads(f.read_text()).get("priority", 0), f.stat().st_mtime))
        if not ok:
            if msg != last_msg:
                print(f"[sched] LOCK {msg}; not starting jobs", flush=True)
                last_msg = msg
        elif pending and len(procs) < C.MAX_JOBS:
            free = free_vram_gib()
            need = need_vram_gib() + C.VRAM_MARGIN_GIB
            n2 = sum(json.loads((q["running"] / f"{j}.json").read_text()).get("dataset") == C.STAGE2 for j in procs)
            eligible = [f for f in pending
                        if json.loads(f.read_text()).get("dataset") != C.STAGE2 or n2 < C.STAGE2_MAX_RUNNING]
            if eligible and (free is None or free >= need):
                f = eligible[0]
                dst = q["running"] / f.name
                f.rename(dst)
                kind = json.loads(dst.read_text()).get("kind", "train")
                rdir = (C.ROOT / "analysis_out" if kind == "analysis" else C.RUNS) / f.stem
                rdir.mkdir(parents=True, exist_ok=True)
                out = open(rdir / "stdout.txt", "a")
                mod = "harness.analyze" if kind == "analysis" else "harness.train"
                p = subprocess.Popen([sys.executable, "-m", mod, "--job", str(dst)], cwd=C.ROOT,
                                     stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "2"})
                (q["running"] / f"{f.stem}.pid").write_text(str(p.pid))
                procs[f.stem] = (p, p.pid)
                started[f.stem] = time.time()
                print(f"[sched] started {f.stem} pid={p.pid} free={free} need={need:.1f}", flush=True)
                time.sleep(60)   # let it allocate before measuring free VRAM again
                continue
        time.sleep(20)


if __name__ == "__main__":
    main()
