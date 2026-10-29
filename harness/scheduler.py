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


def main():
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
                tier = json.loads((q["running"] / f"{jid}.json").read_text()).get("tier", "screen")
                if p is not None and time.time() - started.get(jid, time.time()) > C.TIER_TIME_LIMIT_H[tier] * 3600 * 1.3:
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

        ok, msg = check()
        pending = sorted(q["pending"].glob("*.json"),
                         key=lambda f: (-json.loads(f.read_text()).get("priority", 0), f.stat().st_mtime))
        if not ok:
            if msg != last_msg:
                print(f"[sched] LOCK {msg}; not starting jobs", flush=True)
                last_msg = msg
        elif pending and len(procs) < C.MAX_JOBS:
            free = free_vram_gib()
            need = need_vram_gib() + C.VRAM_MARGIN_GIB
            if free is None or free >= need:
                f = pending[0]
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
