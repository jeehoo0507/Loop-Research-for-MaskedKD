"""Decisions agent/bootstrap.sh makes by rule (so nobody has to type numbers into config.py).

    python -m harness.boot epochs-from-bench          # CIFAR screen/final epochs + MAX_JOBS from state/bench.json
    python -m harness.boot epochs-in100 <smoke run id> # ImageNet-100 final epochs from a smoke run's throughput
    python -m harness.boot pick-keeps                  # ImageNet-100 baseline budgets from the CIFAR baselines
    python -m harness.boot scale-epochs <f>            # CIFAR epochs = f x the bench rule (gate escalation)

Rules (agreed in planning):
  CIFAR screen = bench's "1 hour at the recommended parallelism" (at least 10 epochs); final = 2 x screen.
  ImageNet-100 final = as many epochs as fit in IN100_HOURS at the measured throughput (at least 10).
  pick-keeps = the two budgets where maskedkd beats random by the most on CIFAR screen seed 0.
  scale-epochs: when the CIFAR gate fails on seeds 0-2, bootstrap retries at 2x and then 4x the bench epochs
  (KD needs longer training than CE to pay off); runs at the old budget drop out of every table.
"""
import json
import sys

from . import config as C

IN100_HOURS = 12
IN100_TRAIN = 126689 - C.VAL_PER_CLASS * C.NUM_CLASSES


def _load():
    return json.loads(C.EPOCHS_FILE.read_text()) if C.EPOCHS_FILE.exists() else {"epochs": {}}


def _save(d):
    C.STATE.mkdir(exist_ok=True)
    C.EPOCHS_FILE.write_text(json.dumps(d, indent=1) + "\n")


def epochs_from_bench():
    rec = json.loads((C.STATE / "bench.json").read_text())["recommend"]
    screen = max(10, int(rec["screen_epochs_for_1h_at_that_parallelism"]))
    d = _load()
    d["epochs"]["cifar100"] = {"screen": screen, "final": 2 * screen}
    d["base_screen"], d["gate_scale"] = screen, 1
    d["max_jobs"] = int(rec["parallel_jobs"])
    d["rule"] = "screen = 1 h at bench parallelism (>=10), final = 2x screen"
    _save(d)
    print(json.dumps(d))


def epochs_in100(run_id):
    m = json.loads((C.RUNS / run_id / "metrics.json").read_text())
    ips = m["img_per_s_mean"]
    epochs = max(10, int(IN100_HOURS * 3600 / (IN100_TRAIN / ips)))
    d = _load()
    d["epochs"]["in100"] = {"final": epochs}
    d["in100_rule"] = f"{IN100_HOURS} h at {ips:.0f} img/s measured by smoke run {run_id} (>=10 epochs)"
    _save(d)
    print(json.dumps(d))


def scale_epochs(f):
    d = _load()
    base = d.get("base_screen") or d["epochs"]["cifar100"]["screen"]
    d["base_screen"], d["gate_scale"] = base, f
    d["epochs"]["cifar100"] = {"screen": base * f, "final": 2 * base * f}
    d["rule"] = f"screen = {f} x (1 h at bench parallelism, >=10) after the gate failed at shorter training; final = 2x screen"
    _save(d)
    print(json.dumps(d))


def pick_keeps():
    from .report import load_runs
    runs = load_runs("cifar100")

    def acc(m, k):
        v = [r["val_acc_last"] for r in runs if r["tier"] == "screen" and r["method"] == m and float(r["keep"]) == k
             and r["seed"] == C.SCREEN_SEED]
        return v[-1] if v else None
    gaps = [(acc("maskedkd", k) - acc("random", k), k) for k in C.BASELINE_KEEPS
            if acc("maskedkd", k) is not None and acc("random", k) is not None]
    keeps = [k for _, k in sorted(gaps, reverse=True)[:2]] or [49, 25]
    print(",".join(f"{k:g}" for k in sorted(keeps, reverse=True)))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "epochs-from-bench":
        epochs_from_bench()
    elif cmd == "epochs-in100":
        epochs_in100(sys.argv[2])
    elif cmd == "scale-epochs":
        scale_epochs(int(sys.argv[2]))
    elif cmd == "pick-keeps":
        pick_keeps()
    else:
        raise SystemExit(f"unknown command {cmd}")
