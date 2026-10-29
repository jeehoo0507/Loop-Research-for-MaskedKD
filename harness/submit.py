"""Queue a training job. The agent never launches training directly; harness/scheduler.py does.

    python -m harness.submit --method maskedkd --keep 49 --seed 0 --tier screen
    python -m harness.submit --method my_rollout --keep 25 --seed 0 --tier screen --sel-args '{"layers": 4}'
    python -m harness.submit --gate          # reproduction gate: ce, full@196, maskedkd@98 (seed 0, screen)
    python -m harness.submit --baselines     # human, once after the gate: noise seeds + maskedkd/random grid
    python -m harness.submit --analysis layer_agreement   # run analysis/layer_agreement.py on the GPU queue
    python -m harness.submit --status        # queue and run summary

--method: "ce", a builtin ("full", "maskedkd", "random"), or a file selectors/<name>.py.
The selector source is snapshotted into the job at submit time; editing the file later does not
change a queued job.
"""
import argparse
import hashlib
import json
import time

from . import config as C
from .baselines import BUILTIN
from .validate import SelectorRejected, analysis_source, check_source, selector_source


def queue_dirs():
    d = {k: C.QUEUE / k for k in ("pending", "running", "done")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def all_jobs():
    out = []
    for state, d in queue_dirs().items():
        for f in d.glob("*.json"):
            j = json.loads(f.read_text())
            j["_state"] = state
            out.append(j)
    return out


def make_job(method, keep, seed, tier, sel_args=None, note="", by="agent", priority=0, force=False):
    epochs = C.TIER_EPOCHS.get(tier)
    if epochs is None:
        raise SystemExit(f"tier '{tier}' has no epoch budget yet (a human sets it after harness.bench)")
    if tier == "final" and seed not in C.FINAL_SEEDS:
        raise SystemExit(f"final runs use fresh seeds {C.FINAL_SEEDS} only (screen seed {C.SCREEN_SEED} never enters a claim)")
    if method in C.ANALYSIS_ONLY and tier not in ("screen", "smoke"):
        raise SystemExit(f"'{method}' is an analysis-only reference; screen tier only")
    if method == "ce":
        keep = 0
    elif not (0 <= keep <= C.NUM_PATCHES):
        raise SystemExit(f"keep must be in [0, {C.NUM_PATCHES}]")
    if method == "full":
        keep = C.NUM_PATCHES
    src, sha = None, "builtin"
    if method not in BUILTIN and method not in ("ce",) + C.ANALYSIS_ONLY:
        try:
            src = selector_source(method)
            check_source(src, method)
        except SelectorRejected as e:
            raise SystemExit(f"REJECTED: {e}")
        sha = hashlib.sha256(src.encode()).hexdigest()
    sel_args = sel_args or {}
    key = (method, sha, float(keep), seed, tier, json.dumps(sel_args, sort_keys=True))
    for j in all_jobs():
        jk = (j["method"], j.get("selector_sha"), float(j["keep"]), j["seed"], j["tier"],
              json.dumps(j.get("sel_args", {}), sort_keys=True))
        if jk == key and not force and j.get("returncode", 0) == 0:
            raise SystemExit(f"duplicate of {j['id']} ({j['_state']}); use --force to rerun")
    ts = time.strftime("%m%d_%H%M%S")
    tag = hashlib.sha1(json.dumps(key).encode()).hexdigest()[:4]
    keep_s = f"{keep:g}"
    job = {"id": f"{tier}_{method}_k{keep_s}_s{seed}_{ts}_{tag}", "kind": "train", "method": method, "keep": keep, "seed": seed,
           "tier": tier, "epochs": epochs, "sel_args": sel_args, "selector_src": src, "selector_sha": sha,
           "note": note, "by": by, "priority": priority, "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = queue_dirs()["pending"] / f"{job['id']}.json"
    path.write_text(json.dumps(job, indent=1))
    return job


def make_analysis(name, note="", by="agent", priority=0):
    try:
        src = analysis_source(name)
        check_source(src, name, kind="analysis")
    except SelectorRejected as e:
        raise SystemExit(f"REJECTED: {e}")
    sha = hashlib.sha256(src.encode()).hexdigest()
    ts = time.strftime("%m%d_%H%M%S")
    job = {"id": f"analysis_{name}_{ts}_{sha[:4]}", "kind": "analysis", "method": name, "keep": 0, "seed": 0,
           "tier": "analysis", "epochs": 0, "selector_src": src, "selector_sha": sha, "note": note, "by": by,
           "priority": priority, "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}
    (queue_dirs()["pending"] / f"{job['id']}.json").write_text(json.dumps(job, indent=1))
    return job


def baselines(force=False):
    """Seed-noise runs (maskedkd@98, full; NOISE_SEEDS) and the maskedkd/random grid at BASELINE_KEEPS."""
    ids = []
    for s in C.NOISE_SEEDS:
        for m, k in (("maskedkd", 98), ("full", 196)):
            ids.append(make_job(m, k, s, "screen", by="baselines", priority=5, force=force)["id"])
    for k in C.BASELINE_KEEPS:
        for m in ("maskedkd", "random"):
            ids.append(make_job(m, k, C.SCREEN_SEED, "screen", by="baselines", priority=5, force=force)["id"])
    return ids


def status():
    jobs = all_jobs()
    for state in ("running", "pending", "done"):
        js = sorted((j for j in jobs if j["_state"] == state), key=lambda j: j["submitted"])
        print(f"== {state} ({len(js)})")
        for j in js[-30:]:
            m = C.RUNS / j["id"] / "metrics.json"
            extra = ""
            if m.exists():
                mm = json.loads(m.read_text())
                extra = f" val={mm['val_acc_last']} gflops={(mm['flops'] or {}).get('total_gflops_per_img')}"
            elif (C.RUNS / j["id"] / "error.txt").exists():
                extra = " ERROR: " + (C.RUNS / j["id"] / "error.txt").read_text().strip()[:200]
            print(f"  {j['id']} rc={j.get('returncode', '-')}{extra}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method")
    ap.add_argument("--keep", type=float, default=98)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tier", default="screen", choices=list(C.TIER_EPOCHS))
    ap.add_argument("--sel-args", default="{}")
    ap.add_argument("--note", default="")
    ap.add_argument("--by", default="agent")
    ap.add_argument("--priority", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--analysis")
    a = ap.parse_args()
    if a.status:
        status()
    elif a.baselines:
        print("\n".join(baselines(a.force)))
    elif a.analysis:
        print(make_analysis(a.analysis, a.note, a.by, a.priority)["id"])
    elif a.gate:
        for m, k in (("ce", 0), ("full", 196), ("maskedkd", 98)):
            print(make_job(m, k, 0, a.tier, by="gate", priority=10, force=a.force)["id"])
    else:
        if not a.method:
            ap.error("--method required")
        print(make_job(a.method, a.keep, a.seed, a.tier, json.loads(a.sel_args), a.note, a.by, a.priority,
                       a.force)["id"])
