"""Queue a training job. The agent never launches training directly; harness/scheduler.py does.

    python -m harness.submit --method maskedkd --keep 49 --seed 0 --tier screen
    python -m harness.submit --method my_rollout --keep 25 --seed 0 --tier screen --sel-args '{"layers": 4}'
    python -m harness.submit --gate          # reproduction gate: ce, full@196, maskedkd@98 (seed 0, screen)
    python -m harness.submit --baselines     # human, once after the gate: noise seeds + maskedkd/random grid
    python -m harness.submit --analysis layer_agreement   # run analysis/layer_agreement.py on the GPU queue
    python -m harness.submit --status        # queue and run summary

--method: "ce", a builtin ("full", "maskedkd", "random", "grid", "center", "tome"), or a file selectors/<name>.py.
An agent final run is queued ahead of screen runs and brings its comparisons with it on the same seed
(maskedkd, tome, grid, center, random at the same K, full KD, and META['parent'] at the same K).
The selector source is snapshotted into the job at submit time; editing the file later does not
change a queued job.
"""
import argparse
import hashlib
import json
import sys
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


def audit_passed(method):
    """True if AUDIT.md has a PASS verdict in an audit block whose request names this method."""
    audit = C.ROOT / "AUDIT.md"
    if not audit.exists():
        return False
    for block in audit.read_text().split("\n## ")[1:]:
        if method in block and "VERDICT: PASS" in block:
            return True
    return False


def stage2_rule(method, sha):
    """Stage 2 (ImageNet-100) admits harness baselines, and agent methods that finished CIFAR final on
    all fresh seeds and passed audit, only after the ImageNet-100 gate passed. Returns a label."""
    if method in C.BASELINES:
        return "baseline"
    from .report import gate, load_runs
    runs = load_runs()
    ok, msg = gate(runs, "final", C.STAGE2, seed=C.FINAL_SEEDS[0])
    if not ok:
        raise SystemExit(f"stage 2 is closed until the ImageNet-100 gate passes: {msg}")
    cifar = [r for r in runs if r.get("dataset", "cifar100") == "cifar100" and r["tier"] == "final" and r["method"] == method]
    passed_shas = {r["selector_sha"] for r in cifar
                   if set(C.FINAL_SEEDS) <= {x["seed"] for x in cifar if x["selector_sha"] == r["selector_sha"]}}
    if not passed_shas or not audit_passed(method):
        raise SystemExit(f"'{method}' needs CIFAR final on seeds {C.FINAL_SEEDS} and an audit PASS before stage 2")
    if sha in passed_shas:
        return "stage2_untouched"
    if not any(r["method"] == method and r.get("stage2") == "stage2_untouched" for r in runs):
        raise SystemExit(f"run the unchanged '{method}' on stage 2 first; tuned variants come after")
    return "stage2_tuned"


def make_job(method, keep, seed, tier, sel_args=None, note="", by="agent", priority=0, force=False, dataset="cifar100"):
    if dataset not in C.DATASETS:
        raise SystemExit(f"dataset must be one of {C.DATASETS}")
    epochs = C.TIER_EPOCHS[dataset].get(tier)
    if epochs is None:
        raise SystemExit(f"{dataset} tier '{tier}' has no epoch budget (a human sets it after harness.bench)")
    if tier == "final" and seed not in C.FINAL_SEEDS:
        raise SystemExit(f"final runs use fresh seeds {C.FINAL_SEEDS} only (screen seed {C.SCREEN_SEED} never enters a claim)")
    if method in C.ANALYSIS_ONLY and tier == "final" and dataset == "cifar100":
        raise SystemExit(f"'{method}' is an analysis-only reference; screen tier only on cifar100")
    if method == "ce":
        keep = 0
    elif not (0 <= keep <= C.NUM_PATCHES):
        raise SystemExit(f"keep must be in [0, {C.NUM_PATCHES}]")
    if method == "full":
        keep = C.NUM_PATCHES
    src, sha = None, "builtin"
    if method not in C.BASELINES:
        try:
            src = selector_source(method)
            meta = check_source(src, method)
        except SelectorRejected as e:
            raise SystemExit(f"REJECTED: {e}")
        parent = meta["parent"].strip()
        if parent != "none" and parent not in BUILTIN and not (C.ROOT / "selectors" / f"{parent}.py").exists():
            raise SystemExit(f"REJECTED: META['parent'] = '{parent}' is neither 'none', a builtin, nor selectors/{parent}.py")
        sha = hashlib.sha256(src.encode()).hexdigest()
        if tier == "final" and by == "agent":
            priority = max(priority, C.FINAL_PRIORITY)
    stage2 = None
    if dataset == C.STAGE2 and tier != "smoke":
        stage2 = stage2_rule(method, sha)
        priority = min(priority, C.STAGE2_PRIORITY)
    sel_args = sel_args or {}
    key = (method, sha, float(keep), seed, tier, json.dumps(sel_args, sort_keys=True), dataset, epochs)
    for j in all_jobs():
        jk = (j["method"], j.get("selector_sha"), float(j["keep"]), j["seed"], j["tier"],
              json.dumps(j.get("sel_args", {}), sort_keys=True), j.get("dataset", "cifar100"), j.get("epochs"))
        if jk == key and not force and j.get("returncode", 0) == 0:
            raise SystemExit(f"duplicate of {j['id']} ({j['_state']}); use --force to rerun")
    ts = time.strftime("%m%d_%H%M%S")
    tag = hashlib.sha1(json.dumps(key).encode()).hexdigest()[:4]
    keep_s = f"{keep:g}"
    ds = "" if dataset == "cifar100" else f"{dataset}_"
    job = {"id": f"{ds}{tier}_{method}_k{keep_s}_s{seed}_{ts}_{tag}", "kind": "train", "dataset": dataset,
           "stage2": stage2, "method": method, "keep": keep, "seed": seed,
           "tier": tier, "epochs": epochs, "sel_args": sel_args, "selector_src": src, "selector_sha": sha,
           "note": note, "by": by, "priority": priority, "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = queue_dirs()["pending"] / f"{job['id']}.json"
    path.write_text(json.dumps(job, indent=1))
    return job


def final_refs(method, keep, seed, dataset="cifar100"):
    """With an agent final run, queue the comparisons it needs on the same seed (skips ones that exist):
    the claim references at the same budget, full KD, and the selector's parent at the same budget."""
    # stage 2 has one long GPU slot: only maskedkd and tome there
    refs = [(m, keep) for m in C.CLAIM_REFS if dataset != C.STAGE2 or m in ("maskedkd", "tome")] + [("full", 196)]
    src = selector_source(method)
    parent = check_source(src, method)["parent"].strip() if src else "none"
    if parent != "none":
        refs.append((parent, keep))
    ids = []
    for m, k in refs:
        try:
            ids.append(make_job(m, k, seed, "final", by="final_refs", priority=C.FINAL_PRIORITY, dataset=dataset)["id"])
        except SystemExit as e:   # duplicate (already queued or done) or not admissible
            ids.append(f"skip {m}@{k:g}: {e}")
    return ids


def make_analysis(name, note="", by="agent", priority=0, dataset="cifar100"):
    try:
        src = analysis_source(name)
        check_source(src, name, kind="analysis")
    except SelectorRejected as e:
        raise SystemExit(f"REJECTED: {e}")
    sha = hashlib.sha256(src.encode()).hexdigest()
    ts = time.strftime("%m%d_%H%M%S")
    job = {"id": f"analysis_{name}_{ts}_{sha[:4]}", "kind": "analysis", "dataset": dataset, "method": name, "keep": 0, "seed": 0,
           "tier": "analysis", "epochs": 0, "selector_src": src, "selector_sha": sha, "note": note, "by": by,
           "priority": priority, "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}
    (queue_dirs()["pending"] / f"{job['id']}.json").write_text(json.dumps(job, indent=1))
    return job


GATE = (("ce", 0), ("full", 196), ("maskedkd", 98))


def gate_jobs(dataset="cifar100", force=False, seeds=None):
    """CIFAR: screen, seed 0 unless `seeds` (bootstrap adds 1, 2 when one seed misses). ImageNet-100: final seed 1.
    Runs that already exist at the current epochs are reused (their ids are returned)."""
    tier = "screen" if dataset == "cifar100" else "final"
    seeds = seeds or ([C.SCREEN_SEED] if dataset == "cifar100" else [C.FINAL_SEEDS[0]])
    return [job_or_existing(m, k, s, tier, by="gate", priority=10, force=force, dataset=dataset)
            for s in seeds for m, k in GATE]


def job_or_existing(method, keep, seed, tier, **kw):
    """make_job, but a duplicate at the current epochs returns the existing job id instead of failing."""
    try:
        return make_job(method, keep, seed, tier, **kw)["id"]
    except SystemExit as e:
        if str(e).startswith("duplicate of "):
            return str(e).split()[2]
        raise


def baselines(dataset="cifar100", force=False, keeps=None):
    """CIFAR: seed-noise runs (maskedkd@98, full on NOISE_SEEDS) + maskedkd/random/grid/center/tome at BASELINE_KEEPS.
    ImageNet-100: gate trio on the remaining final seeds + maskedkd/random/tome at `keeps` (seed 1) + oracle."""
    keeps = tuple(keeps or C.BASELINE_KEEPS)
    ids = []
    if dataset == "cifar100":
        for s in C.NOISE_SEEDS:   # may already exist from the gate's extra seeds
            for m, k in (("maskedkd", 98), ("full", 196)):
                ids.append(job_or_existing(m, k, s, "screen", by="baselines", priority=5, force=force))
        for k in keeps:
            for m in ("maskedkd", "random", "grid", "center", "tome"):
                ids.append(make_job(m, k, C.SCREEN_SEED, "screen", by="baselines", priority=5, force=force)["id"])
        ids.append(make_job("oracle", keeps[-1], C.SCREEN_SEED, "screen", by="baselines", priority=5, force=force)["id"])
        return ids
    for s in C.FINAL_SEEDS[1:]:
        for m, k in GATE:
            ids.append(make_job(m, k, s, "final", by="baselines", force=force, dataset=dataset)["id"])
    for k in keeps:
        for m in ("maskedkd", "random", "tome"):
            ids.append(make_job(m, k, C.FINAL_SEEDS[0], "final", by="baselines", force=force, dataset=dataset)["id"])
    ids.append(make_job("oracle", keeps[-1], C.FINAL_SEEDS[0], "final", by="baselines", force=force, dataset=dataset)["id"])
    return ids


def wait(ids, poll=30):
    """Block until every id is in queue/done. Returns 0 if all exited 0."""
    d = queue_dirs()["done"]
    while True:
        done = {i: json.loads((d / f"{i}.json").read_text()) for i in ids if (d / f"{i}.json").exists()}
        if len(done) == len(ids):
            bad = [i for i, j in done.items() if j.get("returncode") != 0]
            for i in bad:
                err = C.RUNS / i / "error.txt"
                print(f"FAILED {i}: {err.read_text().strip()[:300] if err.exists() else 'rc=' + str(done[i].get('returncode'))}")
            return 1 if bad else 0
        time.sleep(poll)


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
    ap.add_argument("--tier", default="screen", choices=["smoke", "screen", "final"])
    ap.add_argument("--dataset", default="cifar100", choices=list(C.DATASETS))
    ap.add_argument("--keeps", default="", help="comma list of budgets for --baselines on in100")
    ap.add_argument("--seeds", default="", help="comma list of seeds for --gate (cifar100)")
    ap.add_argument("--smoke", action="store_true", help="human: a few-minute GPU check of every path")
    ap.add_argument("--sel-args", default="{}")
    ap.add_argument("--note", default="")
    ap.add_argument("--by", default="agent")
    ap.add_argument("--priority", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--analysis")
    ap.add_argument("--wait", nargs="+", help="block until these job ids are done; exit 1 if any failed")
    a = ap.parse_args()
    if a.wait:
        sys.exit(wait(a.wait))
    elif a.status:
        status()
    elif a.smoke:
        for m, k in (("ce", 0), ("full", 196), ("maskedkd", 49), ("random", 49), ("grid", 49), ("center", 49),
                     ("tome", 49), ("oracle", 49)):
            print(make_job(m, k, 0, "smoke", by="smoke", priority=20, force=True, dataset=a.dataset)["id"])
    elif a.baselines:
        keeps = [float(k) for k in a.keeps.split(",") if k] or None
        print("\n".join(baselines(a.dataset, a.force, keeps)))
    elif a.analysis:
        print(make_analysis(a.analysis, a.note, a.by, a.priority, a.dataset)["id"])
    elif a.gate:
        print("\n".join(gate_jobs(a.dataset, a.force, [int(s) for s in a.seeds.split(",") if s] or None)))
    else:
        if not a.method:
            ap.error("--method required")
        job = make_job(a.method, a.keep, a.seed, a.tier, json.loads(a.sel_args), a.note, a.by, a.priority,
                       a.force, a.dataset)
        print(job["id"])
        if a.tier == "final" and a.method not in C.BASELINES:
            print("\n".join(final_refs(a.method, a.keep, a.seed, a.dataset)))
