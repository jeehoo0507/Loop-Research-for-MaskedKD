"""Queue the arms of the surgical study for one dataset (a human runs this; nothing here is automatic).

    python -m surgical.submit_grid --dataset waterbirds --seeds 1 2 3            # the 11 default arms at K=29
    python -m surgical.submit_grid --dataset cifar10 --seeds 1 --dry-run         # print what would be queued
    python -m surgical.submit_grid --dataset waterbirds --seeds 1 2 3 --extra    # the extra block sets as well
    python -m surgical.submit_grid --dataset celeba --seeds 1 2 3 --only "CE" "Full KD"   # the KD-gain gate first

Seeds 1-3 go to the final tier, any other seed to the screen tier (same epochs; the harness reserves 1-3 for final).
A run that already exists is reused, not repeated: its id is printed with "(exists)". "Exists" means finished here
with this server's teacher (or a CE run), or already waiting in the queue. A KD run in the repository that was made
with another teacher (the checkpoints are not in git, so a new server has its own) is run again.
"""
import argparse
import json

from harness import config as C
from harness import submit
from harness.report import load_runs

from .arms import EXTRA, K, REFERENCE, SELECT, teacher_sha, usable


def _same(j, method, k, seed, tier, sel_args, dataset):
    return (j["method"] == method and float(j["keep"]) == float(k) and j["seed"] == seed and j["tier"] == tier
            and json.dumps(j.get("sel_args") or {}, sort_keys=True) == json.dumps(sel_args, sort_keys=True)
            and j.get("dataset", C.STAGE1) == dataset)


def existing(method, k, seed, tier, sel_args, dataset, sha):
    """Id of a run to reuse: finished and comparable, or still in the queue. None = it has to be run."""
    epochs = C.TIER_EPOCHS[dataset][tier]
    for r in load_runs(dataset):
        if _same(r, method, k, seed, tier, sel_args, dataset) and r.get("epochs") == epochs and r.get("status") == "done" \
                and usable(r, sha):
            return r["id"]
    for j in submit.all_jobs():
        if j["_state"] in ("pending", "running") and _same(j, method, k, seed, tier, sel_args, dataset):
            return j["id"]
    return None


def plan(seeds, keep, extra, only):
    arms = {**REFERENCE, **SELECT, **(EXTRA if extra else {})}
    if only:
        missing = [a for a in only if a not in {**arms, **EXTRA}]
        if missing:
            raise SystemExit(f"unknown arm(s) {missing}; arms: {list({**arms, **EXTRA})}")
        arms = {a: {**arms, **EXTRA}[a] for a in only}
    for seed in seeds:
        tier = "final" if seed in C.FINAL_SEEDS else "screen"
        for name, (method, sel_args) in arms.items():
            k = 0 if method == "ce" else C.NUM_PATCHES if method == "full" else keep
            yield name, method, k, seed, tier, sel_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--keep", type=float, default=K)
    ap.add_argument("--extra", action="store_true", help="also the block sets in arms.EXTRA")
    ap.add_argument("--only", nargs="+", help="arm names to queue instead of the default grid")
    ap.add_argument("--priority", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sha = teacher_sha(C.TEACHER_SHARED.get(a.dataset, a.dataset))
    rows = list(plan(a.seeds, a.keep, a.extra, a.only))
    if sha is None and not a.dry_run and any(method != "ce" for _, method, *_ in rows):
        raise SystemExit(f"no teacher checkpoint for {a.dataset} on this server (state/*.pt is not in git): run\n"
                         f"  .venv/bin/python -m harness.finetune_teacher --dataset {C.TEACHER_SHARED.get(a.dataset, a.dataset)}\n"
                         "first, or queue only CE with --only CE")
    for name, method, k, seed, tier, sel_args in rows:
        have = existing(method, k, seed, tier, sel_args, a.dataset, sha)
        if a.dry_run:
            print(f"{a.dataset} {tier} seed {seed}: {name:22s} = {method} keep {k:g} {sel_args or ''}"
                  f"{'  (exists: ' + have + ')' if have else ''}")
            continue
        if have:
            print(f"{have} (exists)  # {name}")
            continue
        job = submit.make_job(method, k, seed, tier, sel_args, f"surgical: {name}", "human", a.priority, True, a.dataset)
        print(f"{job['id']}  # {name}")


if __name__ == "__main__":
    main()
