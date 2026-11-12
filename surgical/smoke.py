"""Server smoke test for the surgical study (a human runs this once, before anything is queued).

    .venv/bin/python -m surgical.smoke                                  # default datasets, all three parts
    .venv/bin/python -m surgical.smoke --datasets cifar10 celeba        # only these
    .venv/bin/python -m surgical.smoke --skip-data                      # speed and end-to-end only
    .venv/bin/python -m surgical.smoke --env-only                       # only "0. This server" (seconds, no GPU work)

It needs the whole GPU: run it while no job is running (the speed numbers are wrong otherwise). It queues nothing,
writes no run into runs/, and touches no locked file. First "0. This server": branch, GPU, free disk, the lock, and
for every dataset whether its teacher checkpoint is here (state/*.pt is not in git, so a server set up later has
none and the KD runs in the repository, made with the old teacher, are not reused). Then three parts:

1. data      every dataset is prepared (downloaded if needed) and read back: split sizes against config, label and
             group counts, and for CIFAR-10-C that every split holds the clean set's images in the same order.
2. speed     random-init models on this dataset's real image shapes: images/s of one job per arm, then of N jobs at
             once (N = what fits in VRAM, at most config.MAX_JOBS), and of the DeiT-B teacher fine-tune.
             From these: minutes per run and hours for the 11-arm grid.
3. end to end one 30-step smoke-tier job of `layerroll` per dataset through harness.train (real data, the real
             pretrained student, the dataset's teacher if it exists), into a temporary runs folder.

The report is printed and saved to surgical_smoke_report.md. Time estimates leave out validation (every
config.EVAL_EVERY epochs) and start-up; read them as a lower bound, about 10 % low.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import torch
from timm.loss import SoftTargetCrossEntropy

from harness import config as C
from harness.data import GpuAugment, load_groups, load_split
from harness.models import make_student, make_teacher
from harness.train import Segments, teacher_paths, train_step
from harness.validate import load_selector, selector_source

from .arms import K, REFERENCE, SELECT, teacher_sha, usable

DEFAULT = ("waterbirds", "cifar10", "cifar10c_gaussian_noise", "celeba")
# one job per kind of cost: no teacher, full teacher, one attention block, all blocks but one (the slowest selector)
SPEED_ARMS = ("CE", "Full KD", "last (MaskedKD)", "B2 3-5", "no last 0-10")
SLOWEST = "no last 0-10"
GRID = {**REFERENCE, **SELECT}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _peak(device):
    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None


def _batch(dataset, device, n):
    x, y = load_split("train", dataset)
    n = min(n, len(y)) // 2 * 2   # Mixup pairs the batch
    return torch.as_tensor(x[:n]).to(device), y[:n].to(device)


def worker(dataset, arm, steps, warmup):
    """Images/s of training steps of one arm on this dataset's image shapes (random-init models)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method, sel_args = GRID[arm]
    keep = 0 if method == "ce" else C.NUM_PATCHES if method == "full" else K
    student = make_student(dataset, pretrained=False).to(device)
    teacher = None if method == "ce" else make_teacher(dataset, pretrained=False).to(device).eval()
    selector = None if method == "ce" else load_selector(method, kwargs=sel_args)[0]
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
    aug, crit = GpuAugment(device, C.NUM_CLASSES[dataset]), SoftTargetCrossEntropy()
    xs, ys = _batch(dataset, device, C.BATCH)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(steps + warmup):
        if i == warmup:
            _sync(device)
            t0 = time.time()
        x, y = aug(xs, ys)
        train_step(student, teacher, selector, method, keep, x, y, opt, crit, 0, 0.0, Segments(False))
    _sync(device)
    return {"img_per_s": round(steps * len(ys) / (time.time() - t0), 1), "peak_vram_gib": _peak(device)}


def teacher_worker(dataset, steps, warmup):
    """Images/s of the DeiT-B fine-tune step (harness.finetune_teacher: batch TEACHER_BATCH, no Mixup)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_teacher(dataset, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.TEACHER_LR, weight_decay=C.WEIGHT_DECAY)
    aug, crit = GpuAugment(device, C.NUM_CLASSES[dataset], smoothing=0.0, mixup=False), SoftTargetCrossEntropy()
    xs, ys = _batch(dataset, device, C.TEACHER_BATCH)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(steps + warmup):
        if i == warmup:
            _sync(device)
            t0 = time.time()
        x, y = aug(xs, ys)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = crit(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    _sync(device)
    return {"img_per_s": round(steps * len(ys) / (time.time() - t0), 1), "peak_vram_gib": _peak(device)}


def _sub(args):
    """Run one worker in its own process (clean VRAM numbers; also how the parallel case is measured)."""
    return subprocess.Popen([sys.executable, "-m", "surgical.smoke", "--worker", *[str(a) for a in args]],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=C.ROOT)


def _result(p):
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError((err or out).strip().splitlines()[-1] if (err or out).strip() else f"exit {p.returncode}")
    return json.loads(out.strip().splitlines()[-1])


# ---------------------------------------------------------------- part 0: this server

def _git(*args):
    r = subprocess.run(["git", *args], cwd=C.ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "?"


def environment(datasets):
    from harness import lock
    from harness.report import load_runs
    env = {"branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "commit": _git("rev-parse", "--short", "HEAD"),
           "uncommitted": bool(_git("status", "--porcelain", "--", "harness", "tests", "selectors", "surgical", "analysis")),
           "torch": torch.__version__, "cuda": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "gpu_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else None}
    root = C.DATA_ROOT if C.DATA_ROOT.exists() else C.ROOT
    env["disk_free_gib"] = round(shutil.disk_usage(root).free / 2**30, 1)
    ok, msg = lock.check()
    env["lock"] = ("ok: " if ok else "NOT LOCKED (the scheduler starts no job): ") + msg
    env["scheduler_running"] = subprocess.run(["tmux", "has-session", "-t", "sched"], capture_output=True).returncode == 0 \
        if shutil.which("tmux") else False
    env["agent_loop_running"] = subprocess.run(["tmux", "has-session", "-t", "agent"], capture_output=True).returncode == 0 \
        if shutil.which("tmux") else False
    env["datasets"] = {}
    for d in datasets:
        td = C.TEACHER_SHARED.get(d, d)
        ckpt, meta = teacher_paths(td)
        sha = teacher_sha(td)
        kd = [r for r in load_runs(d) if r["method"] != "ce" and r["tier"] in ("final", "screen")]
        env["datasets"][d] = {
            "prepared": (C.DATA_ROOT / d / "meta.json").exists(),
            "teacher": "here" if sha else ("checkpoint missing (the record is from another server): fine-tune it here"
                                           if meta.exists() and not ckpt.exists() else
                                           "checkpoint does not match its record" if ckpt.exists() else "none yet: fine-tune it"),
            "kd_runs_in_repo": len(kd), "kd_runs_reusable": sum(usable(r, sha) for r in kd)}
    return env


# ---------------------------------------------------------------- part 1: data

def check_data(dataset, workers):
    from harness import prepare_data as P
    rec = {"dataset": dataset}
    fake = bool(os.environ.get("MASKEDKD_FAKE_DATA"))
    t0 = time.time()
    if not fake:
        P.prepare(dataset, workers)   # downloads if needed; SystemExit with the reason if it cannot
    rec["prepare_s"] = round(time.time() - t0, 1)
    rec["sizes"], rec["shape"], rec["gib_on_gpu_per_job"] = {}, None, None
    for split in ("train", "val", "test"):
        x, y = load_split(split, dataset)
        rec["sizes"][split] = len(y)
        if split == "train":
            rec["shape"] = list(x.shape[1:])
            rec["gib_on_gpu_per_job"] = round(x.numel() / 2**30, 2)   # the loader keeps the uint8 train set on the GPU
            rec["classes_seen"] = int(y.unique().numel())
    g = load_groups("val", dataset)
    if g is not None:
        rec["val_group_counts"] = {int(k): int((g == k).sum()) for k in g.unique().tolist()}
    exp = dict(C.EXPECTED_SIZES[dataset])
    exp.pop("partial", None)
    if not fake:
        if "total" in exp:
            ok = sum(rec["sizes"].values()) == exp["total"]
        else:
            ok = all(rec["sizes"].get(k) == v for k, v in exp.items())
        if not ok:
            raise RuntimeError(f"split sizes {rec['sizes']} != config {exp}")
        if rec["classes_seen"] != C.NUM_CLASSES[dataset]:
            raise RuntimeError(f"train has {rec['classes_seen']} classes, config says {C.NUM_CLASSES[dataset]}")
        if dataset in C.CIFAR_C:   # same images as the clean set, in the same order, and actually corrupted
            if not (C.DATA_ROOT / "cifar10" / "meta.json").exists():
                P.prepare("cifar10", workers)
            for split in ("train", "val"):
                (xc, yc), (xn, yn) = load_split(split, "cifar10"), load_split(split, dataset)
                if not torch.equal(yc, yn):
                    raise RuntimeError(f"{split} labels differ from cifar10: the images are not paired")
                d = (xc.float() - xn.float()).abs().mean().item()
                rec[f"{split}_mean_abs_diff_to_clean"] = round(d, 2)
                if d < 1.0:
                    raise RuntimeError(f"{split} images equal the clean ones (mean abs diff {d:.2f}): not corrupted")
    return rec


# ---------------------------------------------------------------- part 2: speed

def measure(dataset, steps, warmup, parallel, gib_data):
    rec = {"single": {}}
    for arm in SPEED_ARMS:
        rec["single"][arm] = _result(_sub([dataset, arm, steps, warmup]))
        print(f"  {dataset} single {arm}: {rec['single'][arm]}", flush=True)
    rec["teacher_finetune"] = _result(_sub([dataset, "__teacher__", steps, warmup]))
    print(f"  {dataset} teacher fine-tune: {rec['teacher_finetune']}", flush=True)
    n = 1
    if parallel and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        per_job = max(r["peak_vram_gib"] for r in rec["single"].values()) + gib_data   # workers hold one batch only
        n = max(1, min(C.MAX_JOBS, int((total - C.VRAM_MARGIN_GIB) // per_job)))
        rec["vram"] = {"gpu_gib": round(total, 1), "per_job_gib_with_train_set": round(per_job, 2), "jobs_that_fit": n}
    if n > 1:
        outs = [_result(p) for p in [_sub([dataset, SLOWEST, steps, warmup]) for _ in range(n)]]
        agg = sum(o["img_per_s"] for o in outs)
        rec["parallel"] = {"jobs": n, "arm": SLOWEST, "aggregate_img_per_s": round(agg, 1),
                           "speedup_over_one_job": round(agg / rec["single"][SLOWEST]["img_per_s"], 2)}
        print(f"  {dataset} x{n} {SLOWEST}: {rec['parallel']}", flush=True)
    return rec


def estimate(dataset, n_train, sp):
    """Minutes per run and hours for the 11-arm grid, from the measured images/s."""
    epochs = C.TIER_EPOCHS[dataset]["final"]
    images = n_train // C.BATCH * C.BATCH * epochs   # drop_last
    par = sp.get("parallel", {"jobs": 1, "speedup_over_one_job": 1.0})
    single = sp["single"]
    # arms not measured cost what the measured arm of the same kind costs
    like = {"random": "last (MaskedKD)", "all 0-11 (rollout)": SLOWEST, "no first 3-11": SLOWEST,
            "B1 0-2": "B2 3-5", "B3 6-8": "B2 3-5", "B4 9-11": "B2 3-5"}
    ips = {arm: single[like.get(arm, arm)]["img_per_s"] for arm in GRID}
    alone_min = {arm: images / v / 60 for arm, v in ips.items()}
    grid_h = sum(images / (v * par["speedup_over_one_job"]) for v in ips.values()) / 3600
    t_epochs = C.TEACHER_EPOCHS_BY.get(dataset, C.TEACHER_EPOCHS)
    return {"epochs": epochs, "images_per_run": images, "jobs_at_once": par["jobs"],
            "min_per_run_alone": {"fastest (CE)": round(alone_min["CE"], 1), "slowest": round(max(alone_min.values()), 1)},
            "min_per_run_shared": round(max(alone_min.values()) * par["jobs"] / par["speedup_over_one_job"], 1),
            "grid_11_arms_1_seed_h": round(grid_h, 2), "grid_11_arms_3_seeds_h": round(3 * grid_h, 2),
            "teacher_finetune_min": round(n_train // C.TEACHER_BATCH * C.TEACHER_BATCH * t_epochs
                                          / sp["teacher_finetune"]["img_per_s"] / 60, 1),
            "teacher_exists": teacher_paths(dataset)[1].exists()}


# ---------------------------------------------------------------- part 3: end to end

def end_to_end(dataset):
    """A smoke-tier layerroll job through harness.train, into a temporary runs folder (nothing is queued)."""
    from harness import train
    src = selector_source("layerroll")
    job = {"id": f"surgical_smoke_{dataset}", "kind": "train", "dataset": dataset, "stage2": None, "method": "layerroll",
           "keep": K, "seed": 0, "tier": "smoke", "epochs": C.TIER_EPOCHS[dataset]["smoke"],
           "sel_args": {"layers": list(range(11))}, "selector_src": src,
           "selector_sha": hashlib.sha256(src.encode()).hexdigest(), "note": "surgical smoke", "by": "smoke",
           "priority": 0, "lr": None}
    tmp = tempfile.mkdtemp(prefix="surgical_smoke_")
    old = C.RUNS
    try:
        C.RUNS = type(old)(tmp)
        jp = os.path.join(tmp, "job.json")
        with open(jp, "w") as fh:
            json.dump(job, fh)
        t0 = time.time()
        rc = train.run(jp)
        m = json.loads((C.RUNS / job["id"] / "metrics.json").read_text())
        return {"returncode": rc, "status": m["status"], "seconds": round(time.time() - t0, 1),
                "mean_kept": m["mean_kept"], "val_acc": m["val_acc_last"], "val_wga": m["val_wga_last"],
                "teacher": "this dataset's" if teacher_paths(dataset)[1].exists() else "random-init (no teacher yet)"}
    finally:
        C.RUNS = old
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- report

def report(res, env):
    out = ["# Surgical smoke report", "",
           f"- {time.strftime('%Y-%m-%d %H:%M:%S %Z')}, fake data: {bool(os.environ.get('MASKEDKD_FAKE_DATA'))}", "",
           "## 0. This server", "",
           f"- branch `{env['branch']}` at {env['commit']}" + (" with uncommitted changes" if env["uncommitted"] else ""),
           f"- GPU: {env['gpu']} ({env['gpu_gib']} GiB), torch {env['torch']}" if env["cuda"] else "- **no GPU visible**",
           f"- free disk: {env['disk_free_gib']} GiB" + ("  ← **under 20 GiB: the downloads may not fit**" if env["disk_free_gib"] < 20 else ""),
           f"- lock: {env['lock']}",
           f"- scheduler: {'running' if env['scheduler_running'] else 'not running (bash agent/start.sh sched)'}"
           + ("; **the agent loop is running: stop it (touch agent/STOP)**" if env["agent_loop_running"] else ""), "",
           "| dataset | prepared | teacher on this server | KD runs in the repo | of those reusable here |", "|---|---|---|---|---|"]
    for d, e in env["datasets"].items():
        out.append(f"| {d} | {'yes' if e['prepared'] else 'no'} | {e['teacher']} | {e['kd_runs_in_repo']} | {e['kd_runs_reusable']} |")
    if not res:
        return "\n".join(out + [""])
    out += ["", "## 1. Data", "", "| dataset | ok | train / val / test | stored shape | train set on GPU per job | prepare | note |",
           "|---|---|---|---|---|---|---|"]
    for d, r in res.items():
        a = r.get("data", {})
        if "error" in a:
            out.append(f"| {d} | **FAIL** | - | - | - | - | {a['error']} |")
            continue
        s = a.get("sizes", {})
        note = "; ".join(f"{k} {v}" for k, v in a.items() if k.endswith("diff_to_clean") or k == "val_group_counts")
        out.append(f"| {d} | ok | {s.get('train')} / {s.get('val')} / {s.get('test')} | {a.get('shape')} | "
                   f"{a.get('gib_on_gpu_per_job')} GiB | {a.get('prepare_s')} s | {note} |")
    out += ["", "## 2. Speed and time", "",
            "| dataset | img/s alone: CE / Full KD / MaskedKD / B2 / no last | jobs at once | min per run (alone, CE–slowest) | "
            "min per run (shared) | 11 arms × 1 seed | × 3 seeds | teacher fine-tune |", "|---|---|---|---|---|---|---|---|"]
    for d, r in res.items():
        sp, e = r.get("speed", {}), r.get("estimate", {})
        if "error" in sp or not e:
            out.append(f"| {d} | {sp.get('error', 'not measured')} | - | - | - | - | - | - |")
            continue
        ips = " / ".join(str(sp["single"][a]["img_per_s"]) for a in SPEED_ARMS)
        t = f"{e['teacher_finetune_min']} min" + (" (exists already)" if e["teacher_exists"] else "")
        out.append(f"| {d} | {ips} | {e['jobs_at_once']} | {e['min_per_run_alone']['fastest (CE)']}–"
                   f"{e['min_per_run_alone']['slowest']} | {e['min_per_run_shared']} | {e['grid_11_arms_1_seed_h']} h | "
                   f"{e['grid_11_arms_3_seeds_h']} h | {t} |")
    tot1 = sum(r["estimate"]["grid_11_arms_1_seed_h"] for r in res.values() if r.get("estimate"))
    tot3 = sum(r["estimate"]["grid_11_arms_3_seeds_h"] for r in res.values() if r.get("estimate"))
    tt = sum(r["estimate"]["teacher_finetune_min"] for r in res.values() if r.get("estimate") and not r["estimate"]["teacher_exists"])
    out += ["", f"- All datasets above, 11 arms: **{tot1:.1f} h** for one seed, **{tot3:.1f} h** for three "
                f"(runs that already exist are reused, so the real number is lower). Teachers still to fine-tune: {tt:.0f} min.",
            "- Validation and start-up are not in these numbers (about +10 %).", ""]
    for d, r in res.items():
        v = r.get("speed", {}).get("vram")
        if v:
            out.append(f"- {d} VRAM: {v['per_job_gib_with_train_set']} GiB per job with its train set, "
                       f"{v['jobs_that_fit']} job(s) fit in {v['gpu_gib']} GiB")
    out += ["", "## 3. End to end (30 training steps, real data)", "", "| dataset | result | seconds | kept tokens | teacher |",
            "|---|---|---|---|---|"]
    for d, r in res.items():
        e = r.get("e2e", {})
        if "error" in e:
            out.append(f"| {d} | **FAIL**: {e['error']} | - | - | - |")
        elif e:
            out.append(f"| {d} | {e['status']} | {e['seconds']} | {e['mean_kept']} | {e['teacher']} |")
    fails = [f"{d}: {part}" for d, r in res.items() for part in ("data", "speed", "e2e") if "error" in r.get(part, {})]
    out += ["", "## Verdict", "", ("**FAIL** — " + "; ".join(fails)) if fails else "**all parts passed**", ""]
    return "\n".join(out)


def _try(fn, *a):
    try:
        return fn(*a)
    except SystemExit as e:
        return {"error": str(e) or "stopped (SystemExit)"}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT), choices=list(C.DATASETS))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--no-parallel", action="store_true")
    ap.add_argument("--env-only", action="store_true", help="only part 0: what this server has")
    ap.add_argument("--worker", nargs=4, metavar=("DATASET", "ARM", "STEPS", "WARMUP"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        d, arm, steps, warm = a.worker
        r = teacher_worker(d, int(steps), int(warm)) if arm == "__teacher__" else worker(d, arm, int(steps), int(warm))
        print(json.dumps(r))
        return 0
    import faulthandler
    faulthandler.enable()
    env = environment(a.datasets)
    if a.env_only:
        print(report({}, env))
        return 0
    if any(C.QUEUE.glob("running/*.json")):
        print("WARNING: jobs are running on this GPU; the speed numbers below will be too low", flush=True)
    res = {}
    for d in a.datasets:
        print(f"== {d}", flush=True)
        r = res.setdefault(d, {})
        if not a.skip_data:
            r["data"] = _try(check_data, d, a.workers)
            print(f"  data: {r['data']}", flush=True)
            if "error" in r["data"]:
                continue
        try:
            x, y = load_split("train", d)
            n_train, gib = len(y), x.numel() / 2**30
        except Exception as e:
            r["speed"] = {"error": f"dataset not prepared: {type(e).__name__}: {e}"}
            continue
        if not a.skip_speed:
            r["speed"] = _try(measure, d, a.steps, a.warmup, not a.no_parallel, gib)
            if "error" not in r["speed"]:
                r["estimate"] = estimate(d, n_train, r["speed"])
        if not a.skip_e2e:
            r["e2e"] = _try(end_to_end, d)
            print(f"  end to end: {r['e2e']}", flush=True)
    text = report(res, environment(a.datasets))   # again: the datasets are prepared now
    (C.ROOT / "surgical_smoke_report.md").write_text(text + "\n")
    (C.ROOT / "surgical_smoke_report.json").write_text(json.dumps(res, indent=1) + "\n")
    print("\n" + text)
    return 1 if "**FAIL**" in text else 0


if __name__ == "__main__":
    sys.exit(main())
