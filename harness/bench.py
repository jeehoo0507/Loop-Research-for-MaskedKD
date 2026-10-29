"""Measure speed and VRAM on this GPU, to set TIER_EPOCHS and the scheduler's concurrency.

    python -m harness.bench               # writes state/bench.json and prints a recommendation

Uses random-init models on real batch shapes (accuracy is irrelevant here).
"""
import argparse
import json
import subprocess
import sys
import time

import torch
from timm.loss import SoftTargetCrossEntropy

from . import config as C
from .baselines import BUILTIN
from .data import GpuAugment
from .models import make_student, make_teacher
from .train import train_step, Segments

CASES = (("ce", 0), ("full", 196), ("maskedkd", 98), ("maskedkd", 25), ("tome", 49))


def flop_check(device):
    """The cost axis of every result rests on FlopCounterMode. Check it on this GPU against the published
    DeiT-S cost (4.6 GMACs at 197 tokens = 9.2 GFLOPs per image, forward)."""
    from torch.utils.flop_counter import FlopCounterMode
    from .models import teacher_forward_keepmask
    t = make_teacher(pretrained=False).to(device).eval()
    x = torch.randn(4, 3, C.IMG_SIZE, C.IMG_SIZE, device=device)
    keep = torch.ones(4, C.NUM_PATCHES, dtype=torch.bool, device=device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16), FlopCounterMode(display=False) as fc:
        teacher_forward_keepmask(t, x, keep)
    got = fc.get_total_flops() / 4 / 1e9
    err = abs(got - C.TEACHER_REF_GFLOPS) / C.TEACHER_REF_GFLOPS
    return {"teacher_gflops_measured": round(got, 3), "reference": C.TEACHER_REF_GFLOPS, "rel_err": round(err, 4),
            "ok": err <= C.FLOP_CHECK_TOL}


def worker(method, keep, steps):
    device = torch.device("cuda")
    student = make_student().to(device)
    teacher = None if method == "ce" else make_teacher(pretrained=False).to(device).eval()
    selector = BUILTIN[method]() if method in BUILTIN else None
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
    aug = GpuAugment(device)
    crit = SoftTargetCrossEntropy()
    xs = torch.randint(0, 256, (C.BATCH, 3, 32, 32), dtype=torch.uint8, device=device)
    ys = torch.randint(0, C.NUM_CLASSES, (C.BATCH,), device=device)
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 5):
        if i == 5:
            torch.cuda.synchronize()
            t0 = time.time()
        x, y = aug(xs, ys)
        train_step(student, teacher, selector, method, keep, x, y, opt, crit, 0, 0.0, Segments(False))
    torch.cuda.synchronize()
    dt = time.time() - t0
    return {"img_per_s": round(steps * C.BATCH / dt, 1), "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=3)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--max-par", type=int, default=C.MAX_JOBS)
    a = ap.parse_args()
    if a.worker:
        print(json.dumps(worker(a.worker[0], float(a.worker[1]), int(a.worker[2]))))
        return
    if not torch.cuda.is_available():
        sys.exit("bench needs a GPU")
    res = {"gpu": torch.cuda.get_device_name(0), "batch": C.BATCH, "single": {}, "parallel": {}}
    res["flop_check"] = flop_check(torch.device("cuda"))
    print(f"flop check: {res['flop_check']}", flush=True)
    for m, k in CASES:
        r = worker(m, k, a.steps)
        res["single"][f"{m}@{k}"] = r
        print(f"single {m}@{k}: {r}", flush=True)
        torch.cuda.empty_cache()
    for n in range(2, a.max_par + 1):
        ps = [subprocess.Popen([sys.executable, "-m", "harness.bench", "--worker", "maskedkd", "98", str(a.steps)],
                               stdout=subprocess.PIPE, text=True, cwd=C.ROOT) for _ in range(n)]
        outs = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in ps]
        agg = sum(o["img_per_s"] for o in outs)
        res["parallel"][str(n)] = {"aggregate_img_per_s": round(agg, 1), "per_job": outs}
        print(f"parallel x{n} maskedkd@98: aggregate {agg:.0f} img/s", flush=True)
    n_train = 50000 - C.VAL_PER_CLASS * C.NUM_CLASSES
    worst = min(r["img_per_s"] for r in res["single"].values())
    best_par = max([(1, worst)] + [(int(n), v["aggregate_img_per_s"] / int(n)) for n, v in res["parallel"].items()],
                   key=lambda t: t[0] * t[1])
    ep_s = n_train / worst
    res["recommend"] = {
        "epoch_s_slowest_single": round(ep_s, 1),
        "screen_epochs_for_1h_single": int(3600 / ep_s),
        "parallel_jobs": best_par[0],
        "epoch_s_per_job_at_that_parallelism": round(n_train / best_par[1], 1),
        "screen_epochs_for_1h_at_that_parallelism": int(3600 / (n_train / best_par[1])),
    }
    C.STATE.mkdir(exist_ok=True)
    (C.STATE / "bench.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["recommend"], indent=1))
    if not res["flop_check"]["ok"]:
        print(f"FLOP CHECK FAILED: measured {res['flop_check']['teacher_gflops_measured']} GFLOPs vs "
              f"{C.TEACHER_REF_GFLOPS} (tolerance {C.FLOP_CHECK_TOL:.0%}). Do not relock; fix the counter first.")
        sys.exit(2)


if __name__ == "__main__":
    main()
