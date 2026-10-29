"""Paired comparison across seeds (final tier): method A vs method B, val acc difference per seed.

    python -m harness.compare --a my_sel:25 --b maskedkd:98 [--tier final]

Prints per-seed differences, mean, and a 95% t-interval (n-1 dof). With 3 seeds the interval is wide;
report it as is.
"""
import argparse
import statistics

from . import config as C
from .report import load_runs

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def pick(runs, spec, tier):
    method, keep = spec.rsplit(":", 1)
    out = {}
    for r in sorted(runs, key=lambda r: r["id"]):
        if r["tier"] == tier and r["method"] == method and float(r["keep"]) == float(keep):
            out[r["seed"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tier", default="final")
    ap.add_argument("--dataset", default="cifar100", choices=list(C.DATASETS))
    a = ap.parse_args()
    runs = load_runs(a.dataset)
    A, B = pick(runs, a.a, a.tier), pick(runs, a.b, a.tier)
    seeds = sorted(set(A) & set(B))
    if len(seeds) < 2:
        raise SystemExit(f"need >=2 shared seeds, have {seeds}")
    d = [A[s]["val_acc_last"] - B[s]["val_acc_last"] for s in seeds]
    m, sd = statistics.mean(d), statistics.stdev(d)
    half = T95[len(d) - 1] * sd / len(d) ** 0.5
    fa = statistics.mean(A[s]["flops"]["total_gflops_per_img"] for s in seeds)
    fb = statistics.mean(B[s]["flops"]["total_gflops_per_img"] for s in seeds)
    for s in seeds:
        print(f"seed {s}: A {A[s]['val_acc_last']:.2f} ({A[s]['id']})  B {B[s]['val_acc_last']:.2f} ({B[s]['id']})")
    print(f"val acc A-B = {m:+.2f} %p, 95% CI [{m - half:+.2f}, {m + half:+.2f}] (n={len(d)})")
    print(f"train-step GFLOPs/img: A {fa:.2f}  B {fb:.2f}  ratio A/B {fa / fb:.3f}")
    ia = [A[s].get("img_per_s_mean") for s in seeds]
    ib = [B[s].get("img_per_s_mean") for s in seeds]
    if all(ia) and all(ib):
        ma, mb = statistics.mean(ia), statistics.mean(ib)
        print(f"train throughput img/s: A {ma:.0f}  B {mb:.0f}  ratio A/B {ma / mb:.3f} (depends on GPU sharing)")


if __name__ == "__main__":
    main()
