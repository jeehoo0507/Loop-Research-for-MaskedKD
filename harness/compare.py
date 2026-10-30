"""Paired comparison across seeds (final tier): method A vs method B, val acc difference per seed.

    python -m harness.compare --a my_sel:25 --b maskedkd:98 [--tier final]

Prints per-seed differences, mean, and a 95% t-interval (n-1 dof). With 3 seeds the interval is wide;
report it as is.
"""
import argparse
import statistics

from . import config as C
from .report import by_seed, load_runs, paired   # one paired-CI implementation, shared with RESULTS_TABLE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tier", default="final")
    ap.add_argument("--dataset", default=C.STAGE1, choices=list(C.DATASETS))
    a = ap.parse_args()
    runs = load_runs(a.dataset)
    A, B = (by_seed(runs, *spec.rsplit(":", 1), a.tier) for spec in (a.a, a.b))
    p = paired(A, B)
    if p is None:
        raise SystemExit(f"need >=2 shared seeds, have {sorted(set(A) & set(B))}")
    seeds, m = p["seeds"], p["diff"]
    fa = statistics.mean(A[s]["flops"]["total_gflops_per_img"] for s in seeds)
    fb = statistics.mean(B[s]["flops"]["total_gflops_per_img"] for s in seeds)
    for s in seeds:
        print(f"seed {s}: A {A[s]['val_acc_last']:.2f} ({A[s]['id']})  B {B[s]['val_acc_last']:.2f} ({B[s]['id']})")
    print(f"val acc A-B = {m:+.2f} %p, 95% CI [{p['lo']:+.2f}, {p['hi']:+.2f}] (n={len(seeds)})")
    print(f"train-step GFLOPs/img: A {fa:.2f}  B {fb:.2f}  ratio A/B {fa / fb:.3f}")
    ia = [A[s].get("img_per_s_mean") for s in seeds]
    ib = [B[s].get("img_per_s_mean") for s in seeds]
    if all(ia) and all(ib):
        ma, mb = statistics.mean(ia), statistics.mean(ib)
        print(f"train throughput img/s: A {ma:.0f}  B {mb:.0f}  ratio A/B {ma / mb:.3f} (depends on GPU sharing)")


if __name__ == "__main__":
    main()
